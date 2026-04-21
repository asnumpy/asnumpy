/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#pragma once

#include "asnumpy/utils/acl_resource.hpp"
#include "asnumpy/utils/npu_array.hpp"
#include "asnumpy/utils/status_handler.hpp"
#include <aclnn/aclnn_base.h>
#include <fmt/format.h>
#include <functional>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace asnumpy {

namespace detail {

// Format a shape vector as "1x2x3" for logging
inline std::string FormatShape(const std::vector<int64_t>& shape) {
    std::string result;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += 'x';
        result += std::to_string(shape[i]);
    }
    return result.empty() ? "()" : result;
}

} // namespace detail

/**
 * @brief Generic unary operator execution template
 *
 * This template function encapsulates the standard six-step CANN operation pattern
 * for unary operators, providing automatic resource management, error handling,
 * and logging.
 *
 * @tparam GetWorkspaceSizeFunc Type of the function to get workspace size
 * @tparam ExecuteFunc Type of the function to execute the operation
 * @param input Input array
 * @param dtype Output data type (optional, defaults to input dtype)
 * @param get_workspace_size_func Function to get workspace size and executor
 * @param execute_func Function to execute the operator
 * @param op_name Operator name (for logging and error messages)
 * @param aclnn_api ACLNN API name (e.g., "aclnnAbs")
 * @param src_file Source file path (auto-captured via EXECUTE_UNARY_OP macro)
 * @param src_func Source function name (auto-captured via EXECUTE_UNARY_OP macro)
 * @return NPUArray Output array
 */
template<typename GetWorkspaceSizeFunc, typename ExecuteFunc>
NPUArray ExecuteUnaryOp(
    const NPUArray& input,
    std::optional<py::dtype> dtype,
    GetWorkspaceSizeFunc&& get_workspace_size_func,
    ExecuteFunc&& execute_func,
    const std::string& op_name,
    const std::string& aclnn_api,
    const char* src_file,
    const char* src_func
) {
    spdlog::debug("[{}]({}) {} start: input_shape={}, tensorSize={}, aclDtype={}",
                  detail::LogBasename(src_file), src_func, aclnn_api,
                  detail::FormatShape(input.shape),
                  input.tensorSize, AclDtypeName(input.aclDtype));

    // Determine output type and shape
    auto out = NPUArray(input.shape, dtype.value());

    // Get workspace size and executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = std::invoke(get_workspace_size_func,
                             input.tensorPtr, out.tensorPtr,
                             &workspaceSize, &executor);
    CheckAclnnStatus(error, src_file, src_func, aclnn_api + "GetWorkspaceSize");

    // RAII management of workspace
    AclWorkspace workspace(workspaceSize);

    // Execute operation
    error = std::invoke(execute_func, workspace.get(), workspaceSize,
                       executor, nullptr);
    CheckAclnnStatus(error, src_file, src_func, aclnn_api);

    // Synchronize device
    error = aclrtSynchronizeDevice();
    CheckAclRuntimeStatus(error, src_file, src_func, aclnn_api + ": aclrtSynchronizeDevice");

    spdlog::info("[{}]({}) {} completed",
                 detail::LogBasename(src_file), src_func, aclnn_api);

    // All resources automatically freed by RAII
    return out;
}

/**
 * @brief Generic binary operator execution template
 *
 * This template function encapsulates the standard six-step CANN operation pattern
 * for binary operators, providing automatic resource management, error handling,
 * and logging. It automatically handles broadcasting between the two input arrays.
 *
 * @tparam GetWorkspaceSizeFunc Type of the function to get workspace size
 * @tparam ExecuteFunc Type of the function to execute the operation
 * @param x1 First input array
 * @param x2 Second input array
 * @param dtype Output data type (optional, defaults to x1 dtype)
 * @param get_workspace_size_func Function to get workspace size and executor
 * @param execute_func Function to execute the operator
 * @param op_name Operator name (for logging and error messages)
 * @param aclnn_api ACLNN API name (e.g., "aclnnAdd")
 * @param src_file Source file path (auto-captured via EXECUTE_BINARY_OP macro)
 * @param src_func Source function name (auto-captured via EXECUTE_BINARY_OP macro)
 * @return NPUArray Output array
 */
template<typename GetWorkspaceSizeFunc, typename ExecuteFunc>
NPUArray ExecuteBinaryOp(
    const NPUArray& x1,
    const NPUArray& x2,
    std::optional<py::dtype> dtype,
    GetWorkspaceSizeFunc&& get_workspace_size_func,
    ExecuteFunc&& execute_func,
    const std::string& op_name,
    const std::string& aclnn_api,
    const char* src_file,
    const char* src_func
) {
    spdlog::debug("[{}]({}) {} start: x1_shape={}, x2_shape={}, aclDtype={}",
                  detail::LogBasename(src_file), src_func, aclnn_api,
                  detail::FormatShape(x1.shape),
                  detail::FormatShape(x2.shape), AclDtypeName(x1.aclDtype));

    // Determine output shape and type
    auto out_shape = GetBroadcastShape(x1, x2);
    auto out = NPUArray(out_shape, dtype.value());

    // Get workspace size and executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = std::invoke(get_workspace_size_func,
                             x1.tensorPtr, x2.tensorPtr, out.tensorPtr,
                             &workspaceSize, &executor);
    CheckAclnnStatus(error, src_file, src_func, aclnn_api + "GetWorkspaceSize");

    // RAII management of workspace
    AclWorkspace workspace(workspaceSize);

    // Execute operation
    error = std::invoke(execute_func, workspace.get(), workspaceSize,
                       executor, nullptr);
    CheckAclnnStatus(error, src_file, src_func, aclnn_api);

    // Synchronize device
    error = aclrtSynchronizeDevice();
    CheckAclRuntimeStatus(error, src_file, src_func, aclnn_api + ": aclrtSynchronizeDevice");

    spdlog::info("[{}]({}) {} completed",
                 detail::LogBasename(src_file), src_func, aclnn_api);

    // All resources automatically freed by RAII
    return out;
}

} // namespace asnumpy

// Wrapper macros - automatically capture source location at call site
#define EXECUTE_UNARY_OP(input, dtype, get_ws, exec, name, aclnn_api) \
    ::asnumpy::ExecuteUnaryOp(input, dtype, get_ws, exec, name, aclnn_api, __FILE__, __func__)

#define EXECUTE_BINARY_OP(x1, x2, dtype, get_ws, exec, name, aclnn_api) \
    ::asnumpy::ExecuteBinaryOp(x1, x2, dtype, get_ws, exec, name, aclnn_api, __FILE__, __func__)

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
#include <chrono>
#include <functional>
#include <optional>

namespace asnumpy {

/**
 * @brief Generic unary operator execution template
 *
 * This template function encapsulates the standard six-step CANN operation pattern
 * for unary operators, providing automatic resource management, error handling,
 * logging, and performance measurement.
 *
 * @tparam GetWorkspaceSizeFunc Type of the function to get workspace size
 * @tparam ExecuteFunc Type of the function to execute the operation
 * @param input Input array
 * @param dtype Output data type (optional, defaults to input dtype)
 * @param get_workspace_size_func Function to get workspace size and executor
 * @param execute_func Function to execute the operator
 * @param op_name Operator name (for logging and error messages)
 * @return NPUArray Output array
 *
 * Example usage:
 * @code
 * NPUArray Sin(const NPUArray& x) {
 *     ......
 *     return ExecuteUnaryOp(
 *         x,
 *         std::nullopt,
 *         [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
 *             return aclnnSinGetWorkspaceSize(in, out, workspaceSize, executor);
 *         },
 *         [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
 *             return aclnnSin(workspace, workspaceSize, executor, nullptr);
 *         },
 *         "Sin"
 *     );
 * }
 * @endcode
 */
template<typename GetWorkspaceSizeFunc, typename ExecuteFunc>
NPUArray ExecuteUnaryOp(
    const NPUArray& input,
    std::optional<py::dtype> dtype,
    GetWorkspaceSizeFunc&& get_workspace_size_func,
    ExecuteFunc&& execute_func,
    const std::string& op_name
) {
    auto start = std::chrono::high_resolution_clock::now();

    // Determine output type and shape
    auto out = NPUArray(input.shape, dtype.value());

    // Get workspace size and executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = std::invoke(get_workspace_size_func,
                             input.tensorPtr, out.tensorPtr,
                             &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);

    // RAII management of executor and workspace
    AclWorkspace workspace(workspaceSize);

    // Execute operation
    error = std::invoke(execute_func, workspace.get(), workspaceSize,
                       executor, nullptr);
    CheckExecuteAclnnStatus(error, op_name);

    // Synchronize device
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    // Calculate elapsed time
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start
    ).count();

    // All resources automatically freed by RAII
    return out;
}

/**
 * @brief Generic binary operator execution template
 *
 * This template function encapsulates the standard six-step CANN operation pattern
 * for binary operators, providing automatic resource management, error handling,
 * logging, and performance measurement. It automatically handles broadcasting
 * between the two input arrays.
 *
 * @tparam GetWorkspaceSizeFunc Type of the function to get workspace size
 * @tparam ExecuteFunc Type of the function to execute the operation
 * @param x1 First input array
 * @param x2 Second input array
 * @param dtype Output data type (optional, defaults to x1 dtype)
 * @param get_workspace_size_func Function to get workspace size and executor
 * @param execute_func Function to execute the operator
 * @param op_name Operator name (for logging and error messages)
 * @return NPUArray Output array
 *
 * Example usage:
 * @code
 * NPUArray Add(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
 *     ......
 *     return ExecuteBinaryOp(
 *         x1, x2, dtype,
 *         [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
 *             return aclnnAddGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
 *         },
 *         [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
 *             return aclnnAdd(workspace, workspaceSize, executor, nullptr);
 *         },
 *         "Add"
 *     );
 * }
 * @endcode
 */
template<typename GetWorkspaceSizeFunc, typename ExecuteFunc>
NPUArray ExecuteBinaryOp(
    const NPUArray& x1,
    const NPUArray& x2,
    std::optional<py::dtype> dtype,
    GetWorkspaceSizeFunc&& get_workspace_size_func,
    ExecuteFunc&& execute_func,
    const std::string& op_name
) {
    auto start = std::chrono::high_resolution_clock::now();

    // Determine output shape and type
    auto out_shape = GetBroadcastShape(x1, x2);
    auto out = NPUArray(out_shape, dtype.value());

    // Get workspace size and executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = std::invoke(get_workspace_size_func,
                             x1.tensorPtr, x2.tensorPtr, out.tensorPtr,
                             &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);

    // RAII management of executor and workspace
    AclWorkspace workspace(workspaceSize);

    // Execute operation
    error = std::invoke(execute_func, workspace.get(), workspaceSize,
                       executor, nullptr);
    CheckExecuteAclnnStatus(error, op_name);

    // Synchronize device
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    // Calculate elapsed time
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start
    ).count();

    // All resources automatically freed by RAII
    return out;
}

} // namespace asnumpy

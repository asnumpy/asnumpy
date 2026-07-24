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

#include <asnumpy/math/hyperbolic_functions.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/dtype_promotion.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_acosh.h>
#include <aclnnop/aclnn_asinh.h>
#include <aclnnop/aclnn_atanh.h>
#include <aclnnop/aclnn_cosh.h>
#include <aclnnop/aclnn_sinh.h>
#include <aclnnop/aclnn_tanh.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

namespace {

template <typename GetWs, typename Exec>
NPUArray HyperbolicOp(const NPUArray& x, std::optional<py::dtype> dtype, bool supports_float64, GetWs&& get_ws,
                      Exec&& exec, const char* op_name, const char* api_name) {
    aclDataType desired = PromoteUnaryFloating(x.aclDtype);
    ACL_DTYPE_WARN(x.aclDtype, desired, op_name);
    py::dtype out_py_dtype = NPUArray::GetPyDtype(desired);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        desired = NPUArray::GetACLDataType(out_py_dtype);
    }
    aclDataType compute = AclComputeFloatingDtype(desired, supports_float64);
    NPUArray input = EnsureAclDtype(x, compute);
    NPUArray out = EXECUTE_UNARY_OP(input, NPUArray::GetPyDtype(compute), std::forward<GetWs>(get_ws),
                                    std::forward<Exec>(exec), op_name, api_name);
    if (desired != compute) {
        return CastToDtype(out, desired);
    }
    return out;
}

} // namespace

NPUArray Sinh(const NPUArray& x, std::optional<py::dtype> dtype) {
    return HyperbolicOp(
        x, dtype, /*supports_float64=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnSinhGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnSinh(workspace, workspaceSize, executor, nullptr);
        },
        "Sinh", "aclnnSinh");
}

NPUArray Cosh(const NPUArray& x, std::optional<py::dtype> dtype) {
    return HyperbolicOp(
        x, dtype, /*supports_float64=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnCoshGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnCosh(workspace, workspaceSize, executor, nullptr);
        },
        "Cosh", "aclnnCosh");
}

NPUArray Tanh(const NPUArray& x, std::optional<py::dtype> dtype) {
    return HyperbolicOp(
        x, dtype, /*supports_float64=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnTanhGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnTanh(workspace, workspaceSize, executor, nullptr);
        },
        "Tanh", "aclnnTanh");
}

NPUArray Arcsinh(const NPUArray& x, std::optional<py::dtype> dtype) {
    return HyperbolicOp(
        x, dtype, /*supports_float64=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAsinhGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAsinh(workspace, workspaceSize, executor, nullptr);
        },
        "Arcsinh", "aclnnAsinh");
}

NPUArray Arccosh(const NPUArray& x, std::optional<py::dtype> dtype) {
    return HyperbolicOp(
        x, dtype, /*supports_float64=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAcoshGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAcosh(workspace, workspaceSize, executor, nullptr);
        },
        "Arccosh", "aclnnAcosh");
}

NPUArray Arctanh(const NPUArray& x, std::optional<py::dtype> dtype) {
    return HyperbolicOp(
        x, dtype, /*supports_float64=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAtanhGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAtanh(workspace, workspaceSize, executor, nullptr);
        },
        "Arctanh", "aclnnAtanh");
}

} // namespace asnumpy

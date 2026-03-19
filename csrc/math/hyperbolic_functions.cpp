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

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_sinh.h>
#include <aclnnop/aclnn_cosh.h>
#include <aclnnop/aclnn_tanh.h>
#include <aclnnop/aclnn_asinh.h>
#include <aclnnop/aclnn_acosh.h>
#include <aclnnop/aclnn_atanh.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

NPUArray Sinh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    ACL_DTYPE_WARN(in_dtype, out_dtype, __func__);
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    return EXECUTE_UNARY_OP(
        x,
        out_py_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnSinhGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnSinh(workspace, workspaceSize, executor, nullptr);
        },
        "Sinh",
        "aclnnSinh"
    );
}


NPUArray Cosh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    ACL_DTYPE_WARN(in_dtype, out_dtype, __func__);
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    return EXECUTE_UNARY_OP(
        x,
        out_py_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnCoshGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnCosh(workspace, workspaceSize, executor, nullptr);
        },
        "Cosh",
        "aclnnCosh"
    );
}


NPUArray Tanh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    ACL_DTYPE_WARN(in_dtype, out_dtype, __func__);
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    return EXECUTE_UNARY_OP(
        x,
        out_py_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnTanhGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnTanh(workspace, workspaceSize, executor, nullptr);
        },
        "Tanh",
        "aclnnTanh"
    );
}


NPUArray Arcsinh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    ACL_DTYPE_WARN(in_dtype, out_dtype, __func__);
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    return EXECUTE_UNARY_OP(
        x,
        out_py_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAsinhGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAsinh(workspace, workspaceSize, executor, nullptr);
        },
        "Arcsinh",
        "aclnnAsinh"
    );
}


NPUArray Arccosh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    ACL_DTYPE_WARN(in_dtype, out_dtype, __func__);
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    return EXECUTE_UNARY_OP(
        x,
        out_py_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAcoshGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAcosh(workspace, workspaceSize, executor, nullptr);
        },
        "Arccosh",
        "aclnnAcosh"
    );
}


NPUArray Arctanh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    ACL_DTYPE_WARN(in_dtype, out_dtype, __func__);
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    return EXECUTE_UNARY_OP(
        x,
        out_py_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAtanhGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAtanh(workspace, workspaceSize, executor, nullptr);
        },
        "Arctanh",
        "aclnnAtanh"
    );
}

}
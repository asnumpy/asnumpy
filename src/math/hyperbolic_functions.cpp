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

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_sinh.h>
#include <aclnnop/aclnn_cosh.h>
#include <aclnnop/aclnn_tanh.h>
#include <aclnnop/aclnn_asinh.h>
#include <aclnnop/aclnn_acosh.h>
#include <aclnnop/aclnn_atanh.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

NPUArray Sinh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray result(shape, out_py_dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSinhGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Sinh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Sinh: malloc workspace failed, error={}", error));
        }
    }

    // 执行双曲正弦计算
    error = aclnnSinh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Sinh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


NPUArray Cosh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray result(shape, out_py_dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnCoshGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Cosh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Cosh: malloc workspace failed, error={}", error));
        }
    }

    // 执行双曲余弦计算
    error = aclnnCosh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Cosh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


NPUArray Tanh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray result(shape, out_py_dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnTanhGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Tanh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Tanh: malloc workspace failed, error={}", error));
        }
    }

    // 执行双曲正切计算
    error = aclnnTanh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Tanh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


NPUArray Arcsinh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray result(shape, out_py_dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAsinhGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arcsinh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Arcsinh: malloc workspace failed, error={}", error));
        }
    }

    // 执行反双曲正弦计算
    error = aclnnAsinh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arcsinh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


NPUArray Arccosh(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    if (in_dtype == ACL_INT8  || in_dtype == ACL_INT16 ||
        in_dtype == ACL_INT32 || in_dtype == ACL_INT64 ||
        in_dtype == ACL_UINT8 || in_dtype == ACL_BOOL) {
        out_dtype = ACL_FLOAT;  // 默认转 float32
    }
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray result(shape, out_py_dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAcoshGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arccosh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Arccosh: malloc workspace failed, error={}", error));
        }
    }

    // 执行反双曲余弦计算
    error = aclnnAcosh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arccosh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
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
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray result(x.shape, out_py_dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAtanhGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arctanh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Arctanh: malloc workspace failed, error={}", error));
        }
    }

    // 执行反双曲正切计算
    error = aclnnAtanh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arctanh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}

}
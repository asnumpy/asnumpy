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


#include <asnumpy/math/rounding.hpp>
#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_trunc.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_ceil.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

NPUArray Around(const NPUArray& x, int decimals, std::optional<py::dtype> dtype) {
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray out(shape, out_py_dtype);

    if (out.tensorPtr == nullptr) {
        throw std::runtime_error("[math.cpp](around) out.tensorPtr is null, failed to allocate output tensor");
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnRoundDecimalsGetWorkspaceSize(
        x.tensorPtr, decimals, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](around) aclnnRoundDecimalsGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](around) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](around) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    aclrtStream stream = nullptr;
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[math.cpp](around) Failed to get current stream");
    }

    error = aclnnRoundDecimals(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](around) aclnnRoundDecimals error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](around) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}


NPUArray Round_(const NPUArray& x, int decimals, std::optional<py::dtype> dtype) {
    return Around(x, decimals, dtype);
}


NPUArray Rint(const NPUArray& x, std::optional<py::dtype> dtype) {
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray out(shape, out_py_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnRoundGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](rint) aclnnRoundGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](rint) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](rint) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnRound(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](rint) aclnnRound error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](rint) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}


NPUArray Fix(const NPUArray& x, std::optional<py::dtype> dtype) {
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray out(shape, out_py_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnTruncGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](fix) aclnnTruncGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](fix) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](fix) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnTrunc(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](fix) aclnnTrunc error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](fix) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}


NPUArray Floor(const NPUArray& x, std::optional<py::dtype> dtype) {
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray out(shape, out_py_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnFloorGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](floor) aclnnFloorGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](floor) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](floor) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnFloor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](floor) aclnnFloor error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](floor) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}


NPUArray Ceil(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
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
    auto error = aclnnCeilGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Ceil: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Ceil: malloc workspace failed, error={}", error));
        }
    }

    // 执行向上取整计算
    error = aclnnCeil(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Ceil: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


NPUArray Trunc(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
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
    auto error = aclnnTruncGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Trunc: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Trunc: malloc workspace failed, error={}", error));
        }
    }

    // 执行截断计算（保留整数部分，去除小数）
    error = aclnnTrunc(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Trunc: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}

}
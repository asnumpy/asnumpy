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


#include <asnumpy/math/extrema_finding.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/npu_ops_macros.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <aclnnop/aclnn_maximum.h>
#include <aclnnop/aclnn_minimum.h>
#include <aclnnop/aclnn_amax.h>
#include <aclnnop/aclnn_max.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_amin.h>
#include <aclnnop/aclnn_min.h>

#include <cstdint>
#include <fmt/base.h>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>

namespace asnumpy {

/**
 * @brief Element-wise maximum of two arrays.
 *
 * Creates an output array on NPU and computes element-wise max(x1, x2)
 * using the aclnnMaximum operator.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise maxima.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Maximum(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_dtype = x1.dtype;
    auto acl_dtype = x1.aclDtype;
    auto shape = GetBroadcastShape(x1, x2);
    auto temp = NPUArray::GetACLDataType(out_dtype);
    if (temp == ACL_INT16 || temp == ACL_INT32 || temp == ACL_INT64) {
        out_dtype = x2.dtype;
    }
    if (dtype != std::nullopt) {
        out_dtype = *dtype;
    }
    auto out = NPUArray(shape, out_dtype);

    // 4. 获取工作空间大小和 executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMaximumGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[Maximum] aclnnMaximumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[Maximum] Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    // 5. 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[Maximum] aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) {
                error_msg += " - " + std::string(detailed_msg);
            }
            throw std::runtime_error(error_msg);
        }
    }

    // 6. 执行 Maximum 操作
    error = aclnnMaximum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[Maximum] aclnnMaximum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(error_msg);
    }

    // 7. 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[Maximum] aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(error_msg);
    }

    // 8. 释放 workspace
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    // 9. 返回输出
    return out;
}


/**
 * @brief Element-wise minimum of two arrays.
 *
 * Creates an output array on NPU and computes element-wise min(x1, x2)
 * using the aclnnMinimum operator.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise minima.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Minimum(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_dtype = x1.dtype;
    auto acl_dtype = x1.aclDtype;
    auto shape = GetBroadcastShape(x1, x2);
    auto temp = NPUArray::GetACLDataType(out_dtype);
    if (temp == ACL_INT16 || temp == ACL_INT32 || temp == ACL_INT64) {
        out_dtype = x2.dtype;
    }
    if (dtype != std::nullopt) {
        out_dtype = *dtype;
    }
    auto out = NPUArray(shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMinimumGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](minimum) aclnnMinimumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[miscellaneous.cpp](minimum) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](minimum) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) {
                error_msg += " - " + std::string(detailed_msg);
            }
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMinimum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](minimum) aclnnMinimum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](minimum) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

NPUArray Fmax(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_dtype = x1.dtype;
    auto acl_dtype = x1.aclDtype;
    auto shape = GetBroadcastShape(x1, x2);
    auto temp = NPUArray::GetACLDataType(out_dtype);
    if (temp == ACL_INT16 || temp == ACL_INT32 || temp == ACL_INT64) {
        out_dtype = x2.dtype;
    }
    if (dtype != std::nullopt) {
        out_dtype = *dtype;
    }
    auto out = NPUArray(shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMaximumGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmax) aclnnMaximumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[miscellaneous.cpp](fmax) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](fmax) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) {
                error_msg += " - " + std::string(detailed_msg);
            }
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMaximum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmax) aclnnMaximum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmax) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

NPUArray Fmin(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_dtype = x1.dtype;
    auto acl_dtype = x1.aclDtype;
    auto shape = GetBroadcastShape(x1, x2);
    auto temp = NPUArray::GetACLDataType(out_dtype);
    if (temp == ACL_INT16 || temp == ACL_INT32 || temp == ACL_INT64) {
        out_dtype = x2.dtype;
    }
    if (dtype != std::nullopt) {
        out_dtype = *dtype;
    }
    auto out = NPUArray(shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMinimumGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmin) aclnnMinimumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[miscellaneous.cpp](fmin) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](fmin) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) {
                error_msg += " - " + std::string(detailed_msg);
            }
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMinimum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmin) aclnnMinimum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmin) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) {
            error_msg += " - " + std::string(detailed_msg);
        }
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

NPUArray Max(const NPUArray& a, int64_t axis, bool keepdims) {
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    }
    else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> data = {ax};
    auto axis_array = aclCreateIntArray(data.data(), data.size());
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnAmaxGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, 
        result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[extrema_finding.cpp](max) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = aclnnAmax(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnAmax error");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

double Max(const NPUArray& a) {
    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnMaxGetWorkspaceSize(a.tensorPtr, result.tensorPtr, 
        &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[extrema_finding.cpp](max) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = aclnnMax(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnMax error");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    
    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        return results[0];
    } 
    else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        return results[0];
    }
    else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        return results[0];
    }
    else {
        throw std::runtime_error("Unsupported array data type!");
    }
    return 0;
}

NPUArray Nanmax(const NPUArray& a, int64_t axis, bool keepdims) {
    auto shape = a.shape;
    auto temp = NPUArray(a.shape, a.aclDtype);
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    }
    else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> data = {ax};
    auto axis_array = aclCreateIntArray(data.data(), data.size());
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, -std::numeric_limits<float>::infinity(), 
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 
        temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    if (workspaceSize1 < 0ULL) {
        throw std::runtime_error("[extrema_finding.cpp](nanmax) Invalid workspaceSize: " + std::to_string(workspaceSize1));
    }

    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0ULL) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }

    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnNanToNum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    if (workspaceAddr1) {
        aclrtFree(workspaceAddr1);
    }

    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnAmaxGetWorkspaceSize(temp.tensorPtr, axis_array, keepdims, 
        result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[extrema_finding.cpp](nanmax) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = aclnnAmax(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnAmax error");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

double Nanmax(const NPUArray& a) {
    auto temp = NPUArray(a.shape, a.aclDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, -std::numeric_limits<float>::infinity(), 
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 
        temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    if (workspaceSize1 < 0ULL) {
        throw std::runtime_error("[extrema_finding.cpp](nanmax) Invalid workspaceSize: " + std::to_string(workspaceSize1));
    }

    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0ULL) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }

    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnNanToNum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    if (workspaceAddr1) {
        aclrtFree(workspaceAddr1);
    }

    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnMaxGetWorkspaceSize(temp.tensorPtr, result.tensorPtr, 
        &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[extrema_finding.cpp](max) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = aclnnMax(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnMax error");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        return results[0];
    } 
    else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        return results[0];
    }
    else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        return results[0];
    }
    else {
        throw std::runtime_error("Unsupported array data type!");
    }
    return 0;
}

NPUArray Min(const NPUArray& a, int64_t axis, bool keepdims) {
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    }
    else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> data = {ax};
    auto axis_array = aclCreateIntArray(data.data(), data.size());
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnAminGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, 
        result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[extrema_finding.cpp](min) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = aclnnAmin(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnAmin error");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

double Min(const NPUArray& a) {
    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnMinGetWorkspaceSize(a.tensorPtr, result.tensorPtr, 
        &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[extrema_finding.cpp](min) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = aclnnMin(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnMin error");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    
    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        return results[0];
    } 
    else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        return results[0];
    }
    else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        return results[0];
    }
    else {
        throw std::runtime_error("Unsupported array data type!");
    }
    return 0;
}

}

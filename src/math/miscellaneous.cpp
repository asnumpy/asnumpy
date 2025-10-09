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


#include <asnumpy/math/miscellaneous.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/npu_ops_macros.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_flip.h>
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_heaviside.h>
#include <aclnnop/aclnn_maximum.h>
#include <aclnnop/aclnn_minimum.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>

namespace asnumpy {

/**NPUArray Convolve(const NPUArray& a, const NPUArray& v) {
    std::vector<int64_t> dims = {2};
    auto dims_acl = aclCreateIntArray(dims.data(), 1);
    auto shape1 = a.shape;
    auto temp = NPUArray(shape1, a.aclDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnFlipGetWorkspaceSize(a.tensorPtr, dims_acl, temp.tensorPtr, &workspaceSize1, &executor1);
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclnnFlipGetWorkspaceSize error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize1 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](convolve) Invalid workspaceSize: " + std::to_string(workspaceSize1));
    }

    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error1 != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](convolve) aclrtMalloc error = " + std::to_string(error1);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error1 = aclnnFlip(workspaceAddr1, workspaceSize1, executor1, nullptr);
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclnnFlip error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr1) aclrtFree(workspaceAddr1);
        throw std::runtime_error(error_msg);
    }
    error1 = aclrtSynchronizeDevice();
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclrtSynchronizeDevice error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr1) aclrtFree(workspaceAddr1);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr1) aclrtFree(workspaceAddr1);

    auto shape2 = v.shape;
    int64_t size = shape1[2] + shape2[2] - 1;
    std::vector<int64_t> shapeResult = {1, 1, size, 1};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convPads = {shape2[2] - 1, 0};
    std::vector<int64_t> convOutPads = {0, 0};
    std::vector<int64_t> convDilations = {1, 1};
    auto strides = aclCreateIntArray(convStrides.data(), 2);
    auto pads = aclCreateIntArray(convPads.data(), 2);
    auto outPads = aclCreateIntArray(convOutPads.data(), 2);
    auto dilations = aclCreateIntArray(convDilations.data(), 2);
    auto result = NPUArray(shapeResult, ACL_FLOAT);
    result.tensorPtr = aclCreateTensor(result.shape.data(), result.shape.size(), GetACLDataType(result.dtype), result.strides.data(), 0, ACL_FORMAT_NCHW, result.shape.data(), result.shape.size(), result.devicePtr);
    int8_t use_fp16 = 2;
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnConvolutionGetWorkspaceSize(temp.tensorPtr, v.tensorPtr, nullptr, strides, pads, dilations, false, outPads, 1, result.tensorPtr, use_fp16, &workspaceSize2, &executor2);
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclnnConvolutionGetWorkspaceSize error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize2 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](convolve) Invalid workspaceSize: " + std::to_string(workspaceSize2));
    }

    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error2 != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](convolve) aclrtMalloc error = " + std::to_string(error2);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error2 = aclnnConvolution(workspaceAddr2, workspaceSize2, executor2, nullptr);
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclnnConvolution error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr2) aclrtFree(workspaceAddr2);
        throw std::runtime_error(error_msg);
    }

    error2 = aclrtSynchronizeDevice();
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclrtSynchronizeDevice error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr2) aclrtFree(workspaceAddr2);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr2) aclrtFree(workspaceAddr2);
    return result;
}*/

NPUArray Clip(const NPUArray& a, const NPUArray& a_min, const NPUArray& a_max) {
    auto temp = GetBroadcastShape(a, a_min);
    auto x = NPUArray(temp, ACL_FLOAT);
    auto broadcast = GetBroadcastShape(x, a_max);
    auto result = NPUArray(broadcast, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnClampTensorGetWorkspaceSize(a.tensorPtr, a_min.tensorPtr, a_max.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampTensorGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](clip) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnClampTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampTensor error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return result;
}

NPUArray Clip(const NPUArray& a, float a_min, float a_max) {
    auto shape = a.shape;
    auto amin_scalar = aclCreateScalar(&a_min, ACL_FLOAT);
    auto amax_scalar = aclCreateScalar(&a_max, ACL_FLOAT);
    auto result = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnClampGetWorkspaceSize(a.tensorPtr, amin_scalar, amax_scalar, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = aclnnClamp(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnClamp error");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return result;
}

NPUArray Clip(const NPUArray& a, float a_min, const NPUArray& a_max) {
    auto shape = a.shape;
    auto amin_scalar = aclCreateScalar(&a_min, ACL_FLOAT);
    auto temp = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnClampMinGetWorkspaceSize(a.tensorPtr, amin_scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    if (workspaceSize1 < 0ULL) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize1));
    }

    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0ULL) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }

    error1 = aclnnClampMin(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnClampMin error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    if (workspaceAddr1) aclrtFree(workspaceAddr1);

    auto broadcast = GetBroadcastShape(temp, a_max);
    auto result = NPUArray(broadcast, ACL_FLOAT);
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnClampMaxTensorGetWorkspaceSize(temp.tensorPtr, a_max.tensorPtr, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    if (workspaceSize2 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize2));
    }

    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }

    error2 = aclnnClampMaxTensor(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnClampMaxTensor error");

    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    if (workspaceAddr2) aclrtFree(workspaceAddr2);
    return result;
}

NPUArray Clip(const NPUArray& a, const NPUArray& a_min, float a_max) {
    auto shape = a.shape;
    auto amax_scalar = aclCreateScalar(&a_max, ACL_FLOAT);
    auto temp = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnClampMaxGetWorkspaceSize(a.tensorPtr, amax_scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    if (workspaceSize1 < 0ULL) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize1));
    }

    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0ULL) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }

    error1 = aclnnClampMax(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnClampMax error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    if (workspaceAddr1) aclrtFree(workspaceAddr1);

    auto broadcast = GetBroadcastShape(a_min, temp);
    auto result = NPUArray(broadcast, ACL_FLOAT);
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnClampMinTensorGetWorkspaceSize(temp.tensorPtr, a_min.tensorPtr, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    if (workspaceSize2 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize2));
    }

    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }

    error2 = aclnnClampMinTensor(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnClampMinTensor error");

    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    if (workspaceAddr2) aclrtFree(workspaceAddr2);
    return result;
}

NPUArray Square(const NPUArray& x) {
    auto shape = x.shape;
    auto dtype = NPUArray::GetACLDataType(x.dtype);
    auto temp = ACL_FLOAT;
    if (dtype == ACL_DOUBLE) {
        temp = ACL_DOUBLE;
    }
    NPUArray result(shape, temp);
    float two = 2.0f;
    aclScalar* scalar = aclCreateScalar(&two, ACL_FLOAT);;

    // 获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPowTensorScalarGetWorkspaceSize(
        x.tensorPtr, scalar, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](square) aclnnPowTensorScalarGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        aclDestroyScalar(scalar);
        throw std::runtime_error(error_msg);
    }

    // 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](square) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            aclDestroyScalar(scalar);
            throw std::runtime_error(error_msg);
        }
    }

    // 执行计算
    error = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](square) aclnnPowTensorScalar error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(scalar);
        throw std::runtime_error(error_msg);
    }

    // 同步
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](square) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(scalar);
        throw std::runtime_error(error_msg);
    }

    // 释放资源
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(scalar);

    return result;
}


DEFINE_UNARY_OP(Absolute, aclnnAbsGetWorkspaceSize, aclnnAbs)
DEFINE_UNARY_OP(Sign, aclnnSignGetWorkspaceSize, aclnnSign)
DEFINE_BINARY_OP(Heaviside, aclnnHeavisideGetWorkspaceSize, aclnnHeaviside)

NPUArray Fabs(const NPUArray& x){
    // absolute 处理所有数据类型（包括复数等） fabs只处理float和int，
    // 但aclnnAbs不支持复数，所以这里默认fabs=absolute
    return asnumpy::Absolute(x);
}

/**
 * @brief Replace NaN and infinities in an array using NPU.
 *
 * Creates an output array and applies aclnnNanToNum to replace NaN, +inf, and -inf.
 */
NPUArray Nan_to_num(const NPUArray& x, float nan, py::object posinf, py::object neginf) {
    auto out = NPUArray(x.shape, x.aclDtype);

    // Convert optional posinf/neginf to floats; use NaN as "not provided" sentinel.
    float pos_val = std::numeric_limits<float>::max();
    float neg_val = -std::numeric_limits<float>::max();
    if (!posinf.is_none()) pos_val = posinf.cast<float>();
    if (!neginf.is_none()) neg_val = neginf.cast<float>();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto error = aclnnNanToNumGetWorkspaceSize(
        x.tensorPtr,          // input
        nan,                  // NaN replacement
        pos_val,              // +inf replacement (NaN sentinel means "use default")
        neg_val,              // -inf replacement (NaN sentinel means "use default")
        out.tensorPtr,        // output
        &workspaceSize,
        &executor
    );
    CheckGetWorkspaceSizeAclnnStatus(error);
    if (workspaceSize < 0ULL) {
        throw std::runtime_error("[floating_point_routines.cpp](nan_to_num) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = aclnnNanToNum(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnNanToNum error");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

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
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
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
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    // 6. 执行 Maximum 操作
    error = aclnnMaximum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[Maximum] aclnnMaximum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    // 7. 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[Maximum] aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
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
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
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
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMinimum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](minimum) aclnnMinimum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](minimum) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
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
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
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
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMaximum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmax) aclnnMaximum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmax) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
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
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
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
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMinimum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmin) aclnnMinimum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmin) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

}
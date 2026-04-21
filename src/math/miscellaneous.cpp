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
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_flip.h>
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_relu.h>
#include <aclnnop/aclnn_gelu.h> 
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_heaviside.h>

#include <fmt/core.h>
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
    LOG_DEBUG("aclnnClampTensor start: a_shape={}, aclDtype={}",
              detail::FormatShape(a.shape), AclDtypeName(a.aclDtype));
    auto temp = GetBroadcastShape(a, a_min);
    auto x = NPUArray(temp, ACL_FLOAT);
    auto broadcast = GetBroadcastShape(x, a_max);
    auto result = NPUArray(broadcast, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnClampTensorGetWorkspaceSize(a.tensorPtr, a_min.tensorPtr, a_max.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnClampTensorGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnClampTensor(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnClampTensor");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnClampTensor completed");
    return result;
}

NPUArray Clip(const NPUArray& a, float a_min, float a_max) {
    LOG_DEBUG("aclnnClamp start: a_shape={}, aclDtype={}, a_min={}, a_max={}",
              detail::FormatShape(a.shape), AclDtypeName(a.aclDtype), a_min, a_max);
    auto shape = a.shape;
    auto amin_scalar = aclCreateScalar(&a_min, ACL_FLOAT);
    auto amax_scalar = aclCreateScalar(&a_max, ACL_FLOAT);
    auto result = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnClampGetWorkspaceSize(a.tensorPtr, amin_scalar, amax_scalar, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnClampGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnClamp(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnClamp");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnClamp completed");
    return result;
}

NPUArray Clip(const NPUArray& a, float a_min, const NPUArray& a_max) {
    LOG_DEBUG("aclnnClampMin start: a_shape={}, aclDtype={}, a_min={}",
              detail::FormatShape(a.shape), AclDtypeName(a.aclDtype), a_min);
    auto shape = a.shape;
    auto amin_scalar = aclCreateScalar(&a_min, ACL_FLOAT);
    auto temp = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnClampMinGetWorkspaceSize(a.tensorPtr, amin_scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnClampMinGetWorkspaceSize");

    AclWorkspace workspace1(workspaceSize1);

    error1 = aclnnClampMin(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnClampMin");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnClampMin completed");

    py::dtype dtype = NPUArray::GetPyDtype(ACL_FLOAT);
    return EXECUTE_BINARY_OP(
        temp,
        a_max,
        dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnClampMaxTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnClampMaxTensor(workspace, workspaceSize, executor, nullptr);
        },
        "Clip",
        "aclnnClampMaxTensor"
    );
}

NPUArray Clip(const NPUArray& a, const NPUArray& a_min, float a_max) {
    LOG_DEBUG("aclnnClampMax start: a_shape={}, aclDtype={}, a_max={}",
              detail::FormatShape(a.shape), AclDtypeName(a.aclDtype), a_max);
    auto shape = a.shape;
    auto amax_scalar = aclCreateScalar(&a_max, ACL_FLOAT);
    auto temp = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnClampMaxGetWorkspaceSize(a.tensorPtr, amax_scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnClampMaxGetWorkspaceSize");

    AclWorkspace workspace1(workspaceSize1);

    error1 = aclnnClampMax(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnClampMax");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnClampMax completed");

    py::dtype dtype = NPUArray::GetPyDtype(ACL_FLOAT);
    return EXECUTE_BINARY_OP(
        temp,
        a_min,
        dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnClampMinTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnClampMinTensor(workspace, workspaceSize, executor, nullptr);
        },
        "Clip",
        "aclnnClampMinTensor"
    );
}

NPUArray Sqrt(const NPUArray& x) {
    aclDataType aclType = ACL_DOUBLE;
    if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE ||
        x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
        aclType = x.aclDtype;
    }
    ACL_DTYPE_WARN(x.aclDtype, aclType, __func__);
    py::dtype dtype = NPUArray::GetPyDtype(aclType);
    return EXECUTE_UNARY_OP(
        x,
        dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnSqrtGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnSqrt(workspace, workspaceSize, executor, nullptr);
        },
        "Sqrt",
        "aclnnSqrt"
    );
}

NPUArray Square(const NPUArray& x) {
    LOG_DEBUG("aclnnPowTensorScalar start: input_shape={}, tensorSize={}, aclDtype={}",
              detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype));
    auto shape = x.shape;
    auto dtype = NPUArray::GetACLDataType(x.dtype);
    auto temp = ACL_FLOAT;
    if (dtype == ACL_DOUBLE) {
        temp = ACL_DOUBLE;
    }
    ACL_DTYPE_WARN(dtype, temp, __func__);
    NPUArray result(shape, temp);
    float two = 2.0f;
    aclScalar* scalar = aclCreateScalar(&two, ACL_FLOAT);;

    // 获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPowTensorScalarGetWorkspaceSize(
        x.tensorPtr, scalar, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnPowTensorScalarGetWorkspaceSize");

    // 分配 workspace
    AclWorkspace workspace(workspaceSize);

    // 执行计算
    error = aclnnPowTensorScalar(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnPowTensorScalar");

    // 同步
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    // 释放资源
    aclDestroyScalar(scalar);

    LOG_INFO("aclnnPowTensorScalar completed");

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
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}",
              detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype));
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
    ACLNN_CHECK(error, "aclnnNanToNumGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnNanToNum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnNanToNum");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    LOG_INFO("aclnnNanToNum completed");

    return out;
}


/**
 * @brief Compute element-wise Rectified Linear Unit (ReLU).
 * 
 * Applies ReLU activation function element-wise: max(0, x).
 * Equivalent to numpy.maximum(x, 0).
 * 
 * @param x Input array.
 * @param dtype Optional target numpy dtype for the output array. If not provided, uses input dtype.
 * @return NPUArray Array with element-wise ReLU values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
 NPUArray Relu(const NPUArray& x, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.has_value() ? dtype.value() : x.dtype;
    return EXECUTE_UNARY_OP(
        x,
        out_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnReluGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnRelu(workspace, workspaceSize, executor, nullptr);
        },
        "Relu",
        "aclnnRelu"
    );
}


/**
 * @brief Compute element-wise Gaussian Error Linear Unit (GELU).
 * 
 * Applies GELU activation function element-wise: GELU(x) = x * Φ(x)
 * where Φ(x) is the cumulative distribution function of the standard normal distribution.
 * 
 * GELU is commonly used in models like BERT and GPT. It provides smoother gradients
 * compared to ReLU and incorporates probabilistic properties.
 * 
 * @param x Input array.
 * @param dtype Optional target numpy dtype for the output array. If not provided, uses input dtype.
 * @return NPUArray Array with element-wise GELU values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
 NPUArray Gelu(const NPUArray& x, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.has_value() ? dtype.value() : x.dtype;
    return EXECUTE_UNARY_OP(
        x,
        out_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnGeluGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnGelu(workspace, workspaceSize, executor, nullptr);
        },
        "Gelu",
        "aclnnGelu"
    );
}
}
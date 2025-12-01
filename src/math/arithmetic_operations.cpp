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


#include <asnumpy/math/arithmetic_operations.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/npu_scalar.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_floor_divide.h>
#include <aclnnop/aclnn_reciprocal.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_trunc.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_pow_tensor_tensor.h>
#include <aclnnop/aclnn_fmod_tensor.h>
#include <aclnnop/aclnn_remainder.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

/**
 * @brief Element-wise addition using aclnnAdd.
 */
NPUArray Add(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.has_value() ? dtype.value() : x1.dtype;

    auto out_shape = GetBroadcastShape(x1, x2);
    auto out = NPUArray(out_shape, out_dtype);

    int32_t one = 1;
    aclScalar* alpha_scalar = aclCreateScalar(&one, ACL_INT32);
    if (!alpha_scalar) {
        throw std::runtime_error("[arithmetic_operations.cpp](Add) Failed to create alpha scalar");
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAddGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, alpha_scalar, out.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Add) aclnnAddGetWorkspaceSize error = " + std::to_string(error);
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        aclDestroyScalar(alpha_scalar);
        throw std::runtime_error(msg);
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            const char* detail = aclGetRecentErrMsg();
            std::string msg = "[arithmetic_operations.cpp](Add) aclrtMalloc error = " + std::to_string(error);
            if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
            aclDestroyScalar(alpha_scalar);
            throw std::runtime_error(msg);
        }
    }

    error = aclnnAdd(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Add) aclnnAdd error = " + std::to_string(error);
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(alpha_scalar);
        throw std::runtime_error(msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Add) aclrtSynchronizeDevice error = " + std::to_string(error);
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(alpha_scalar);
        throw std::runtime_error(msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(alpha_scalar);

    return out;
}

/**
 * @brief Element-wise reciprocal using aclnnReciprocal.
 */
NPUArray Reciprocal(const NPUArray& x, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.has_value() ? dtype.value() : x.dtype;
    auto out = NPUArray(x.shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnReciprocalGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Reciprocal) aclnnReciprocalGetWorkspaceSize error = " + std::to_string(error);
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            const char* detail = aclGetRecentErrMsg();
            std::string msg = "[arithmetic_operations.cpp](Reciprocal) aclrtMalloc error = " + std::to_string(error);
            if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
            throw std::runtime_error(msg);
        }
    }

    error = aclnnReciprocal(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Reciprocal) aclnnReciprocal error = " + std::to_string(error);
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Reciprocal) aclrtSynchronizeDevice error = " + std::to_string(error);
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return out;
}

/**
 * @brief Positive operator: copy or cast input array.
 */
NPUArray Positive(const NPUArray& x, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.has_value() ? dtype.value() : x.dtype;

    if (out_dtype.is(x.dtype)) {
        return NPUArray(x);  // 深拷贝
    }

    auto out = NPUArray(x.shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnCastGetWorkspaceSize(
        x.tensorPtr, out.aclDtype, out.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Positive) aclnnCastGetWorkspaceSize error = " + std::to_string(error);
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            const char* detail = aclGetRecentErrMsg();
            std::string msg = "[arithmetic_operations.cpp](Positive) aclrtMalloc error = " + std::to_string(error);
            if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
            throw std::runtime_error(msg);
        }
    }

    error = aclnnCast(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Positive) aclnnCast error = " + std::to_string(error);
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        std::string msg = "[arithmetic_operations.cpp](Positive) aclrtSynchronizeDevice error = " + std::to_string(error);
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return out;
}

/**
 * @brief Unary negative operator using aclnnNeg.
 */
NPUArray Negative(const NPUArray& x, std::optional<py::dtype> dtype) {
    auto out_dtype = dtype.value_or(x.dtype);
    auto out = NPUArray(x.shape, out_dtype);

    // 1. 获取 workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNegGetWorkspaceSize(x.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Negative) aclnnNegGetWorkspaceSize error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    // 2. 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string msg = "[arithmetic_operations.cpp](Negative) aclrtMalloc error = "
                              + std::to_string(error);
            const char* detail = aclGetRecentErrMsg();
            if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
            throw std::runtime_error(msg);
        }
    }

    // 3. 执行 Neg
    error = aclnnNeg(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Negative) aclnnNeg error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    // 4. 同步
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Negative) aclrtSynchronizeDevice error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    // 5. 释放资源
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise multiplication using aclnnMul.
 */
NPUArray Multiply(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 1. 广播输出形状
    auto out_shape = GetBroadcastShape(x1, x2);
    auto out_dtype = dtype.value_or(x1.dtype);
    auto out = NPUArray(out_shape, out_dtype);

    // 2. 获取 workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMulGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Multiply) aclnnMulGetWorkspaceSize error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    // 3. 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string msg = "[arithmetic_operations.cpp](Multiply) aclrtMalloc error = "
                              + std::to_string(error);
            const char* detail = aclGetRecentErrMsg();
            if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
            throw std::runtime_error(msg);
        }
    }

    // 4. 执行算子
    error = aclnnMul(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Multiply) aclnnMul error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    // 5. 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Multiply) aclrtSynchronizeDevice error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    // 6. 释放 workspace
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise division using aclnnDiv.
 */
NPUArray Divide(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 1. 广播输出形状
    auto out_shape = GetBroadcastShape(x1, x2);
    auto out_dtype = dtype.value_or(x1.dtype);
    auto out = NPUArray(out_shape, out_dtype);

    // 2. 获取 workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnDivGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Divide) aclnnDivGetWorkspaceSize error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    // 3. 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string msg = "[arithmetic_operations.cpp](Divide) aclrtMalloc error = "
                              + std::to_string(error);
            const char* detail = aclGetRecentErrMsg();
            if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
            throw std::runtime_error(msg);
        }
    }

    // 4. 执行算子
    error = aclnnDiv(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Divide) aclnnDiv error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    // 5. 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Divide) aclrtSynchronizeDevice error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    // 6. 释放 workspace
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise true division (delegates to Divide).
 */
NPUArray TrueDivide(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    return Divide(x1, x2, dtype);
}

/**
 * @brief Element-wise subtraction using aclnnSub.
 */
NPUArray Subtract(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 1. 广播输出形状
    auto out_shape = GetBroadcastShape(x1, x2);
    auto out_dtype = dtype.value_or(x1.dtype);
    auto out = NPUArray(out_shape, out_dtype);

    // 2. 创建 alpha = 1 标量
    int32_t one = 1;
    aclScalar* alpha_scalar = aclCreateScalar(&one, ACL_INT32);
    if (!alpha_scalar) {
        throw std::runtime_error("[arithmetic_operations.cpp](Subtract) Failed to create alpha scalar");
    }

    // 3. 获取 workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSubGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, alpha_scalar, out.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Subtract) aclnnSubGetWorkspaceSize error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        aclDestroyScalar(alpha_scalar);
        throw std::runtime_error(msg);
    }

    // 4. 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string msg = "[arithmetic_operations.cpp](Subtract) aclrtMalloc error = "
                              + std::to_string(error);
            const char* detail = aclGetRecentErrMsg();
            if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
            aclDestroyScalar(alpha_scalar);
            throw std::runtime_error(msg);
        }
    }

    // 5. 执行算子
    error = aclnnSub(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Subtract) aclnnSub error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(alpha_scalar);
        throw std::runtime_error(msg);
    }

    // 6. 同步
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Subtract) aclrtSynchronizeDevice error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(alpha_scalar);
        throw std::runtime_error(msg);
    }

    // 7. 释放资源
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(alpha_scalar);

    return out;
}

/**
 * @brief Element-wise floor division using aclnnFloorDivide.
 */
NPUArray FloorDivide(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 1. 广播输出形状
    auto out_shape = GetBroadcastShape(x1, x2);
    auto out_dtype = dtype.value_or(x1.dtype);
    auto out = NPUArray(out_shape, out_dtype);

    // 2. 获取 workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnFloorDivideGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](FloorDivide) aclnnFloorDivideGetWorkspaceSize error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    // 3. 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string msg = "[arithmetic_operations.cpp](FloorDivide) aclrtMalloc error = "
                              + std::to_string(error);
            const char* detail = aclGetRecentErrMsg();
            if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
            throw std::runtime_error(msg);
        }
    }

    // 4. 执行算子
    error = aclnnFloorDivide(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](FloorDivide) aclnnFloorDivide error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    // 5. 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](FloorDivide) aclrtSynchronizeDevice error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && strlen(detail) > 0) msg += " - " + std::string(detail);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(msg);
    }

    // 6. 释放 workspace
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise power using aclnnPowTensorTensor.
 */
NPUArray Power(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(x1.dtype);
    auto out_shape = GetBroadcastShape(x1, x2);
    auto out = NPUArray(out_shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPowTensorTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error("[arithmetic_operations.cpp](Power TensorTensor) GetWorkspaceSize error = " +
                                 std::to_string(error));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error("[arithmetic_operations.cpp](Power TensorTensor) aclrtMalloc error = " +
                                     std::to_string(error));
        }
    }

    error = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[arithmetic_operations.cpp](Power TensorTensor) error = " +
                                 std::to_string(error));
    }

    aclrtSynchronizeDevice();
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return out;
}

/**
 * @brief Scalar ** Tensor power using aclnnPowScalarTensor.
 */
NPUArray Power(const py::object& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    if (x1.is_none()) {
        throw std::runtime_error("[arithmetic_operations.cpp](Power ScalarTensor) Input scalar is None");
    }

    double value = 0;
    try {
        value = py::cast<double>(x1);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[arithmetic_operations.cpp](Power ScalarTensor) Conversion error: " +
                                 std::string(e.what()));
    }

    aclScalar* x1_scalar = CreateScalar(value, ACL_FLOAT);
    auto out = NPUArray(x2.shape, ACL_DOUBLE);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPowScalarTensorGetWorkspaceSize(
        x1_scalar, x2.tensorPtr, out.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(x1_scalar);
        throw std::runtime_error("[arithmetic_operations.cpp](Power ScalarTensor) GetWorkspaceSize error = " +
                                 std::to_string(error));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(x1_scalar);
            throw std::runtime_error("[arithmetic_operations.cpp](Power ScalarTensor) aclrtMalloc error = " +
                                     std::to_string(error));
        }
    }

    error = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(x1_scalar);
        throw std::runtime_error("[arithmetic_operations.cpp](Power ScalarTensor) error = " +
                                 std::to_string(error));
    }

    aclrtSynchronizeDevice();
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(x1_scalar);
    return out;
}

/**
 * @brief Tensor ** Scalar power using aclnnPowTensorScalar.
 */
NPUArray Power(const NPUArray& x1, const py::object& x2, std::optional<py::dtype> dtype) {
    if (x2.is_none()) {
        throw std::runtime_error("[arithmetic_operations.cpp](Power TensorScalar) Input scalar is None");
    }

    double value = 0;
    try {
        value = py::cast<double>(x2);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[arithmetic_operations.cpp](Power TensorScalar) Conversion error: " +
                                 std::string(e.what()));
    }

    aclScalar* x2_scalar = CreateScalar(value, ACL_FLOAT);
    auto out = NPUArray(x1.shape, ACL_DOUBLE);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPowTensorScalarGetWorkspaceSize(
        x1.tensorPtr, x2_scalar, out.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(x2_scalar);
        throw std::runtime_error("[arithmetic_operations.cpp](Power TensorScalar) GetWorkspaceSize error = " +
                                 std::to_string(error));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(x2_scalar);
            throw std::runtime_error("[arithmetic_operations.cpp](Power TensorScalar) aclrtMalloc error = " +
                                     std::to_string(error));
        }
    }

    error = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(x2_scalar);
        throw std::runtime_error("[arithmetic_operations.cpp](Power TensorScalar) error = " +
                                 std::to_string(error));
    }

    aclrtSynchronizeDevice();
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(x2_scalar);
    return out;
}

/**
 * @brief Element-wise floating-point power using aclnnPowTensorTensor.
 */
NPUArray FloatPower(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_shape = GetBroadcastShape(x1, x2);

    // 输出必须是浮点，默认 float32
    py::dtype out_dtype = dtype.value_or(py::dtype::of<float>());
    if (!(out_dtype.is(py::dtype::of<float>()) || out_dtype.is(py::dtype::of<double>()))) {
        throw std::runtime_error("[arithmetic_operations.cpp](FloatPower) dtype must be float or double");
    }

    auto out = NPUArray(out_shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPowTensorTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](FloatPower) aclnnPowTensorTensorGetWorkspaceSize error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error("[arithmetic_operations.cpp](FloatPower) aclrtMalloc error = "
                                     + std::to_string(error));
        }
    }

    error = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[arithmetic_operations.cpp](FloatPower) aclnnPowTensorTensor error = "
                                 + std::to_string(error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[arithmetic_operations.cpp](FloatPower) aclrtSynchronizeDevice error = "
                                 + std::to_string(error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise floating-point remainder using aclnnFmodTensor.
 */
NPUArray Fmod(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_shape = GetBroadcastShape(x1, x2);

    py::dtype out_dtype = dtype.value_or(py::dtype::of<float>());
    if (!(out_dtype.is(py::dtype::of<float>()) || out_dtype.is(py::dtype::of<double>()))) {
        throw std::runtime_error("[arithmetic_operations.cpp](Fmod) dtype must be float or double");
    }

    auto out = NPUArray(out_shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnFmodTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Fmod) aclnnFmodTensorGetWorkspaceSize error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error("[arithmetic_operations.cpp](Fmod) aclrtMalloc error = "
                                     + std::to_string(error));
        }
    }

    error = aclnnFmodTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[arithmetic_operations.cpp](Fmod) aclnnFmodTensor error = "
                                 + std::to_string(error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[arithmetic_operations.cpp](Fmod) aclrtSynchronizeDevice error = "
                                 + std::to_string(error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise remainder using aclnnRemainderTensorTensor.
 */
NPUArray Mod(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_shape = GetBroadcastShape(x1, x2);

    py::dtype out_dtype = dtype.value_or(py::dtype::of<float>());
    if (!(out_dtype.is(py::dtype::of<float>()) || out_dtype.is(py::dtype::of<double>()))) {
        throw std::runtime_error("[arithmetic_operations.cpp](Mod) dtype must be float or double");
    }

    auto out = NPUArray(out_shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnRemainderTensorTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string msg = "[arithmetic_operations.cpp](Mod) aclnnRemainderTensorTensorGetWorkspaceSize error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error("[arithmetic_operations.cpp](Mod) aclrtMalloc error = "
                                     + std::to_string(error));
        }
    }

    error = aclnnRemainderTensorTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[arithmetic_operations.cpp](Mod) aclnnRemainderTensorTensor error = "
                                 + std::to_string(error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[arithmetic_operations.cpp](Mod) aclrtSynchronizeDevice error = "
                                 + std::to_string(error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise modf using aclnnFloor and aclnnSub.
 */
std::pair<NPUArray, NPUArray> Modf(const NPUArray& x) {
    if (!(x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_DOUBLE)) {
        throw std::runtime_error("[arithmetic_operations.cpp](Modf) input must be float or double");
    }

    auto int_part  = NPUArray(x.shape, x.aclDtype);
    auto frac_part = NPUArray(x.shape, x.aclDtype);

    // === Floor ===
    uint64_t floor_ws = 0;
    aclOpExecutor* floor_exec = nullptr;
    auto error = aclnnFloorGetWorkspaceSize(x.tensorPtr, int_part.tensorPtr, &floor_ws, &floor_exec);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error("[arithmetic_operations.cpp](Modf) aclnnFloorGetWorkspaceSize error = " +
                                 std::to_string(error));
    }

    void* floor_ws_addr = nullptr;
    if (floor_ws > 0) {
        error = aclrtMalloc(&floor_ws_addr, floor_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error("[arithmetic_operations.cpp](Modf) aclrtMalloc for floor error = " +
                                     std::to_string(error));
        }
    }

    error = aclnnFloor(floor_ws_addr, floor_ws, floor_exec, nullptr);
    if (error != ACL_SUCCESS) {
        if (floor_ws_addr) aclrtFree(floor_ws_addr);
        throw std::runtime_error("[arithmetic_operations.cpp](Modf) aclnnFloor error = " +
                                 std::to_string(error));
    }
    if (floor_ws_addr) aclrtFree(floor_ws_addr);

    // === Sub (frac = x - int_part) ===
    uint64_t sub_ws = 0;
    aclOpExecutor* sub_exec = nullptr;
    int32_t one = 1;
    aclScalar* alpha = aclCreateScalar(&one, ACL_INT32);
    if (!alpha) {
        throw std::runtime_error("[arithmetic_operations.cpp](Modf) Failed to create alpha scalar");
    }

    error = aclnnSubGetWorkspaceSize(x.tensorPtr, int_part.tensorPtr, alpha, frac_part.tensorPtr, &sub_ws, &sub_exec);
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(alpha);
        throw std::runtime_error("[arithmetic_operations.cpp](Modf) aclnnSubGetWorkspaceSize error = " +
                                 std::to_string(error));
    }

    void* sub_ws_addr = nullptr;
    if (sub_ws > 0) {
        error = aclrtMalloc(&sub_ws_addr, sub_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(alpha);
            throw std::runtime_error("[arithmetic_operations.cpp](Modf) aclrtMalloc for sub error = " +
                                     std::to_string(error));
        }
    }

    error = aclnnSub(sub_ws_addr, sub_ws, sub_exec, nullptr);
    if (error != ACL_SUCCESS) {
        if (sub_ws_addr) aclrtFree(sub_ws_addr);
        aclDestroyScalar(alpha);
        throw std::runtime_error("[arithmetic_operations.cpp](Modf) aclnnSub error = " +
                                 std::to_string(error));
    }

    if (sub_ws_addr) aclrtFree(sub_ws_addr);
    aclDestroyScalar(alpha);

    // === 同步设备 ===
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        throw std::runtime_error("[arithmetic_operations.cpp](Modf) aclrtSynchronizeDevice error = " +
                                 std::to_string(error));
    }

    return {frac_part, int_part};
}

/**
 * @brief Element-wise remainder, reusing Mod().
 */
NPUArray Remainder(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    return Mod(x1, x2, dtype.value_or(x1.dtype));
}

/**
 * @brief Element-wise divmod using aclnnDivMod (mode=2) + Multiply/Subtract.
 */
std::pair<NPUArray, NPUArray> Divmod(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 1. 确定输出 dtype（默认和 x1 一致）
    py::dtype out_dtype = dtype.value_or(x1.dtype);

    // 2. 广播后的输出形状
    auto out_shape = GetBroadcastShape(x1, x2);

    // 3. 商 q = floor(x1 / x2) via aclnnDivMod(mode=2)
    NPUArray quotient(out_shape, out_dtype);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnDivModGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, /*mode=*/2,
        quotient.tensorPtr, &ws_size, &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error("[arithmetic_operations.cpp](Divmod) aclnnDivModGetWorkspaceSize error = "
                                 + std::to_string(error));
    }

    void* ws_addr = nullptr;
    if (ws_size > 0ULL) {
        error = aclrtMalloc(&ws_addr, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error("[arithmetic_operations.cpp](Divmod) aclrtMalloc error = "
                                     + std::to_string(error));
        }
    }

    error = aclnnDivMod(ws_addr, ws_size, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (ws_addr) {
            aclrtFree(ws_addr);
        }
        throw std::runtime_error("[arithmetic_operations.cpp](Divmod) aclnnDivMod error = "
                                 + std::to_string(error));
    }
    if (ws_addr) {
        aclrtFree(ws_addr);
    }

    // 4. 余数 r = x1 - q * x2
    NPUArray qx2 = Multiply(quotient, x2, out_dtype);
    NPUArray remainder = Subtract(x1, qx2, out_dtype);

    // 5. 同步
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        throw std::runtime_error("[arithmetic_operations.cpp](Divmod) aclrtSynchronizeDevice error = "
                                 + std::to_string(error));
    }

    return {quotient, remainder};
}

/**
 * @brief Element-wise power using aclnnPowTensorTensor.
 */
NPUArray Pow(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    return Power(x1, x2, dtype);
}

/**
 * @brief Scalar ** Tensor power using aclnnPowScalarTensor.
 */
NPUArray Pow(const py::object& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    return Power(x1, x2, dtype); 
}

/**
 * @brief Tensor ** Scalar power using aclnnPowTensorScalar.
 */
NPUArray Pow(const NPUArray& x1, const py::object& x2, std::optional<py::dtype> dtype) {
    return Power(x1, x2, dtype);
}

}
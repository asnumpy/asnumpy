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
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>

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

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

/**
 * @brief Element-wise addition using aclnnAdd.
 */
NPUArray Add(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnAdd start: x1_shape={}, x2_shape={}, aclDtype={}", detail::FormatShape(x1.shape), detail::FormatShape(x2.shape), AclDtypeName(x1.aclDtype));
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
    ACLNN_CHECK(error, "aclnnAddGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAdd(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAdd");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    aclDestroyScalar(alpha_scalar);

    LOG_INFO("aclnnAdd completed");
    return out;
}

/**
 * @brief Element-wise reciprocal using aclnnReciprocal.
 */
NPUArray Reciprocal(const NPUArray& x, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.has_value() ? dtype.value() : x.dtype;
    return EXECUTE_UNARY_OP(
        x,
        out_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnReciprocalGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnReciprocal(workspace, workspaceSize, executor, nullptr);
        },
        "Reciprocal",
        "aclnnReciprocal"
    );
}

/**
 * @brief Positive operator: copy or cast input array.
 */
NPUArray Positive(const NPUArray& x, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnCast start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype));
    py::dtype out_dtype = dtype.has_value() ? dtype.value() : x.dtype;

    if (out_dtype.is(x.dtype)) {
        LOG_INFO("aclnnCast completed");
        return NPUArray(x);  // 深拷贝
    }

    auto out = NPUArray(x.shape, out_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnCastGetWorkspaceSize(
        x.tensorPtr, out.aclDtype, out.tensorPtr,
        &workspaceSize, &executor
    );
    ACLNN_CHECK(error, "aclnnCastGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnCast(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnCast");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    LOG_INFO("aclnnCast completed");
    return out;
}

/**
 * @brief Unary negative operator using aclnnNeg.
 */
NPUArray Negative(const NPUArray& x, std::optional<py::dtype> dtype) {
    auto out_dtype = dtype.value_or(x.dtype);
    return EXECUTE_UNARY_OP(
        x,
        out_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnNegGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnNeg(workspace, workspaceSize, executor, nullptr);
        },
        "Negative",
        "aclnnNeg"
    );
}

/**
 * @brief Element-wise multiplication using aclnnMul.
 */
NPUArray Multiply(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_dtype = dtype.value_or(x1.dtype);
    return EXECUTE_BINARY_OP(
        x1,
        x2,
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnMulGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnMul(workspace, workspaceSize, executor, nullptr);
        },
        "Multiply",
        "aclnnMul"
    );
}

/**
 * @brief Element-wise division using aclnnDiv.
 */
NPUArray Divide(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out_dtype = dtype.value_or(x1.dtype);
    return EXECUTE_BINARY_OP(
        x1,
        x2,
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnDivGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnDiv(workspace, workspaceSize, executor, nullptr);
        },
        "Divide",
        "aclnnDiv"
    );
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
    LOG_DEBUG("aclnnSub start: x1_shape={}, x2_shape={}, aclDtype={}", detail::FormatShape(x1.shape), detail::FormatShape(x2.shape), AclDtypeName(x1.aclDtype));
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
    ACLNN_CHECK(error, "aclnnSubGetWorkspaceSize");

    // 4. 分配 workspace
    AclWorkspace workspace(workspaceSize);

    // 5. 执行算子
    error = aclnnSub(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnSub");

    // 6. 同步
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    // 7. 释放资源
    aclDestroyScalar(alpha_scalar);

    LOG_INFO("aclnnSub completed");
    return out;
}

/**
 * @brief Element-wise floor division using aclnnFloorDivide.
 */
NPUArray FloorDivide(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnFloorDivide start: x1_shape={}, x2_shape={}, aclDtype={}", detail::FormatShape(x1.shape), detail::FormatShape(x2.shape), AclDtypeName(x1.aclDtype));
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
    ACLNN_CHECK(error, "aclnnFloorDivideGetWorkspaceSize");

    // 3. 分配 workspace
    AclWorkspace workspace(workspaceSize);

    // 4. 执行算子
    error = aclnnFloorDivide(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnFloorDivide");

    // 5. 同步设备
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    LOG_INFO("aclnnFloorDivide completed");
    return out;
}

/**
 * @brief Element-wise power using aclnnPowTensorTensor.
 */
NPUArray Power(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(x1.dtype);
    return EXECUTE_BINARY_OP(
        x1,
        x2,
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnPowTensorTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnPowTensorTensor(workspace, workspaceSize, executor, nullptr);
        },
        "Power",
        "aclnnPowTensorTensor"
    );
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

    LOG_DEBUG("aclnnPowScalarTensor start: scalar={}, x2_shape={}, aclDtype={}", value, detail::FormatShape(x2.shape), AclDtypeName(x2.aclDtype));

    aclScalar* x1_scalar = CreateScalar(value, ACL_FLOAT);
    auto out = NPUArray(x2.shape, ACL_DOUBLE);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPowScalarTensorGetWorkspaceSize(
        x1_scalar, x2.tensorPtr, out.tensorPtr,
        &workspaceSize, &executor
    );
    ACLNN_CHECK(error, "aclnnPowScalarTensorGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnPowScalarTensor(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnPowScalarTensor");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    aclDestroyScalar(x1_scalar);
    LOG_INFO("aclnnPowScalarTensor completed");
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

    LOG_DEBUG("aclnnPowTensorScalar start: x1_shape={}, scalar={}, aclDtype={}", detail::FormatShape(x1.shape), value, AclDtypeName(x1.aclDtype));

    aclScalar* x2_scalar = CreateScalar(value, ACL_FLOAT);
    auto out = NPUArray(x1.shape, ACL_DOUBLE);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPowTensorScalarGetWorkspaceSize(
        x1.tensorPtr, x2_scalar, out.tensorPtr,
        &workspaceSize, &executor
    );
    ACLNN_CHECK(error, "aclnnPowTensorScalarGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnPowTensorScalar(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnPowTensorScalar");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    aclDestroyScalar(x2_scalar);
    LOG_INFO("aclnnPowTensorScalar completed");
    return out;
}

/**
 * @brief Element-wise floating-point power using aclnnPowTensorTensor.
 */
NPUArray FloatPower(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 输出必须是浮点，默认 float32
    py::dtype out_dtype = dtype.value_or(py::dtype::of<float>());
    if (!(out_dtype.is(py::dtype::of<float>()) || out_dtype.is(py::dtype::of<double>()))) {
        throw std::runtime_error("[arithmetic_operations.cpp](FloatPower) dtype must be float or double");
    }
    return EXECUTE_BINARY_OP(
        x1,
        x2,
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnPowTensorTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnPowTensorTensor(workspace, workspaceSize, executor, nullptr);
        },
        "FloatPower",
        "aclnnPowTensorTensor"
    );
}

/**
 * @brief Element-wise floating-point remainder using aclnnFmodTensor.
 */
NPUArray Fmod(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<float>());
    if (!(out_dtype.is(py::dtype::of<float>()) || out_dtype.is(py::dtype::of<double>()))) {
        throw std::runtime_error("[arithmetic_operations.cpp](Fmod) dtype must be float or double");
    }
    return EXECUTE_BINARY_OP(
        x1,
        x2,
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnFmodTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnFmodTensor(workspace, workspaceSize, executor, nullptr);
        },
        "Fmod",
        "aclnnFmodTensor"
    );
}

/**
 * @brief Element-wise remainder using aclnnRemainderTensorTensor.
 */
NPUArray Mod(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<float>());
    if (!(out_dtype.is(py::dtype::of<float>()) || out_dtype.is(py::dtype::of<double>()))) {
        throw std::runtime_error("[arithmetic_operations.cpp](Mod) dtype must be float or double");
    }
    return EXECUTE_BINARY_OP(
        x1,
        x2,
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnRemainderTensorTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnRemainderTensorTensor(workspace, workspaceSize, executor, nullptr);
        },
        "Mod",
        "aclnnRemainderTensorTensor"
    );
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
    LOG_DEBUG("aclnnFloor start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype));
    uint64_t floor_ws = 0;
    aclOpExecutor* floor_exec = nullptr;
    auto error = aclnnFloorGetWorkspaceSize(x.tensorPtr, int_part.tensorPtr, &floor_ws, &floor_exec);
    ACLNN_CHECK(error, "aclnnFloorGetWorkspaceSize");

    AclWorkspace floor_ws_addr(floor_ws);

    error = aclnnFloor(floor_ws_addr.get(), floor_ws, floor_exec, nullptr);
    ACLNN_CHECK(error, "aclnnFloor");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnFloor completed");

    // === Sub (frac = x - int_part) ===
    LOG_DEBUG("aclnnSub start: x_shape={}, int_part_shape={}, aclDtype={}", detail::FormatShape(x.shape), detail::FormatShape(int_part.shape), AclDtypeName(x.aclDtype));
    uint64_t sub_ws = 0;
    aclOpExecutor* sub_exec = nullptr;
    int32_t one = 1;
    aclScalar* alpha = aclCreateScalar(&one, ACL_INT32);
    if (!alpha) {
        throw std::runtime_error("[arithmetic_operations.cpp](Modf) Failed to create alpha scalar");
    }

    error = aclnnSubGetWorkspaceSize(x.tensorPtr, int_part.tensorPtr, alpha, frac_part.tensorPtr, &sub_ws, &sub_exec);
    ACLNN_CHECK(error, "aclnnSubGetWorkspaceSize");

    AclWorkspace sub_ws_addr(sub_ws);

    error = aclnnSub(sub_ws_addr.get(), sub_ws, sub_exec, nullptr);
    ACLNN_CHECK(error, "aclnnSub");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnSub completed");

    aclDestroyScalar(alpha);

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
    LOG_DEBUG("aclnnDivMod start: x1_shape={}, x2_shape={}, aclDtype={}", detail::FormatShape(x1.shape), detail::FormatShape(x2.shape), AclDtypeName(x1.aclDtype));
    // 1. 确定输出 dtype（默认和 x1 一致）
    py::dtype out_dtype = dtype.has_value() ? dtype.value() : x1.dtype;

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
    ACLNN_CHECK(error, "aclnnDivModGetWorkspaceSize");

    AclWorkspace ws(ws_size);

    error = aclnnDivMod(ws.get(), ws_size, executor, nullptr);
    ACLNN_CHECK(error, "aclnnDivMod");

    // 4. 余数 r = x1 - q * x2
    NPUArray qx2 = Multiply(quotient, x2, out_dtype);
    NPUArray remainder = Subtract(x1, qx2, out_dtype);

    // 5. 同步
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    LOG_INFO("aclnnDivMod completed");
    return {quotient, remainder};
}

}
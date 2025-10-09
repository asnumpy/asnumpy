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


#include <asnumpy/logic/logic.hpp>
#include <fmt/base.h>

#include <aclnnop/aclnn_all.h>
#include <aclnnop/aclnn_any.h>
#include <aclnnop/aclnn_isfinite.h>
#include <aclnnop/aclnn_is_inf.h>
#include <aclnnop/aclnn_isneginf.h>
#include <aclnnop/aclnn_isposinf.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_or.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_logical_xor.h>
#include <aclnnop/aclnn_gt_tensor.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_ge_tensor.h>
#include <aclnnop/aclnn_ge_scalar.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_lt_tensor.h>
#include <aclnnop/aclnn_le_tensor.h>
#include <aclnnop/aclnn_le_scalar.h>
#include <aclnnop/aclnn_equal.h>
#include <aclnnop/aclnn_ne_scalar.h>
#include <aclnnop/aclnn_ne_tensor.h>

namespace asnumpy {

/// Reduce array by logical AND operation over all elements.
NPUArray All(const NPUArray& x) {
    auto result = NPUArray({}, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // dim=[] (全局 reduce)
    aclIntArray* aclDim = aclCreateIntArray(nullptr, 0);
    if (!aclDim) {
        throw std::runtime_error("[logic.cpp](All) failed to create empty aclIntArray");
    }

    auto error = aclnnAllGetWorkspaceSize(
        x.tensorPtr,
        aclDim,
        false,  // keepdims = false
        result.tensorPtr,
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](All) aclnnAllGetWorkspaceSize failed, error=" + std::to_string(error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyIntArray(aclDim);
            throw std::runtime_error("[logic.cpp](All) aclrtMalloc failed, error=" + std::to_string(error));
        }
    }

    error = aclnnAll(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](All) aclnnAll failed, error=" + std::to_string(error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](All) sync device failed, error=" + std::to_string(error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyIntArray(aclDim);

    return result;
}

/// Reduce array by logical AND operation over specified dimensions.
NPUArray All(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims) {
    auto result = NPUArray({}, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // 构造 dim 数组
    aclIntArray* aclDim = aclCreateIntArray(dim.data(), dim.size());
    if (!aclDim) {
        throw std::runtime_error("[logic.cpp](All) failed to create aclIntArray");
    }

    auto error = aclnnAllGetWorkspaceSize(
        x.tensorPtr,
        aclDim,
        keepdims,
        result.tensorPtr,
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](All) aclnnAllGetWorkspaceSize failed, error=" + std::to_string(error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyIntArray(aclDim);
            throw std::runtime_error("[logic.cpp](All) aclrtMalloc failed, error=" + std::to_string(error));
        }
    }

    error = aclnnAll(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](All) aclnnAll failed, error=" + std::to_string(error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](All) sync device failed, error=" + std::to_string(error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyIntArray(aclDim);

    return result;
}

/// Reduce array by logical OR operation over all elements.
NPUArray Any(const NPUArray& x) {
    auto result = NPUArray({}, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // dim=[] (全局 reduce)
    aclIntArray* aclDim = aclCreateIntArray(nullptr, 0);
    if (!aclDim) {
        throw std::runtime_error("[logic.cpp](Any) failed to create empty aclIntArray");
    }

    auto error = aclnnAnyGetWorkspaceSize(
        x.tensorPtr,
        aclDim,
        false,  // keepdims = false
        result.tensorPtr,
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](Any) aclnnAnyGetWorkspaceSize failed, error=" + std::to_string(error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyIntArray(aclDim);
            throw std::runtime_error("[logic.cpp](Any) aclrtMalloc failed, error=" + std::to_string(error));
        }
    }

    error = aclnnAny(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](Any) aclnnAny failed, error=" + std::to_string(error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](Any) sync device failed, error=" + std::to_string(error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyIntArray(aclDim);

    return result;
}

/// Reduce array by logical OR operation over specified dimensions.
NPUArray Any(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims) {
    auto result = NPUArray({}, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // 构造 dim 数组
    aclIntArray* aclDim = aclCreateIntArray(dim.data(), dim.size());
    if (!aclDim) {
        throw std::runtime_error("[logic.cpp](Any) failed to create aclIntArray");
    }

    auto error = aclnnAnyGetWorkspaceSize(
        x.tensorPtr,
        aclDim,
        keepdims,
        result.tensorPtr,
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](Any) aclnnAnyGetWorkspaceSize failed, error=" + std::to_string(error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyIntArray(aclDim);
            throw std::runtime_error("[logic.cpp](Any) aclrtMalloc failed, error=" + std::to_string(error));
        }
    }

    error = aclnnAny(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](Any) aclnnAny failed, error=" + std::to_string(error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyIntArray(aclDim);
        throw std::runtime_error("[logic.cpp](Any) sync device failed, error=" + std::to_string(error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyIntArray(aclDim);

    return result;
}

/// Check element-wise finiteness of the input array.
NPUArray IsFinite(const NPUArray& x) {
    // 输出布尔数组，shape 与输入一致
    auto result = NPUArray(x.shape, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // 获取 workspace 大小 & 执行器（按 NNOP 两段式）
    auto error = aclnnIsFiniteGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "[logic.cpp](IsFinite) GetWorkspaceSize failed, error={}, dtype={}",
            error, std::string(py::str(py::cast<py::object>(x.dtype)))));
    }

    // 分配 workspace
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "[logic.cpp](IsFinite) aclrtMalloc failed, error={}", error));
        }
    }

    // 执行计算
    error = aclnnIsFinite(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format(
            "[logic.cpp](IsFinite) computation failed, error={}", error));
    }

    // 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format(
            "[logic.cpp](IsFinite) sync device failed, error={}", error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

/// Check element-wise infinity of the input array.
NPUArray IsInf(const NPUArray& x) {
    auto result = NPUArray(x.shape, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    auto error = aclnnIsInfGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "[logic.cpp](IsInf) GetWorkspaceSize failed, error={}, dtype={}",
            error, std::string(py::str(py::cast<py::object>(x.dtype)))));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[logic.cpp](IsInf) aclrtMalloc failed, error={}", error));
        }
    }

    error = aclnnIsInf(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](IsInf) computation failed, error={}", error));
    }
{

}
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](IsInf) sync device failed, error={}", error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

/// Test element-wise for negative infinity (-inf).
NPUArray IsNegInf(const NPUArray& x) {
    auto result = NPUArray(x.shape, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;

    // 获取 workspace 大小与执行器
    auto error = aclnnIsNegInfGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("[logic.cpp](IsNegInf) GetWorkspaceSize failed, error={}", error));
    }

    // 分配 workspace
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[logic.cpp](IsNegInf) aclrtMalloc failed, error={}", error));
        }
    }

    // 创建执行流
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](IsNegInf) create stream failed, error={}", error));
    }

    // 执行计算
    error = aclnnIsNegInf(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](IsNegInf) computation failed, error={}", error));
    }

    // 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](IsNegInf) sync device failed, error={}", error));
    }

    // 释放资源
    aclrtDestroyStream(stream);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }{

    }
    return result;
}

/// Test element-wise for positive infinity (+inf).
NPUArray IsPosInf(const NPUArray& x) {
    auto result = NPUArray(x.shape, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;

    // 获取 workspace 大小与执行器
    auto error = aclnnIsPosInfGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("[logic.cpp](IsPosInf) GetWorkspaceSize failed, error={}", error));
    }

    // 分配 workspace
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[logic.cpp](IsPosInf) aclrtMalloc failed, error={}", error));
        }
    }

    // 创建执行流
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](IsPosInf) create stream failed, error={}", error));
    }

    // 执行计算
    error = aclnnIsPosInf(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](IsPosInf) computation failed, error={}", error));
    }

    // 同步设备（保持和其它算子一致）
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](IsPosInf) sync device failed, error={}", error));
    }

    // 释放资源
    aclrtDestroyStream(stream);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

/// Perform element-wise logical AND between two boolean arrays.
NPUArray LogicalAnd(const NPUArray& x, const NPUArray& y) {
    auto result = NPUArray(GetBroadcastShape(x, y), ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    auto error = aclnnLogicalAndGetWorkspaceSize(
        x.tensorPtr, y.tensorPtr, result.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "[logic.cpp](LogicalAnd) GetWorkspaceSize failed, error={}, dtype_x={}, dtype_y={}",
            error,
            std::string(py::str(py::cast<py::object>(x.dtype))),
            std::string(py::str(py::cast<py::object>(y.dtype)))
        ));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[logic.cpp](LogicalAnd) aclrtMalloc failed, error={}", error));
        }
    }

    error = aclnnLogicalAnd(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](LogicalAnd) computation failed, error={}", error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](LogicalAnd) sync device failed, error={}", error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

/// Perform element-wise logical OR between two boolean arrays.
NPUArray LogicalOr(const NPUArray& x, const NPUArray& y) {
    auto result = NPUArray(GetBroadcastShape(x, y), ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    auto error = aclnnLogicalOrGetWorkspaceSize(
        x.tensorPtr, y.tensorPtr, result.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "[logic.cpp](LogicalOr) GetWorkspaceSize failed, error={}, dtype_x={}, dtype_y={}",
            error,
            std::string(py::str(py::cast<py::object>(x.dtype))),
            std::string(py::str(py::cast<py::object>(y.dtype)))
        ));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[logic.cpp](LogicalOr) aclrtMalloc failed, error={}", error));
        }
    }

    error = aclnnLogicalOr(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](LogicalOr) computation failed, error={}", error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](LogicalOr) sync device failed, error={}", error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

/// Perform element-wise logical NOT on a boolean array.
NPUArray LogicalNot(const NPUArray& x) {
    auto result = NPUArray(x.shape, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    auto error = aclnnLogicalNotGetWorkspaceSize(
        x.tensorPtr, result.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "[logic.cpp](LogicalNot) GetWorkspaceSize failed, error={}, dtype={}",
            error,
            std::string(py::str(py::cast<py::object>(x.dtype)))
        ));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[logic.cpp](LogicalNot) aclrtMalloc failed, error={}", error));
        }
    }

    error = aclnnLogicalNot(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](LogicalNot) computation failed, error={}", error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](LogicalNot) sync device failed, error={}", error));
    }{
     }


    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

/// Perform element-wise logical XOR between two boolean arrays.
NPUArray LogicalXor(const NPUArray& x, const NPUArray& y) {
    auto result = NPUArray(GetBroadcastShape(x, y), ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    auto error = aclnnLogicalXorGetWorkspaceSize(
        x.tensorPtr, y.tensorPtr, result.tensorPtr,
        &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "[logic.cpp](LogicalXor) GetWorkspaceSize failed, error={}, dtype_x={}, dtype_y={}",
            error,
            std::string(py::str(py::cast<py::object>(x.dtype))),
            std::string(py::str(py::cast<py::object>(y.dtype)))
        ));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[logic.cpp](LogicalXor) aclrtMalloc failed, error={}", error));
        }
    }

    error = aclnnLogicalXor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](LogicalXor) computation failed, error={}", error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        throw std::runtime_error(fmt::format("[logic.cpp](LogicalXor) sync device failed, error={}", error));
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return result;
}

/// Element-wise greater-than comparison between two arrays.
NPUArray greater(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2),
                        dtype.value_or(py::dtype::of<bool>()));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGtTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](greater) aclnnGtTensorGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[logic.cpp](greater) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](greater) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnGtTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](greater) aclnnGtTensor error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }    
        std::string error_msg = "[logic.cpp](greater) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return out;
}

/// Element-wise greater-than comparison between an array and a scalar.
NPUArray greater(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](greater) Invalid scalar type: "
                                 + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGtScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater) aclnnGtScalarGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](greater) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](greater) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnGtScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        {
            aclDestroyScalar(acl_scalar);
        }
        std::string error_msg = "[logic.cpp](greater) aclnnGtScalar error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        {
            aclDestroyScalar(acl_scalar);
        }
        std::string error_msg = "[logic.cpp](greater) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    {
        aclDestroyScalar(acl_scalar);
    }
    return out;
}

/// Element-wise greater-than-or-equal comparison between two arrays.
NPUArray greater_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2),
                        dtype.value_or(py::dtype::of<bool>()));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGeTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](greater_equal) aclnnGeTensorGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[logic.cpp](greater_equal) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](greater_equal) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnGeTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](greater_equal) aclnnGeTensor error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](greater_equal) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return out;
}

/// Element-wise greater-than-or-equal comparison between an array and a scalar.
NPUArray greater_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](greater_equal) Invalid scalar type: "
                                 + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater_equal) aclnnGeScalarGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](greater_equal) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](greater_equal) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnGeScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater_equal) aclnnGeScalar error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater_equal) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(acl_scalar);
    return out;
}


/// Element-wise less-than comparison between two arrays.
NPUArray less(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2),
                        dtype.value_or(py::dtype::of<bool>()));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLtTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](less) aclnnLtTensorGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[logic.cpp](less) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](less) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnLtTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](less) aclnnLtTensor error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](less) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return out;
}

/// Element-wise less-than comparison between an array and a scalar.
NPUArray less(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](less) Invalid scalar type: "
                                 + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLtScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less) aclnnLtScalarGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](less) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](less) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnLtScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less) aclnnLtScalar error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(acl_scalar);
    return out;
}


/// Element-wise less-than-or-equal comparison between two arrays.
NPUArray less_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2),
                        dtype.value_or(py::dtype::of<bool>()));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLeTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](less_equal) aclnnLeTensorGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[logic.cpp](less_equal) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](less_equal) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnLeTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](less_equal) aclnnLeTensor error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](less_equal) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return out;
}

/// Element-wise less-than-or-equal comparison between an array and a scalar.
NPUArray less_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](less_equal) Invalid scalar type: "
                                 + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less_equal) aclnnLeScalarGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](less_equal) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](less_equal) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnLeScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less_equal) aclnnLeScalar error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less_equal) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(acl_scalar);
    return out;
}

/// Element-wise equality comparison between two arrays.
/// Note: aclnnEqual does not support broadcasting. Inputs must have identical shapes.
NPUArray equal(const NPUArray& x1, const NPUArray& x2,
               std::optional<py::dtype> dtype) {
    // 1. Shape check
    if (x1.shape != x2.shape) {
        throw std::runtime_error(
            "[logic.cpp](equal) Input shapes must match exactly. "
            "Broadcasting is not supported by aclnnEqual. "
            "Got shape mismatch.");
    }

    // 2. Allocate output
    auto out = NPUArray(x1.shape, dtype.value_or(py::dtype::of<bool>()));

    // 3. Query workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnEqualGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg =
            "[logic.cpp](equal) aclnnEqualGetWorkspaceSize error = " +
            std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    // 4. Allocate workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize,
                            ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error("[logic.cpp](equal) aclrtMalloc error = " +
                                     std::to_string(error));
        }
    }

    // 5. Execute
    error = aclnnEqual(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg =
            "[logic.cpp](equal) aclnnEqual error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    // 6. Sync
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error("[logic.cpp](equal) aclrtSynchronizeDevice error = " +
                                 std::to_string(error));
    }

    // 7. Free
    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/// Element-wise not-equal comparison between two arrays.
NPUArray not_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2),
                        dtype.value_or(py::dtype::of<bool>()));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNeTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](not_equal) aclnnNeTensorGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[logic.cpp](not_equal) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](not_equal) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnNeTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](not_equal) aclnnNeTensor error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        std::string error_msg = "[logic.cpp](not_equal) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return out;
}

/// Element-wise not-equal comparison between an array and a scalar.
NPUArray not_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](not_equal) Invalid scalar type: "
                                 + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](not_equal) aclnnNeScalarGetWorkspaceSize error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](not_equal) Invalid workspaceSize: "
                                 + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](not_equal) aclrtMalloc error = "
                                    + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnNeScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](not_equal) aclnnNeScalar error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](not_equal) aclrtSynchronizeDevice error = "
                                + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(acl_scalar);
    return out;
}

}
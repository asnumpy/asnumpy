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


#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <asnumpy/math/math.hpp>
#include <fmt/base.h>
#include <fmt/format.h>
#include <shape.h>
#include <stdexcept>

NPUArray Cumprod(const NPUArray& a, int64_t axis, py::dtype dtype) {
    auto shape = a.shape;
    auto axis_scalar = aclCreateScalar(&axis, ACL_INT64);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnCumprodGetWorkspaceSize(a.tensorPtr, axis_scalar, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnCumprod(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Cumsum(const NPUArray& a, int64_t axis, py::dtype dtype) {
    auto shape = a.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnCumsumGetWorkspaceSize(a.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnCumsum(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Nancumprod(const NPUArray& a, int64_t axis, py::dtype dtype) {
    auto shape = a.shape;
    auto axis_scalar = aclCreateScalar(&axis, ACL_INT64);
    float scalar = 1.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    error1 = aclrtSynchronizeDevice();
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnCumprodGetWorkspaceSize(temp.tensorPtr, axis_scalar, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error2 = aclnnCumprod(workspaceAddr2, workspaceSize2, executor2, nullptr);
    error2 = aclrtSynchronizeDevice();
    return result;
}

NPUArray Nancumsum(const NPUArray& a, int64_t axis, py::dtype dtype) {
    auto shape = a.shape;
    float scalar = 0.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    error1 = aclrtSynchronizeDevice();
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnCumsumGetWorkspaceSize(temp.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error2 = aclnnCumsum(workspaceAddr2, workspaceSize2, executor2, nullptr);
    error2 = aclrtSynchronizeDevice();
    return result;
}

NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axisa, int64_t axisb, int64_t axisc, int64_t axis) {
    auto broadcast = GetBroadcastShape(a, b);
    auto result = NPUArray(broadcast, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLinalgCrossGetWorkspaceSize(a.tensorPtr, b.tensorPtr, axis, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnLinalgCross(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Exp(const NPUArray& x, py::dtype dtype) {
    auto shape = x.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnExpGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnExp(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Expm1(const NPUArray& x, py::dtype dtype) {
    auto shape = x.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnExpm1GetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnExpm1(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Exp2(const NPUArray& x, py::dtype dtype) {
    auto shape = x.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnExp2GetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnExp2(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Log(const NPUArray& x, py::dtype dtype) {
    auto shape = x.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLogGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnLog(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Log10(const NPUArray& x, py::dtype dtype) {
    auto shape = x.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLog10GetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnLog10(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Log2(const NPUArray& x, py::dtype dtype) {
    auto shape = x.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLog2GetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnLog2(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Log1p(const NPUArray& x, py::dtype dtype) {
    auto shape = x.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLog1pGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnLog1p(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Logaddexp(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto broadcast = GetBroadcastShape(x1, x2);
    auto result = NPUArray(broadcast, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLogAddExpGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnLogAddExp(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Logaddexp2(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto broadcast = GetBroadcastShape(x1, x2);
    auto result = NPUArray(broadcast, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLogAddExp2GetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnLogAddExp2(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Sinc(const NPUArray& x) {
    auto shape = x.shape;
    auto result = NPUArray(shape, x.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnSincGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnSinc(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Real(const NPUArray& val) {
    auto shape = val.shape;
    auto result = NPUArray(shape, val.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnRealGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnReal(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Prod(const NPUArray& a, int64_t axis, py::dtype dtype, bool keepdims) {
    auto shape = a.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnProdDimGetWorkspaceSize(a.tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnProdDim(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Prod(const NPUArray& a, py::dtype dtype) {
    auto shape = (1, );
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnProdGetWorkspaceSize(a.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnProd(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Sum(const NPUArray& a, const std::vector<int64_t>& axis, py::dtype dtype, bool keepdims) {
    auto shape = a.shape;
    aclIntArray* axis_array = aclCreateIntArray(axis.data(), axis.size());
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnReduceSumGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnReduceSum(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

NPUArray Sum(const NPUArray& a, py::dtype dtype) {
    auto shape = a.shape;
    std::vector<aclTensor *> tmp{a};
    auto input = aclCreateTensorList(tmp.data(), tmp.size());
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnSumGetWorkspaceSize(input, temp.tensorPtr, &workspaceSize1, &executor1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error1 = aclnnSum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    error1 = aclrtSynchronizeDevice();
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnCastGetWorkspaceSize(temp.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error2 = aclnnCast(workspaceAddr2, workspaceSize2, executor2, nullptr);
    error2 = aclrtSynchronizeDevice();
    return result;
}

NPUArray Nanprod(const NPUArray& a, int64_t axis, py::dtype dtype, bool keepdims) {
    auto shape = a.shape;
    float scalar = 1.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    error1 = aclrtSynchronizeDevice();
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnProdDimGetWorkspaceSize(temp.tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error2 = aclnnProdDim(workspaceAddr2, workspaceSize2, executor2, nullptr);
    error2 = aclrtSynchronizeDevice();
    return result;
}

NPUArray Nanprod(const NPUArray& a, py::dtype dtype) {
    auto shape = (1, );
    float scalar = 1.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    error1 = aclrtSynchronizeDevice();
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnProdGetWorkspaceSize(temp.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error2 = aclnnProd(workspaceAddr2, workspaceSize2, executor2, nullptr);
    error2 = aclrtSynchronizeDevice();
    return result;
}

NPUArray Nansum(const NPUArray& a, const std::vector<int64_t>& axis, py::dtype dtype, bool keepdims) {
    auto shape = a.shape;
    float scalar = 0.0;
    aclIntArray* axis_array = aclCreateIntArray(axis.data(), axis.size());
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    error1 = aclrtSynchronizeDevice();
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnReduceSumGetWorkspaceSize(temp.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error2 = aclnnReduceSum(workspaceAddr2, workspaceSize2, executor2, nullptr);
    error2 = aclrtSynchronizeDevice();
    return result;
}

NPUArray Nansum(const NPUArray& a, py::dtype dtype) {
    auto shape = a.shape;
    float scalar = 0.0;
    auto temp1 = NPUArray(shape, a.aclDtype);
    auto temp2 = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp1.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnNanToNum(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    
    std::vector<aclTensor *> tmp{temp1};
    auto input = aclCreateTensorList(tmp.data(), tmp.size());
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnSumGetWorkspaceSize(tmp, temp2.tensorPtr, &workspaceSize1, &executor1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error1 = aclnnSum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    error1 = aclrtSynchronizeDevice();
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnCastGetWorkspaceSize(temp2.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error2 = aclnnCast(workspaceAddr2, workspaceSize2, executor2, nullptr);
    error2 = aclrtSynchronizeDevice();
    return result;
}


/**
 * @brief Compute element-wise square root of sum of squares of two arrays.
 * 
 * Equivalent to numpy.hypot(a, b), computes √(a² + b²) for each element.
 * This is the hypotenuse of a right-angled triangle with legs a and b.
 * 
 * @param a NPUArray, first input array (one leg of the right triangle)
 * @param b NPUArray, second input array (other leg of the right triangle)
 * @return NPUArray Element-wise result of √(a² + b²)
 */
NPUArray Hypot(const NPUArray& a, const NPUArray& b) {
    // 检查输入形状是否匹配
    if (a.shape != b.shape) {
        throw invalid_argument("Hypot: a and b must have the same shape");
    }

    // 初始化结果数组
    auto shape = a.shape;
    auto dtype = a.dtype;
    NPUArray result(shape, dtype);

    // 步骤1: 计算a的平方 (a²)
    NPUArray a_squared(shape, dtype);
    uint64_t a_sq_workspace_size = 0;
    aclOpExecutor* a_sq_executor = nullptr;
    auto error = aclnnMulGetWorkspaceSize(
        a.tensorPtr, a.tensorPtr,
        a_squared.tensorPtr,
        &a_sq_workspace_size,
        &a_sq_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Hypot: a² workspace size failed, error={}", error));
    }

    void* a_sq_workspace = nullptr;
    if (a_sq_workspace_size > 0) {
        error = aclrtMalloc(&a_sq_workspace, a_sq_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Hypot: a² workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(a_sq_workspace, a_sq_workspace_size, a_sq_executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Hypot: a² computation failed, error={}", error));
    }

    // 步骤2: 计算b的平方 (b²)
    NPUArray b_squared(shape, dtype);
    uint64_t b_sq_workspace_size = 0;
    aclOpExecutor* b_sq_executor = nullptr;
    error = aclnnMulGetWorkspaceSize(
        b.tensorPtr, b.tensorPtr,
        b_squared.tensorPtr,
        &b_sq_workspace_size,
        &b_sq_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Hypot: b² workspace size failed, error={}", error));
    }

    void* b_sq_workspace = nullptr;
    if (b_sq_workspace_size > 0) {
        error = aclrtMalloc(&b_sq_workspace, b_sq_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Hypot: b² workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(b_sq_workspace, b_sq_workspace_size, b_sq_executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Hypot: b² computation failed, error={}", error));
    }

    // 步骤3: 计算平方和 (a² + b²)
    NPUArray sum_squares(shape, dtype);
    uint64_t add_workspace_size = 0;
    aclOpExecutor* add_executor = nullptr;
    int32_t alpha = 1;
    auto alpha_scalar = aclCreateScalar(&alpha, a.aclDtype);
    
    error = aclnnAddGetWorkspaceSize(
        a_squared.tensorPtr, b_squared.tensorPtr,
        alpha_scalar, sum_squares.tensorPtr,
        &add_workspace_size, &add_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Hypot: sum workspace size failed, error={}", error));
    }

    void* add_workspace = nullptr;
    if (add_workspace_size > 0) {
        error = aclrtMalloc(&add_workspace, add_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Hypot: sum workspace malloc failed, error={}", error));
        }
    }

    error = aclnnAdd(add_workspace, add_workspace_size, add_executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Hypot: sum computation failed, error={}", error));
    }

    // 步骤4: 计算平方根 (√(a² + b²))
    uint64_t sqrt_workspace_size = 0;
    aclOpExecutor* sqrt_executor = nullptr;
    error = aclnnSqrtGetWorkspaceSize(
        sum_squares.tensorPtr, result.tensorPtr,
        &sqrt_workspace_size, &sqrt_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Hypot: sqrt workspace size failed, error={}", error));
    }

    void* sqrt_workspace = nullptr;
    if (sqrt_workspace_size > 0) {
        error = aclrtMalloc(&sqrt_workspace, sqrt_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Hypot: sqrt workspace malloc failed, error={}", error));
        }
    }

    error = aclnnSqrt(sqrt_workspace, sqrt_workspace_size, sqrt_executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Hypot: sqrt computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    aclDestroyScalar(alpha_scalar);
    if (a_sq_workspace) aclrtFree(a_sq_workspace);
    if (b_sq_workspace) aclrtFree(b_sq_workspace);
    if (add_workspace) aclrtFree(add_workspace);
    if (sqrt_workspace) aclrtFree(sqrt_workspace);

    return result;
}


/**
 * @brief Compute element-wise arctangent of y/x considering quadrant.
 * 
 * Equivalent to numpy.arctan2(y, x), returns values in [-π, π] radians.
 * 
 * @param y NPUArray, numerator (y-coordinate)
 * @param x NPUArray, denominator (x-coordinate)
 * @return NPUArray Result of element-wise arctan2(y, x)
 */
NPUArray Arctan2(const NPUArray& y, const NPUArray& x) {
    // 检查输入形状是否匹配
    if (y.shape != x.shape) {
        throw invalid_argument("Arctan2: y and x must have the same shape");
    }

    // 初始化结果数组
    auto shape = y.shape;
    auto dtype = y.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAtan2GetWorkspaceSize(
        y.tensorPtr, x.tensorPtr,
        result.tensorPtr,
        &workspace_size, &executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Arctan2: workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Arctan2: workspace malloc failed, error={}", error));
        }
    }

    // 执行计算
    error = aclnnAtan2(workspace, workspace_size, executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Arctan2: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace) aclrtFree(workspace);

    return result;
}


/**
 * @brief Convert angles from degrees to radians.
 * 
 * Equivalent to numpy.radians(x), computes x * (π / 180).
 * 
 * @param x NPUArray, input angles in degrees
 * @return NPUArray Angles converted to radians
 */
NPUArray Radians(const NPUArray& x) {
    // 初始化结果数组
    auto shape = x.shape;
    auto dtype = x.dtype;
    auto acl_dtype = x.aclDtype;
    NPUArray result(shape, dtype);

    // 1. 创建转换系数标量 (π / 180)
    void* scalar_data = nullptr;
    if (acl_dtype == ACL_FLOAT) {
        static const float val = static_cast<float>(M_PI / 180.0);
        scalar_data = const_cast<float*>(&val);
    } else if (acl_dtype == ACL_DOUBLE) {
        static const double val = M_PI / 180.0;
        scalar_data = const_cast<double*>(&val);
    } else {
        throw invalid_argument(fmt::format("Radians: unsupported dtype={}", acl_dtype));
    }

    // 2. 将标量转换为1D张量（解决参数类型不匹配问题）
    aclTensorDesc* scalar_desc = aclCreateTensorDesc(acl_dtype, 0, nullptr);  // 0维标量描述符
    aclTensor* scalar_tensor = nullptr;
    auto error = aclCreateTensorWithData(scalar_desc, scalar_data, 
                                        aclGetDataTypeSize(acl_dtype), 
                                        ACL_MEMCPY_HOST_TO_DEVICE, 
                                        &scalar_tensor);
    if (error != ACL_SUCCESS) {
        aclDestroyTensorDesc(scalar_desc);
        throw runtime_error(fmt::format("Radians: create scalar tensor failed, error={}", error));
    }

    // 3. 获取工作空间大小（现在第二个参数是aclTensor*类型，匹配接口要求）
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    error = aclnnMulGetWorkspaceSize(
        x.tensorPtr,       // 输入张量
        scalar_tensor,     // 转换为张量的标量（修复类型不匹配）
        result.tensorPtr,  // 输出张量
        &workspace_size, 
        &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyTensor(scalar_tensor);
        aclDestroyTensorDesc(scalar_desc);
        throw runtime_error(fmt::format("Radians: workspace size failed, error={}", error));
    }

    // 4. 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyTensor(scalar_tensor);
            aclDestroyTensorDesc(scalar_desc);
            throw runtime_error(fmt::format("Radians: workspace malloc failed, error={}", error));
        }
    }

    // 5. 执行计算 (x * (π / 180))
    error = aclnnMul(workspace, workspace_size, executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(workspace);
        aclDestroyTensor(scalar_tensor);
        aclDestroyTensorDesc(scalar_desc);
        throw runtime_error(fmt::format("Radians: computation failed, error={}", error));
    }

    // 6. 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(scalar_tensor);
    aclDestroyTensorDesc(scalar_desc);

    return result;
}


/**
 * @brief Compute element-wise hyperbolic sine.
 * 
 * Equivalent to numpy.sinh(x), calculates sinh(x) = (e^x - e^(-x))/2 for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic sine of x
 */
NPUArray Sinh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

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
        throw runtime_error(fmt::format("Sinh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Sinh: malloc workspace failed, error={}", error));
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
        throw runtime_error(fmt::format("Sinh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise hyperbolic cosine.
 * 
 * Equivalent to numpy.cosh(x), calculates cosh(x) = (e^x + e^(-x))/2 for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic cosine of x
 */
NPUArray Cosh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

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
        throw runtime_error(fmt::format("Cosh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Cosh: malloc workspace failed, error={}", error));
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
        throw runtime_error(fmt::format("Cosh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise hyperbolic tangent.
 * 
 * Equivalent to numpy.tanh(x), calculates tanh(x) = sinh(x)/cosh(x) for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic tangent of x
 */
NPUArray Tanh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

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
        throw runtime_error(fmt::format("Tanh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Tanh: malloc workspace failed, error={}", error));
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
        throw runtime_error(fmt::format("Tanh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise inverse hyperbolic sine.
 * 
 * Equivalent to numpy.arcsinh(x), calculates arcsinh(x) = ln(x + √(x² + 1)) for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise inverse hyperbolic sine of x
 */
NPUArray Arcsinh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

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
        throw runtime_error(fmt::format("Arcsinh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Arcsinh: malloc workspace failed, error={}", error));
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
        throw runtime_error(fmt::format("Arcsinh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise inverse hyperbolic cosine.
 * 
 * Equivalent to numpy.arccosh(x), calculates arccosh(x) = ln(x + √(x² - 1)) for x ≥ 1.
 * 
 * @param x NPUArray, input array (must contain values ≥ 1)
 * @return NPUArray Element-wise inverse hyperbolic cosine of x
 */
NPUArray Arccosh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

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
        throw runtime_error(fmt::format("Arccosh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Arccosh: malloc workspace failed, error={}", error));
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
        throw runtime_error(fmt::format("Arccosh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise inverse hyperbolic tangent.
 * 
 * Equivalent to numpy.arctanh(x), calculates arctanh(x) = 0.5*ln((1+x)/(1-x)) for |x| < 1.
 * 
 * @param x NPUArray, input array (must contain values with absolute value < 1)
 * @return NPUArray Element-wise inverse hyperbolic tangent of x
 */
NPUArray Arctanh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

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
        throw runtime_error(fmt::format("Arctanh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Arctanh: malloc workspace failed, error={}", error));
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
        throw runtime_error(fmt::format("Arctanh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise ceiling of the input.
 * 
 * Equivalent to numpy.ceil(x), returns the smallest integer greater than or equal to each element.
 * 
 * @param x NPUArray, input array (floating-point type)
 * @return NPUArray Element-wise ceiling values of x
 */
NPUArray Ceil(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

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
        throw runtime_error(fmt::format("Ceil: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Ceil: malloc workspace failed, error={}", error));
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
        throw runtime_error(fmt::format("Ceil: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise truncation of the input.
 * 
 * Equivalent to numpy.trunc(x), returns the integer part of each element by removing fractional parts.
 * 
 * @param x NPUArray, input array (floating-point type)
 * @return NPUArray Element-wise truncated values of x
 */
NPUArray Trunc(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

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
        throw runtime_error(fmt::format("Trunc: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Trunc: malloc workspace failed, error={}", error));
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
        throw runtime_error(fmt::format("Trunc: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise sign bit check.
 * 
 * Equivalent to numpy.signbit(x), returns a boolean array indicating whether the sign bit is set (negative values).
 * 
 * @param x NPUArray, input array (numeric type)
 * @return NPUArray Boolean array where True indicates negative elements (sign bit set)
 */
NPUArray Signbit(const NPUArray& x) {
    // 初始化结果数组（形状与输入一致，数据类型为布尔型）
    auto shape = x.shape;
    NPUArray result(shape, ACL_BOOL);  // 布尔型输出（True表示负数）

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSignBitGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Signbit: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Signbit: malloc workspace failed, error={}", error));
        }
    }

    // 执行符号位检查（检测是否为负数）
    error = aclnnSignBit(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Signbit: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise least common multiple (LCM).
 * 
 * Equivalent to numpy.lcm(x1, x2), returns the smallest positive integer divisible by both x1 and x2.
 * 
 * @param x1 NPUArray, input array (integer type)
 * @param x2 NPUArray, input array (integer type)
 * @return NPUArray Element-wise LCM of x1 and x2
 */
/**
 * @brief Compute element-wise least common multiple (LCM).
 * 
 * Equivalent to numpy.lcm(x1, x2), returns the smallest positive integer divisible by both x1 and x2.
 * Implemented using the relationship: LCM(a, b) = |a * b| / GCD(a, b)
 * 
 * @param x1 NPUArray, input array (integer type)
 * @param x2 NPUArray, input array (integer type)
 * @return NPUArray Element-wise LCM of x1 and x2
 */
NPUArray Lcm(const NPUArray& x1, const NPUArray& x2) {
    // 检查输入形状是否匹配
    if (x1.shape != x2.shape) {
        throw invalid_argument("Lcm: x1 and x2 must have the same shape");
    }

    // 初始化中间结果和最终结果数组
    auto shape = x1.shape;
    auto dtype = x1.dtype;
    auto acl_dtype = x1.aclDtype;

    // 步骤1: 计算x1和x2的乘积 (a * b)
    NPUArray product(shape, dtype);
    uint64_t mul_workspace_size = 0;
    aclOpExecutor* mul_executor = nullptr;
    auto error = aclnnMulGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr,
        product.tensorPtr,
        &mul_workspace_size, &mul_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Lcm: product workspace size failed, error={}", error));
    }

    void* mul_workspace = nullptr;
    if (mul_workspace_size > 0) {
        error = aclrtMalloc(&mul_workspace, mul_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Lcm: product workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(mul_workspace, mul_workspace_size, mul_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        throw runtime_error(fmt::format("Lcm: product computation failed, error={}", error));
    }

    // 步骤2: 计算x1和x2的绝对值乘积 (|a * b|)
    NPUArray abs_product(shape, dtype);
    uint64_t abs_workspace_size = 0;
    aclOpExecutor* abs_executor = nullptr;
    error = aclnnAbsGetWorkspaceSize(
        product.tensorPtr, abs_product.tensorPtr,
        &abs_workspace_size, &abs_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        throw runtime_error(fmt::format("Lcm: abs workspace size failed, error={}", error));
    }

    void* abs_workspace = nullptr;
    if (abs_workspace_size > 0) {
        error = aclrtMalloc(&abs_workspace, abs_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(mul_workspace);
            throw runtime_error(fmt::format("Lcm: abs workspace malloc failed, error={}", error));
        }
    }

    error = aclnnAbs(abs_workspace, abs_workspace_size, abs_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        aclrtFree(abs_workspace);
        throw runtime_error(fmt::format("Lcm: abs computation failed, error={}", error));
    }

    // 步骤3: 计算x1和x2的最大公约数 (GCD(a, b))
    NPUArray gcd_result = Gcd(x1, x2);  // 复用已实现的Gcd函数

    // 步骤4: 计算LCM = |a*b| / GCD(a,b)
    NPUArray result(shape, dtype);
    uint64_t div_workspace_size = 0;
    aclOpExecutor* div_executor = nullptr;
    error = aclnnDivGetWorkspaceSize(
        abs_product.tensorPtr, gcd_result.tensorPtr,
        result.tensorPtr,
        &div_workspace_size, &div_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        aclrtFree(abs_workspace);
        throw runtime_error(fmt::format("Lcm: division workspace size failed, error={}", error));
    }

    void* div_workspace = nullptr;
    if (div_workspace_size > 0) {
        error = aclrtMalloc(&div_workspace, div_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(mul_workspace);
            aclrtFree(abs_workspace);
            throw runtime_error(fmt::format("Lcm: division workspace malloc failed, error={}", error));
        }
    }

    error = aclnnDiv(div_workspace, div_workspace_size, div_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        aclrtFree(abs_workspace);
        aclrtFree(div_workspace);
        throw runtime_error(fmt::format("Lcm: division computation failed, error={}", error));
    }

    // 同步设备并释放所有资源
    aclrtSynchronizeDevice();
    aclrtFree(mul_workspace);
    aclrtFree(abs_workspace);
    aclrtFree(div_workspace);

    return result;
}
    

/**
 * @brief Compute element-wise greatest common divisor (GCD).
 * 
 * Equivalent to numpy.gcd(x1, x2), returns the largest positive integer dividing both x1 and x2.
 * 
 * @param x1 NPUArray, input array (integer type)
 * @param x2 NPUArray, input array (integer type)
 * @return NPUArray Element-wise GCD of x1 and x2
 */
NPUArray Gcd(const NPUArray& x1, const NPUArray& x2) {
    // 检查输入形状是否匹配
    if (x1.shape != x2.shape) {
        throw invalid_argument("Gcd: x1 and x2 must have the same shape");
    }

    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x1.shape;
    auto dtype = x1.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGcdGetWorkspaceSize(
        x1.tensorPtr,
        x2.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Gcd: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Gcd: malloc workspace failed, error={}", error));
        }
    }

    // 执行最大公约数计算
    error = aclnnGcd(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Gcd: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise floating-point power.
 * 
 * Equivalent to numpy.float_power(x1, x2), computes x1 raised to the power of x2 using floating-point arithmetic.
 * Implemented using the mathematical identity: x1^x2 = exp(x2 * ln(x1))
 * 
 * @param x1 NPUArray, base array (floating-point type)
 * @param x2 NPUArray, exponent array (floating-point type)
 * @return NPUArray Element-wise result of x1^x2
 */
NPUArray FloatPower(const NPUArray& x1, const NPUArray& x2) {
    // 检查输入形状是否匹配
    if (x1.shape != x2.shape) {
        throw invalid_argument("FloatPower: x1 and x2 must have the same shape");
    }

    // 初始化中间结果和最终结果数组
    auto shape = x1.shape;
    auto dtype = x1.dtype;
    NPUArray result(shape, dtype);

    // 步骤1: 计算x1的自然对数 (ln(x1))
    NPUArray log_x1(shape, dtype);
    uint64_t log_workspace_size = 0;
    aclOpExecutor* log_executor = nullptr;
    auto error = aclnnLogGetWorkspaceSize(
        x1.tensorPtr,
        log_x1.tensorPtr,
        &log_workspace_size,
        &log_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("FloatPower: log workspace size failed, error={}", error));
    }

    void* log_workspace = nullptr;
    if (log_workspace_size > 0) {
        error = aclrtMalloc(&log_workspace, log_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("FloatPower: log workspace malloc failed, error={}", error));
        }
    }

    error = aclnnLog(log_workspace, log_workspace_size, log_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(log_workspace);
        throw runtime_error(fmt::format("FloatPower: log computation failed, error={}", error));
    }

    // 步骤2: 计算x2与ln(x1)的乘积 (x2 * ln(x1))
    NPUArray product(shape, dtype);
    uint64_t mul_workspace_size = 0;
    aclOpExecutor* mul_executor = nullptr;
    error = aclnnMulGetWorkspaceSize(
        x2.tensorPtr,
        log_x1.tensorPtr,
        product.tensorPtr,
        &mul_workspace_size,
        &mul_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(log_workspace);
        throw runtime_error(fmt::format("FloatPower: mul workspace size failed, error={}", error));
    }

    void* mul_workspace = nullptr;
    if (mul_workspace_size > 0) {
        error = aclrtMalloc(&mul_workspace, mul_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(log_workspace);
            throw runtime_error(fmt::format("FloatPower: mul workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(mul_workspace, mul_workspace_size, mul_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(log_workspace);
        aclrtFree(mul_workspace);
        throw runtime_error(fmt::format("FloatPower: mul computation failed, error={}", error));
    }

    // 步骤3: 计算指数函数 (exp(x2 * ln(x1)) = x1^x2)
    uint64_t exp_workspace_size = 0;
    aclOpExecutor* exp_executor = nullptr;
    error = aclnnExpGetWorkspaceSize(
        product.tensorPtr,
        result.tensorPtr,
        &exp_workspace_size,
        &exp_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(log_workspace);
        aclrtFree(mul_workspace);
        throw runtime_error(fmt::format("FloatPower: exp workspace size failed, error={}", error));
    }

    void* exp_workspace = nullptr;
    if (exp_workspace_size > 0) {
        error = aclrtMalloc(&exp_workspace, exp_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(log_workspace);
            aclrtFree(mul_workspace);
            throw runtime_error(fmt::format("FloatPower: exp workspace malloc failed, error={}", error));
        }
    }

    error = aclnnExp(exp_workspace, exp_workspace_size, exp_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(log_workspace);
        aclrtFree(mul_workspace);
        aclrtFree(exp_workspace);
        throw runtime_error(fmt::format("FloatPower: exp computation failed, error={}", error));
    }

    // 同步设备并释放所有资源
    aclrtSynchronizeDevice();
    aclrtFree(log_workspace);
    aclrtFree(mul_workspace);
    aclrtFree(exp_workspace);

    return result;
}


/**
 * @brief Compute element-wise floating-point remainder of division.
 * 
 * Equivalent to numpy.fmod(x1, x2), returns the remainder of x1 divided by x2.
 * The result has the same sign as x1, following the formula:
 * fmod(x1, x2) = x1 - x2 * floor(x1 / x2)
 * 
 * @param x1 NPUArray, dividend array (floating-point type)
 * @param x2 NPUArray, divisor array (floating-point type)
 * @return NPUArray Element-wise remainder of x1 / x2
 */
NPUArray Fmod(const NPUArray& x1, const NPUArray& x2) {
    // 检查输入形状是否匹配
    if (x1.shape != x2.shape) {
        throw invalid_argument("Fmod: x1 and x2 must have the same shape");
    }

    // 初始化中间结果和最终结果数组
    auto shape = x1.shape;
    auto dtype = x1.dtype;

    // 步骤1: 计算x1 / x2（浮点数除法）
    NPUArray division(shape, dtype);
    uint64_t div_workspace_size = 0;
    aclOpExecutor* div_executor = nullptr;
    auto error = aclnnDivGetWorkspaceSize(
        x1.tensorPtr,
        x2.tensorPtr,
        division.tensorPtr,
        &div_workspace_size,
        &div_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Fmod: division workspace size failed, error={}", error));
    }

    void* div_workspace = nullptr;
    if (div_workspace_size > 0) {
        error = aclrtMalloc(&div_workspace, div_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Fmod: division workspace malloc failed, error={}", error));
        }
    }

    error = aclnnDiv(div_workspace, div_workspace_size, div_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        throw runtime_error(fmt::format("Fmod: division computation failed, error={}", error));
    }

    // 步骤2: 对除法结果取 floor（向负无穷方向取整）
    NPUArray floor_div(shape, dtype);
    uint64_t floor_workspace_size = 0;
    aclOpExecutor* floor_executor = nullptr;
    error = aclnnFloorGetWorkspaceSize(
        division.tensorPtr,
        floor_div.tensorPtr,
        &floor_workspace_size,
        &floor_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        throw runtime_error(fmt::format("Fmod: floor workspace size failed, error={}", error));
    }

    void* floor_workspace = nullptr;
    if (floor_workspace_size > 0) {
        error = aclrtMalloc(&floor_workspace, floor_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(div_workspace);
            throw runtime_error(fmt::format("Fmod: floor workspace malloc failed, error={}", error));
        }
    }

    error = aclnnFloor(floor_workspace, floor_workspace_size, floor_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(floor_workspace);
        throw runtime_error(fmt::format("Fmod: floor computation failed, error={}", error));
    }

    // 步骤3: 计算 x2 * floor(x1/x2)
    NPUArray product(shape, dtype);
    uint64_t mul_workspace_size = 0;
    aclOpExecutor* mul_executor = nullptr;
    error = aclnnMulGetWorkspaceSize(
        x2.tensorPtr,
        floor_div.tensorPtr,
        product.tensorPtr,
        &mul_workspace_size,
        &mul_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(floor_workspace);
        throw runtime_error(fmt::format("Fmod: multiplication workspace size failed, error={}", error));
    }

    void* mul_workspace = nullptr;
    if (mul_workspace_size > 0) {
        error = aclrtMalloc(&mul_workspace, mul_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(div_workspace);
            aclrtFree(floor_workspace);
            throw runtime_error(fmt::format("Fmod: multiplication workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(mul_workspace, mul_workspace_size, mul_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(floor_workspace);
        aclrtFree(mul_workspace);
        throw runtime_error(fmt::format("Fmod: multiplication computation failed, error={}", error));
    }

    // 步骤4: 计算最终结果 x1 - (x2 * floor(x1/x2))
    NPUArray result(shape, dtype);
    uint64_t sub_workspace_size = 0;
    aclOpExecutor* sub_executor = nullptr;
    error = aclnnSubGetWorkspaceSize(
        x1.tensorPtr,
        product.tensorPtr,
        result.tensorPtr,
        &sub_workspace_size,
        &sub_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(floor_workspace);
        aclrtFree(mul_workspace);
        throw runtime_error(fmt::format("Fmod: subtraction workspace size failed, error={}", error));
    }

    void* sub_workspace = nullptr;
    if (sub_workspace_size > 0) {
        error = aclrtMalloc(&sub_workspace, sub_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(div_workspace);
            aclrtFree(floor_workspace);
            aclrtFree(mul_workspace);
            throw runtime_error(fmt::format("Fmod: subtraction workspace malloc failed, error={}", error));
        }
    }

    error = aclnnSub(sub_workspace, sub_workspace_size, sub_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(floor_workspace);
        aclrtFree(mul_workspace);
        aclrtFree(sub_workspace);
        throw runtime_error(fmt::format("Fmod: subtraction computation failed, error={}", error));
    }

    // 同步设备并释放所有资源
    aclrtSynchronizeDevice();
    aclrtFree(div_workspace);
    aclrtFree(floor_workspace);
    aclrtFree(mul_workspace);
    aclrtFree(sub_workspace);

    return result;
}
    

/**
 * @brief Compute element-wise remainder of division.
 * 
 * Equivalent to numpy.mod(x1, x2), returns the remainder of x1 divided by x2.
 * The result has the same sign as x2, following the formula:
 * mod(x1, x2) = x1 - x2 * trunc(x1 / x2)
 * 
 * @param x1 NPUArray, dividend array (numeric type)
 * @param x2 NPUArray, divisor array (numeric type)
 * @return NPUArray Element-wise remainder of x1 / x2
 */
NPUArray Mod(const NPUArray& x1, const NPUArray& x2) {
    // 检查输入形状是否匹配
    if (x1.shape != x2.shape) {
        throw invalid_argument("Mod: x1 and x2 must have the same shape");
    }

    // 初始化中间结果和最终结果数组
    auto shape = x1.shape;
    auto dtype = x1.dtype;

    // 步骤1: 计算x1 / x2（浮点数除法）
    NPUArray division(shape, dtype);
    uint64_t div_workspace_size = 0;
    aclOpExecutor* div_executor = nullptr;
    auto error = aclnnDivGetWorkspaceSize(
        x1.tensorPtr,
        x2.tensorPtr,
        division.tensorPtr,
        &div_workspace_size,
        &div_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Mod: division workspace size failed, error={}", error));
    }

    void* div_workspace = nullptr;
    if (div_workspace_size > 0) {
        error = aclrtMalloc(&div_workspace, div_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Mod: division workspace malloc failed, error={}", error));
        }
    }

    error = aclnnDiv(div_workspace, div_workspace_size, div_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        throw runtime_error(fmt::format("Mod: division computation failed, error={}", error));
    }

    // 步骤2: 对除法结果取trunc（向零方向取整）
    NPUArray trunc_div(shape, dtype);
    uint64_t trunc_workspace_size = 0;
    aclOpExecutor* trunc_executor = nullptr;
    error = aclnnTruncGetWorkspaceSize(
        division.tensorPtr,
        trunc_div.tensorPtr,
        &trunc_workspace_size,
        &trunc_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        throw runtime_error(fmt::format("Mod: trunc workspace size failed, error={}", error));
    }

    void* trunc_workspace = nullptr;
    if (trunc_workspace_size > 0) {
        error = aclrtMalloc(&trunc_workspace, trunc_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(div_workspace);
            throw runtime_error(fmt::format("Mod: trunc workspace malloc failed, error={}", error));
        }
    }

    error = aclnnTrunc(trunc_workspace, trunc_workspace_size, trunc_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(trunc_workspace);
        throw runtime_error(fmt::format("Mod: trunc computation failed, error={}", error));
    }

    // 步骤3: 计算 x2 * trunc(x1/x2)
    NPUArray product(shape, dtype);
    uint64_t mul_workspace_size = 0;
    aclOpExecutor* mul_executor = nullptr;
    error = aclnnMulGetWorkspaceSize(
        x2.tensorPtr,
        trunc_div.tensorPtr,
        product.tensorPtr,
        &mul_workspace_size,
        &mul_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(trunc_workspace);
        throw runtime_error(fmt::format("Mod: multiplication workspace size failed, error={}", error));
    }

    void* mul_workspace = nullptr;
    if (mul_workspace_size > 0) {
        error = aclrtMalloc(&mul_workspace, mul_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(div_workspace);
            aclrtFree(trunc_workspace);
            throw runtime_error(fmt::format("Mod: multiplication workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(mul_workspace, mul_workspace_size, mul_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(trunc_workspace);
        aclrtFree(mul_workspace);
        throw runtime_error(fmt::format("Mod: multiplication computation failed, error={}", error));
    }

    // 步骤4: 计算最终结果 x1 - (x2 * trunc(x1/x2))
    NPUArray result(shape, dtype);
    uint64_t sub_workspace_size = 0;
    aclOpExecutor* sub_executor = nullptr;
    error = aclnnSubGetWorkspaceSize(
        x1.tensorPtr,
        product.tensorPtr,
        result.tensorPtr,
        &sub_workspace_size,
        &sub_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(trunc_workspace);
        aclrtFree(mul_workspace);
        throw runtime_error(fmt::format("Mod: subtraction workspace size failed, error={}", error));
    }

    void* sub_workspace = nullptr;
    if (sub_workspace_size > 0) {
        error = aclrtMalloc(&sub_workspace, sub_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(div_workspace);
            aclrtFree(trunc_workspace);
            aclrtFree(mul_workspace);
            throw runtime_error(fmt::format("Mod: subtraction workspace malloc failed, error={}", error));
        }
    }

    error = aclnnSub(sub_workspace, sub_workspace_size, sub_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(div_workspace);
        aclrtFree(trunc_workspace);
        aclrtFree(mul_workspace);
        aclrtFree(sub_workspace);
        throw runtime_error(fmt::format("Mod: subtraction computation failed, error={}", error));
    }

    // 同步设备并释放所有资源
    aclrtSynchronizeDevice();
    aclrtFree(div_workspace);
    aclrtFree(trunc_workspace);
    aclrtFree(mul_workspace);
    aclrtFree(sub_workspace);

    return result;
}


/**
 * @brief Decompose elements into integer and fractional parts.
 * 
 * Equivalent to numpy.modf(x), returns a pair of arrays (integer_part, fractional_part)
 * where each element is split into an integer component and a fractional component.
 * Both parts have the same sign as the input and the same data type.
 * 
 * @param x NPUArray, input array (floating-point type)
 * @return std::pair<NPUArray, NPUArray> Integer part and fractional part of x
 */
std::pair<NPUArray, NPUArray> Modf(const NPUArray& x) {
    auto shape = x.shape;
    auto dtype = x.dtype;

    // 步骤1: 计算输入的整数部分（向零取整）
    NPUArray integer_part(shape, dtype);
    uint64_t trunc_workspace_size = 0;
    aclOpExecutor* trunc_executor = nullptr;
    auto error = aclnnTruncGetWorkspaceSize(
        x.tensorPtr,
        integer_part.tensorPtr,
        &trunc_workspace_size,
        &trunc_executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Modf: trunc workspace size failed, error={}", error));
    }

    void* trunc_workspace = nullptr;
    if (trunc_workspace_size > 0) {
        error = aclrtMalloc(&trunc_workspace, trunc_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Modf: trunc workspace malloc failed, error={}", error));
        }
    }

    error = aclnnTrunc(trunc_workspace, trunc_workspace_size, trunc_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(trunc_workspace);
        throw runtime_error(fmt::format("Modf: trunc computation failed, error={}", error));
    }

    // 步骤2: 计算小数部分（输入 - 整数部分）
    NPUArray fractional_part(shape, dtype);
    uint64_t sub_workspace_size = 0;
    aclOpExecutor* sub_executor = nullptr;
    error = aclnnSubGetWorkspaceSize(
        x.tensorPtr,
        integer_part.tensorPtr,
        fractional_part.tensorPtr,
        &sub_workspace_size,
        &sub_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(trunc_workspace);
        throw runtime_error(fmt::format("Modf: subtraction workspace size failed, error={}", error));
    }

    void* sub_workspace = nullptr;
    if (sub_workspace_size > 0) {
        error = aclrtMalloc(&sub_workspace, sub_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(trunc_workspace);
            throw runtime_error(fmt::format("Modf: subtraction workspace malloc failed, error={}", error));
        }
    }

    error = aclnnSub(sub_workspace, sub_workspace_size, sub_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(trunc_workspace);
        aclrtFree(sub_workspace);
        throw runtime_error(fmt::format("Modf: subtraction computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    aclrtFree(trunc_workspace);
    aclrtFree(sub_workspace);

    // 返回整数部分和小数部分的 pair
    return {integer_part, fractional_part};
}


/**
 * @brief Compute element-wise remainder of division.
 * 
 * Equivalent to numpy.remainder(x1, x2), same as numpy.mod - returns x1 - x2 * floor(x1 / x2).
 * 
 * @param x1 NPUArray, dividend array (numeric type)
 * @param x2 NPUArray, divisor array (numeric type)
 * @return NPUArray Element-wise remainder result
 */
NPUArray Remainder(const NPUArray& x1, const NPUArray& x2) {
    // 与mod功能相同，复用aclMod接口
    return Mod(x1, x2);
}


/**
 * @brief Compute element-wise quotient and remainder.
 * 
 * Equivalent to numpy.divmod(x1, x2), returns a pair (quotient, remainder) where:
 * - quotient = floor(x1 / x2)
 * - remainder = x1 - x2 * quotient
 * 
 * @param x1 NPUArray, dividend array (numeric type)
 * @param x2 NPUArray, divisor array (numeric type)
 * @return pair<NPUArray, NPUArray> Pair of quotient and remainder
 */
pair<NPUArray, NPUArray> Divmod(const NPUArray& x1, const NPUArray& x2) {
    if (x1.shape != x2.shape) {
        throw invalid_argument("Divmod: x1 and x2 must have the same shape");
    }

    auto shape = x1.shape;
    auto dtype = x1.dtype;
    NPUArray quotient(shape, dtype);  // 商
    NPUArray remainder(shape, dtype);  // 余数

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnDivmodGetWorkspaceSize(
        x1.tensorPtr,
        x2.tensorPtr,
        quotient.tensorPtr,
        remainder.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Divmod: get workspace size failed, error={}", error));
    }

    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw runtime_error(fmt::format("Divmod: malloc workspace failed, error={}", error));
        }
    }

    error = aclnnDivmod(
        workspace,
        workspace_size,
        executor,
        nullptr
    );
    if (error != ACL_SUCCESS) {
        throw runtime_error(fmt::format("Divmod: computation failed, error={}", error));
    }

    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return {quotient, remainder};
}


/**
 * @brief Compute the sine of each element in the input array.
 * 
 * Creates an output array stored on NPU and calculates element-wise sine values
 * using the aclnnSin operator.
 * 
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise sine values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray sin(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSinGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](sin) aclnnSinGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) throw std::runtime_error("[math.cpp](sin) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](sin) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnSin(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](sin) aclnnSin error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](sin) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Compute the cosine of each element in the input array.
 * 
 * Creates an output array stored on NPU and calculates element-wise cosine values
 * using the aclnnCos operator.
 * 
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise cosine values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray cos(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnCosGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](cos) aclnnCosGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) throw std::runtime_error("[math.cpp](cos) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](cos) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnCos(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](cos) aclnnCos error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](cos) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Compute the tangent of each element in the input array.
 * 
 * Creates an output array stored on NPU and calculates element-wise tangent values
 * using the aclnnTan operator.
 * 
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise tangent values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray tan(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnTanGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](tan) aclnnTanGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) throw std::runtime_error("[math.cpp](tan) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](tan) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnTan(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](tan) aclnnTan error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](tan) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Compute the inverse sine (arcsin) of each element in the input array.
 *
 * Creates an output array on NPU and computes element-wise arcsin using aclnnAsin.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise arcsin values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray arcsin(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAsinGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arcsin) aclnnAsinGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](arcsin) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](arcsin) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnAsin(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arcsin) aclnnAsin error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arcsin) aclrtSynchronizeDevice error = " + std::to_string(error);
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

/**
 * @brief Compute the inverse cosine (arccos) of each element in the input array.
 *
 * Creates an output array on NPU and computes element-wise arccos using aclnnAcos.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise arccos values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray arccos(const NPUArray& x, py::dtype dtype) {

    auto out = NPUArray(x.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAcosGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arccos) aclnnAcosGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](arccos) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](arccos) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnAcos(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arccos) aclnnAcos error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arccos) aclrtSynchronizeDevice error = " + std::to_string(error);
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

/**
 * @brief Compute the element-wise arc tangent of input array.
 *
 * Creates a new array stored on NPU and applies aclnnAtan element-wise.
 *
 * @param x Input NPUArray.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array where each element is the arctangent of the corresponding input element.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Arctan(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto error = aclnnAtanGetWorkspaceSize(x.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("[math.cpp](arctan) aclnnAtanGetWorkspaceSize error = {}", error));
    }

    if(workspaceSize < 0) {
        throw std::runtime_error(fmt::format("[math.cpp](arctan) Invalid workspaceSize: {}", workspaceSize));
    }

    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[math.cpp](arctan) aclrtMalloc error = {}", error));
        }
    }

    error = aclnnAtan(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("[math.cpp](arctan) aclnnAtan error = {}", error));
    }

    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("[math.cpp](arctan) aclrtSynchronizeDevice error = {}", error));
    }

    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Round elements of the array to the given number of decimals.
 *
 * Creates an output array stored on NPU and rounds element-wise values
 * using the aclnnRoundDecimals operator.
 *
 * @param a Input array.
 * @param decimals Number of decimal places to round to (default 0).
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the specified decimals.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray around(const NPUArray& a, int decimals, py::dtype dtype) {
    auto out = NPUArray(a.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnRoundDecimalsGetWorkspaceSize(
        a.tensorPtr, decimals, out.tensorPtr, &workspaceSize, &executor
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

    error = aclnnRoundDecimals(workspaceAddr, workspaceSize, executor, nullptr);
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

/**
 * @brief Round elements of the array to the given number of decimals.
 *
 * This function is an alias of around(), provided for NumPy API compatibility.
 *
 * @param a Input array.
 * @param decimals Number of decimal places to round to (default 0).
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the specified decimals.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray round_(const NPUArray& a, int decimals, py::dtype dtype) {
    return around(a, decimals, dtype);
}

/**
 * @brief Round elements of the array to the nearest integer.
 *
 * Creates an output array stored on NPU and rounds element-wise values
 * to the nearest integer using the aclnnRound operator.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the nearest integer.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray rint(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

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

/**
 * @brief Truncate elements of the array towards zero.
 *
 * Creates an output array stored on NPU and applies element-wise truncation
 * using the aclnnTrunc operator.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with truncated values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray fix(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

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

/**
 * @brief Compute the floor of each element in the input array.
 *
 * Creates an output array stored on NPU and applies element-wise floor
 * using the aclnnFloor operator.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with floored values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray floor(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

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

/**
 * @brief Element-wise addition of two arrays.
 *
 * Creates an output array on NPU and computes element-wise addition using aclnnAdd.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise sums.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray add(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    // 输出形状应与广播结果一致；此处先以 x1.shape() 分配。
    // 如需显式广播形状，请在有统一工具后替换为广播形状。
    auto out = NPUArray(x1.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAddGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](add) aclnnAddGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](add) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](add) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnAdd(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](add) aclnnAdd error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](add) aclrtSynchronizeDevice error = " + std::to_string(error);
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

/**
 * @brief Compute the reciprocal (1/x) of each element in the input array.
 *
 * Creates an output array stored on NPU and applies element-wise reciprocal
 * using the aclnnReciprocal operator.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise reciprocals.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray reciprocal(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnReciprocalGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](reciprocal) aclnnReciprocalGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](reciprocal) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](reciprocal) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnReciprocal(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](reciprocal) aclnnReciprocal error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](reciprocal) aclrtSynchronizeDevice error = " + std::to_string(error);
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

/**
 * @brief Return the input array itself (with optional dtype conversion).
 *
 * Equivalent to applying the unary plus operator. No numerical change occurs.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Same array values, possibly with a new dtype.
 */
NPUArray positive(const NPUArray& x, py::dtype dtype) {
    // 如果 dtype 相同，直接返回一个拷贝；如果不同，进行 dtype 转换
    if (x.dtype() == dtype) {
        return NPUArray(x);  // 调用拷贝构造函数
    } else {
        return NPUArray(x, dtype);  // 使用已有构造逻辑做 dtype 转换
    }
}

/**
 * @brief Compute the numerical negative of each element in the input array.
 *
 * Creates an output array stored on NPU and applies element-wise negation
 * using the aclnnNeg operator.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise negated values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray negative(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNegGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](negative) aclnnNegGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](negative) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](negative) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnNeg(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](negative) aclnnNeg error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](negative) aclrtSynchronizeDevice error = " + std::to_string(error);
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

/**
 * @brief Element-wise multiplication of two arrays.
 *
 * Creates an output array on NPU and computes element-wise multiplication using aclnnMul.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise products.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray multiply(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    // 输出形状应与广播结果一致；此处先以 x1.shape() 分配。
    // 如需显式广播形状，请在有统一工具后替换为广播形状。
    auto out = NPUArray(x1.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMulGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](multiply) aclnnMulGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](multiply) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](multiply) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMul(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](multiply) aclnnMul error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](multiply) aclrtSynchronizeDevice error = " + std::to_string(error);
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

/**
 * @brief Element-wise true division of two arrays.
 *
 * Creates an output array on NPU and computes element-wise true division using aclnnDiv.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise quotients.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray divide(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    // 输出形状应与广播结果一致；此处先以 x1.shape() 分配。
    // 如需显式广播形状，请在有统一工具后替换为广播形状。
    auto out = NPUArray(x1.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnDivGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](divide) aclnnDivGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](divide) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](divide) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnDiv(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](divide) aclnnDiv error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](divide) aclrtSynchronizeDevice error = " + std::to_string(error);
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

/**
 * @brief Element-wise subtraction of two arrays.
 *
 * Creates an output array on NPU and computes element-wise subtraction using aclnnSub.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise differences.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray subtract(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    // 输出形状应与广播结果一致；此处先以 x1.shape() 分配。
    // 如需显式广播形状，请在有统一工具后替换为广播形状。
    auto out = NPUArray(x1.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSubGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](subtract) aclnnSubGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](subtract) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](subtract) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnSub(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](subtract) aclnnSub error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](subtract) aclrtSynchronizeDevice error = " + std::to_string(error);
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

/**
 * @brief Element-wise true division of two arrays (alias of divide).
 *
 * Provided for NumPy API compatibility.
 */
NPUArray true_divide(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    return divide(x1, x2, dtype);
}

/**
 * @brief Element-wise floor division of two arrays.
 *
 * Creates an output array on NPU and computes element-wise floor(x1 / x2)
 * using the aclnnFloorDivide operator.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise floor division results.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray floor_divide(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    // 输出形状应与广播结果一致；此处先以 x1.shape() 分配。
    // 如需显式广播形状，请在有统一工具后替换为广播形状。
    auto out = NPUArray(x1.shape(), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnFloorDivideGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](floor_divide) aclnnFloorDivideGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](floor_divide) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](floor_divide) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnFloorDivide(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](floor_divide) aclnnFloorDivide error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](floor_divide) aclrtSynchronizeDevice error = " + std::to_string(error);
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

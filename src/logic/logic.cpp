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
#include <asnumpy/utils/status_handler.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <fmt/core.h>

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
#include <aclnnop/aclnn_eq_tensor.h>
#include <aclnnop/aclnn_eq_scalar.h>
#include <aclnnop/aclnn_ne_scalar.h>
#include <aclnnop/aclnn_ne_tensor.h>

namespace asnumpy {

/**
 * @brief 为标量比较操作创建 aclScalar 对象
 *
 * 根据输入数组的数据类型和可选的输出数据类型，确定标量的目标数据类型，
 * 然后使用 CreateScalar 函数创建 aclScalar 对象。
 *
 * @param x1 输入数组
 * @param scalar Python 标量对象
 * @param dtype 可选的输出数据类型
 * @return aclScalar* 创建的标量对象
 */
static aclScalar* CreateScalarForComparison(
    const NPUArray& x1,
    const py::object& scalar,
    std::optional<py::dtype> dtype) {
    
    // 确定标量的数据类型：如果提供了dtype则使用dtype，否则使用输入数组的数据类型
    aclDataType scalar_dtype;
    if (dtype.has_value()) {
        scalar_dtype = NPUArray::GetACLDataType(*dtype);
    } else {
        scalar_dtype = x1.aclDtype;
    }
    
    return CreateScalar(scalar, scalar_dtype);
}

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
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnAll(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "All");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyIntArray(aclDim);

    return result;
}

/// Reduce array by logical AND operation over specified dimensions.
NPUArray All(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims) {
    std::vector<int64_t> shape = x.shape;
    if (keepdims) {
        for (int i=0; i<dim.size(); i++) {
            shape[dim[i]] = 1;
        }
    }
    else {
        std::vector<int64_t> dim_remove = dim;
        std::sort(dim_remove.begin(), dim_remove.end(), std::greater<int64_t>());
        for (int i=0; i<dim.size(); i++) {
            shape.erase(shape.begin() + dim_remove[i]);
        }
    }
    auto result = NPUArray(shape, ACL_BOOL);

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
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnAll(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "All");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
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
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnAny(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "Any");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyIntArray(aclDim);

    return result;
}

/// Reduce array by logical OR operation over specified dimensions.
NPUArray Any(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims) {
    std::vector<int64_t> shape = x.shape;
    if (keepdims) {
        for (int i=0; i<dim.size(); i++) {
            shape[dim[i]] = 1;
        }
    }
    else {
        std::vector<int64_t> dim_remove = dim;
        std::sort(dim_remove.begin(), dim_remove.end(), std::greater<int64_t>());
        for (int i=0; i<dim.size(); i++) {
            shape.erase(shape.begin() + dim_remove[i]);
        }
    }
    auto result = NPUArray(shape, ACL_BOOL);

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
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnAny(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "Any");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyIntArray(aclDim);

    return result;
}

/// Check element-wise finiteness of the input array.
NPUArray IsFinite(const NPUArray& x) {
    // 输出布尔数组，shape 与输入一致
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return ExecuteUnaryOp(
        x,
        dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnIsFiniteGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnIsFinite(workspace, workspaceSize, executor, nullptr);
        },
        "IsFinite"
    );
}

/// Check element-wise infinity of the input array.
NPUArray IsInf(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return ExecuteUnaryOp(
        x,
        dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnIsInfGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnIsInf(workspace, workspaceSize, executor, nullptr);
        },
        "IsInf"
    );
}

/// Test element-wise for negative infinity (-inf).
NPUArray IsNegInf(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return ExecuteUnaryOp(
        x,
        dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnIsNegInfGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnIsNegInf(workspace, workspaceSize, executor, nullptr);
        },
        "IsNegInf"
    );
}

/// Test element-wise for positive infinity (+inf).
NPUArray IsPosInf(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return ExecuteUnaryOp(
        x,
        dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnIsPosInfGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnIsPosInf(workspace, workspaceSize, executor, nullptr);
        },
        "IsPosInf"
    );
}

/// Perform element-wise logical AND between two boolean arrays.
NPUArray LogicalAnd(const NPUArray& x, const NPUArray& y) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return ExecuteBinaryOp(
        x, 
        y, 
        dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogicalAndGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogicalAnd(workspace, workspaceSize, executor, nullptr);
        },
        "LogicalAnd"
    );
}

/// Perform element-wise logical OR between two boolean arrays.
NPUArray LogicalOr(const NPUArray& x, const NPUArray& y) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return ExecuteBinaryOp(
        x, 
        y, 
        dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogicalOrGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogicalOr(workspace, workspaceSize, executor, nullptr);
        },
        "LogicalOr"
    );
}

/// Perform element-wise logical NOT on a boolean array.
NPUArray LogicalNot(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return ExecuteUnaryOp(
        x,
        dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogicalNotGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogicalNot(workspace, workspaceSize, executor, nullptr);
        },
        "LogicalNot"
    );
}

/// Perform element-wise logical XOR between two boolean arrays.
NPUArray LogicalXor(const NPUArray& x, const NPUArray& y) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return ExecuteBinaryOp(
        x, 
        y, 
        dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogicalXorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogicalXor(workspace, workspaceSize, executor, nullptr);
        },
        "LogicalXor"
    );
}

/// Element-wise greater-than comparison between two arrays.
NPUArray greater(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return ExecuteBinaryOp(
        x1, 
        x2, 
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnGtTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnGtTensor(workspace, workspaceSize, executor, nullptr);
        },
        "greater"
    );
}

/// Element-wise greater-than comparison between an array and a scalar.
NPUArray greater(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));
    
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGtScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnGtScalar(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "greater");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyScalar(acl_scalar);
    return out;
}

/// Element-wise greater-than-or-equal comparison between two arrays.
NPUArray greater_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return ExecuteBinaryOp(
        x1, 
        x2, 
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnGeTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnGeTensor(workspace, workspaceSize, executor, nullptr);
        },
        "greater_equal"
    );
}

/// Element-wise greater-than-or-equal comparison between an array and a scalar.
NPUArray greater_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));
    
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnGeScalar(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "greater_equal");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyScalar(acl_scalar);
    return out;
}


/// Element-wise less-than comparison between two arrays.
NPUArray less(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return ExecuteBinaryOp(
        x1, 
        x2, 
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLtTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLtTensor(workspace, workspaceSize, executor, nullptr);
        },
        "less"
    );
}

/// Element-wise less-than comparison between an array and a scalar.
NPUArray less(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));
    
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLtScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnLtScalar(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "less");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyScalar(acl_scalar);
    return out;
}


/// Element-wise less-than-or-equal comparison between two arrays.
NPUArray less_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return ExecuteBinaryOp(
        x1, 
        x2, 
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLeTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLeTensor(workspace, workspaceSize, executor, nullptr);
        },
        "less_equal"
    );
}

/// Element-wise less-than-or-equal comparison between an array and a scalar.
NPUArray less_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));
    
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnLeScalar(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "less_equal");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyScalar(acl_scalar);
    return out;
}

/// Element-wise equality comparison between two arrays.
NPUArray equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return ExecuteBinaryOp(
        x1, 
        x2, 
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnEqTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnEqTensor(workspace, workspaceSize, executor, nullptr);
        },
        "equal"
    );
}

/// Element-wise equal comparison between an array and a scalar.
NPUArray equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape, dtype.value_or(py::dtype::of<bool>()));
    
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnEqScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnEqScalar(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "equal");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyScalar(acl_scalar);
    return out;
}

/// Element-wise not-equal comparison between two arrays.
NPUArray not_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return ExecuteBinaryOp(
        x1, 
        x2, 
        out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnNeTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnNeTensor(workspace, workspaceSize, executor, nullptr);
        },
        "not_equal"
    );
}

/// Element-wise not-equal comparison between an array and a scalar.
NPUArray not_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    auto out = NPUArray(x1.shape,
                        dtype.value_or(py::dtype::of<bool>()));
    
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace workspace(workspaceSize);

    error = aclnnNeScalar(workspace.get(), workspaceSize, executor, nullptr);
    CheckExecuteAclnnStatus(error, "not_equal");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    aclDestroyScalar(acl_scalar);
    return out;
}

}
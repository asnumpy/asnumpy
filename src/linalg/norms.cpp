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


#include <asnumpy/linalg/norms.hpp>
#include <asnumpy/utils/status_handler.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_norm.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_slogdet.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_cast.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

using namespace asnumpy;

namespace {

// Helper: cast NPUArray to target dtype
NPUArray CastToDtype(const NPUArray& input, aclDataType targetDtype) {
    auto result = NPUArray(input.shape, targetDtype);
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    auto err = aclnnCastGetWorkspaceSize(input.tensorPtr, targetDtype, result.tensorPtr,
        &wsSize, &exec);
    CheckGetWorkspaceSizeAclnnStatus(err);
    AclWorkspace ws(wsSize);
    err = aclnnCast(ws.get(), wsSize, exec, nullptr);
    CheckAclnnStatus(err, "aclnnCast error");
    err = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(err);
    return result;
}

} // anonymous namespace

NPUArray Linalg_Norm(const NPUArray& a, double ord, const std::vector<int64_t>& axis, bool keepdims) {
    auto shape = a.shape;
    if (keepdims) {
        for (int i=0; i<axis.size(); i++) {
            shape[axis[i]] = 1;
        }
    }
    else {
        for (int i=0; i<axis.size(); i++) {
            shape.erase(shape.begin() + axis[i]);
        }
    }
    auto ord_scalar = aclCreateScalar(&ord, ACL_DOUBLE);
    aclIntArray* axis_array = aclCreateIntArray(axis.data(), axis.size());
    auto result = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNormGetWorkspaceSize(a.tensorPtr, ord_scalar, axis_array, keepdims, result.tensorPtr, 
        &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    AclWorkspace workspace(workspaceSize);
    error = aclnnNorm(workspace.get(), workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnNorm error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Linalg_Det(const NPUArray& a) {
    std::vector<int64_t> shape = a.shape;
    shape.erase(shape.end() - 2, shape.end());

    // Cast input to double for numerical stability in LU decomposition
    NPUArray aDouble = (a.aclDtype != ACL_DOUBLE) ? CastToDtype(a, ACL_DOUBLE) : a;

    auto sign = NPUArray(shape, ACL_DOUBLE);
    auto logdet = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnSlogdetGetWorkspaceSize(aDouble.tensorPtr, sign.tensorPtr, logdet.tensorPtr,
        &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    AclWorkspace workspace(workspaceSize);
    error = aclnnSlogdet(workspace.get(), workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnSlogdet error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    auto absDet = ExecuteUnaryOp(
        logdet,
        NPUArray::GetPyDtype(ACL_DOUBLE),
        [](aclTensor* in, aclTensor* out, uint64_t* ws, aclOpExecutor** exec) {
            return aclnnExpGetWorkspaceSize(in, out, ws, exec);
        },
        [](void* ws, uint64_t wsSize, aclOpExecutor* exec, void* stream) {
            return aclnnExp(ws, wsSize, exec, nullptr);
        },
        "Linalg_Det_Exp"
    );

    auto detDouble = ExecuteBinaryOp(
        sign, absDet, NPUArray::GetPyDtype(ACL_DOUBLE),
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* ws, aclOpExecutor** exec) {
            return aclnnMulGetWorkspaceSize(in1, in2, out, ws, exec);
        },
        [](void* ws, uint64_t wsSize, aclOpExecutor* exec, void* stream) {
            return aclnnMul(ws, wsSize, exec, nullptr);
        },
        "Linalg_Det_Mul"
    );

    // Cast result back to input dtype
    return (a.aclDtype != ACL_DOUBLE) ? CastToDtype(detDouble, a.aclDtype) : detDouble;
}

std::pair<NPUArray, NPUArray> Linalg_Slogdet(const NPUArray& a) {
    auto shape = a.shape;
    shape.erase(shape.end() - 2, shape.end());

    // Cast input to double for numerical stability in LU decomposition
    NPUArray aDouble = (a.aclDtype != ACL_DOUBLE) ? CastToDtype(a, ACL_DOUBLE) : a;

    auto signout = NPUArray(shape, ACL_DOUBLE);
    auto logout = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnSlogdetGetWorkspaceSize(aDouble.tensorPtr, signout.tensorPtr, logout.tensorPtr,
        &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    AclWorkspace workspace(workspaceSize);
    error = aclnnSlogdet(workspace.get(), workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnSlogdet error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    // Cast results back to input dtype
    if (a.aclDtype != ACL_DOUBLE) {
        return {CastToDtype(signout, a.aclDtype), CastToDtype(logout, a.aclDtype)};
    }
    return {signout, logout};
}
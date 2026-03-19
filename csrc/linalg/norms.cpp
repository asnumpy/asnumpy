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
#include <aclnnop/aclnn_logdet.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_slogdet.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

using namespace asnumpy;

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
    auto ord_scalar = aclCreateScalar(&ord, ACL_FLOAT);
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
    auto temp = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnLogdetGetWorkspaceSize(a.tensorPtr, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnLogdet(workspace1.get(), workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnLogdet error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    
    py::dtype dtype = NPUArray::GetPyDtype(ACL_DOUBLE);
    return ExecuteUnaryOp(
        temp,
        dtype, 
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnExpGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnExp(workspace, workspaceSize, executor, nullptr);
        },
        "Linalg_Det"
    );
}

std::pair<NPUArray, NPUArray> Linalg_Slogdet(const NPUArray& a) {
    auto shape = a.shape;
    shape.erase(shape.end() - 2, shape.end());
    auto signout = NPUArray(shape, ACL_DOUBLE);
    auto logout = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnSlogdetGetWorkspaceSize(a.tensorPtr, signout.tensorPtr, logout.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    AclWorkspace workspace(workspaceSize);
    error = aclnnSlogdet(workspace.get(), workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnSlogdet error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return {signout, logout};
}
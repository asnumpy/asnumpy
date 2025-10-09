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

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_norm.h>
#include <aclnnop/aclnn_logdet.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_slogdet.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

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
    auto error = aclnnNormGetWorkspaceSize(a.tensorPtr, ord_scalar, axis_array, keepdims, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnNorm(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnNorm error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Linalg_Det(const NPUArray& a) {
    auto shape = {a.shape[0]};
    auto temp = NPUArray(shape, ACL_DOUBLE);
    auto result = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnLogdetGetWorkspaceSize(a.tensorPtr, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }
    error1 = aclnnLogdet(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnLogdet error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnExpGetWorkspaceSize(temp.tensorPtr, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }
    error2 = aclnnExp(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnExp error");
    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    return result;
}

std::vector<NPUArray> Linalg_Slogdet(const NPUArray& a) {
    std::vector<int64_t> shape = {a.shape[0]};
    auto signout = NPUArray(shape, ACL_DOUBLE);
    auto logout = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnSlogdetGetWorkspaceSize(a.tensorPtr, signout.tensorPtr, logout.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnSlogdet(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnSlogdet error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    std::vector<NPUArray> result;
    result.push_back(signout);
    result.push_back(logout);
    return result;
}
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


#include <asnumpy/linalg/solving_inverting.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_inverse.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

NPUArray Linalg_Inv(const NPUArray& a) {
    auto shape = a.shape;
    auto result = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInverseGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInverse(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnInverse error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}
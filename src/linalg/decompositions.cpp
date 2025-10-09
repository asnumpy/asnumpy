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


#include <asnumpy/linalg/decompositions.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_linalg_qr.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

std::vector<NPUArray> Linalg_Qr(const NPUArray& a, const std::string& mode) {
    int size = a.shape.size();
    int k = a.shape[size-2] < a.shape.back() ? a.shape[size-2] : a.shape.back();
    auto shapeQ = a.shape;
    auto shapeR = a.shape;
    int64_t num = 0;
    if (mode == "complete") {
        num = 1;
    }
    else if (mode == "r") {
        shapeQ = {};
        shapeR[size-2] = k;
        num = 2;
    }
    else {
        shapeQ.back() = k;
        shapeR[size-2] = k;
        num = 0;
    }
    auto resultQ = NPUArray(shapeQ, a.aclDtype);
    auto resultR = NPUArray(shapeR, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLinalgQrGetWorkspaceSize(a.tensorPtr, num, resultQ.tensorPtr, resultR.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnLinalgQr(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnLinalgQr error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    std::vector<NPUArray> result;
    result.push_back(resultQ);
    result.push_back(resultR);
    return result;
}
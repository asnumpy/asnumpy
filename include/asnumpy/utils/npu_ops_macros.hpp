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

#pragma once

#include <asnumpy/utils/status_handler.hpp>
#include <asnumpy/utils/acl_resource.hpp>

#define EXECUTE_OP_WORKSPACE(OpName, workspaceSize, executor, AclnnFunc, AclnnApiName)                                \
    do {                                                                                                               \
        asnumpy::AclWorkspace workspace(workspaceSize);                                                               \
        auto error_func = AclnnFunc(workspace.get(), workspace.size(), executor, nullptr);                            \
        ACLNN_CHECK(error_func, AclnnApiName);                                                                        \
        auto error_sync = aclrtSynchronizeDevice();                                                                    \
        ACL_RT_CHECK(error_sync, AclnnApiName ": aclrtSynchronizeDevice");                                            \
    }                                                                                                                  \
    while (0)

#define DEFINE_UNARY_OP(OpName, AclnnGetWorkspaceSizeFunc, AclnnFunc)                                                  \
    NPUArray OpName(const NPUArray& x) {                                                                               \
        LOG_DEBUG("{} start: input_shape={}, tensorSize={}, aclDtype={}",                                              \
                  #AclnnFunc, detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype));                    \
        auto shape = x.shape;                                                                                          \
        auto dtype = x.dtype;                                                                                          \
        auto result = NPUArray(shape, dtype);                                                                          \
        uint64_t workspaceSize = 0;                                                                                    \
        aclOpExecutor* executor;                                                                                       \
        auto error = AclnnGetWorkspaceSizeFunc(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);              \
        ACLNN_CHECK(error, #AclnnFunc "GetWorkspaceSize");                                                            \
        EXECUTE_OP_WORKSPACE(OpName, workspaceSize, executor, AclnnFunc, #AclnnFunc);                                 \
        LOG_INFO("{} completed", #AclnnFunc);                                                                          \
        return result;                                                                                                 \
    }

#define DEFINE_BINARY_OP(OpName, AclnnGetWorkspaceSizeFunc, AclnnFunc)                                                 \
    NPUArray OpName(const NPUArray& x1, const NPUArray& x2) {                                                          \
        LOG_DEBUG("{} start: x1_shape={}, x2_shape={}, aclDtype={}",                                                   \
                  #AclnnFunc, detail::FormatShape(x1.shape), detail::FormatShape(x2.shape), AclDtypeName(x1.aclDtype));  \
        auto shape = GetBroadcastShape(x1, x2);                                                                        \
        auto dtype = x1.dtype;                                                                                         \
        auto result = NPUArray(shape, dtype);                                                                          \
        uint64_t workspaceSize = 0;                                                                                    \
        aclOpExecutor* executor;                                                                                       \
        auto error =                                                                                                   \
            AclnnGetWorkspaceSizeFunc(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, &workspaceSize, &executor);        \
        ACLNN_CHECK(error, #AclnnFunc "GetWorkspaceSize");                                                            \
        EXECUTE_OP_WORKSPACE(OpName, workspaceSize, executor, AclnnFunc, #AclnnFunc);                                 \
        LOG_INFO("{} completed", #AclnnFunc);                                                                          \
        return result;                                                                                                 \
    }

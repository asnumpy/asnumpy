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


#include <asnumpy/math/exponents_and_logarithms.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_expm1.h>
#include <aclnnop/aclnn_exp2.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_log10.h>
#include <aclnnop/aclnn_log2.h>
#include <aclnnop/aclnn_log1p.h>
#include <aclnnop/aclnn_logaddexp.h>
#include <aclnnop/aclnn_logaddexp2.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {
    NPUArray Exp(const NPUArray& x) {
        //因AOL算子限制，不支持64位int之外的int类型
        auto shape = x.shape;
        aclDataType aclType = x.aclDtype;
        if (x.aclDtype == ACL_BOOL || x.aclDtype == ACL_INT64){
            aclType = ACL_DOUBLE;
        }
        auto result = NPUArray(shape, aclType);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnExpGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnExp(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnExp error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Expm1(const NPUArray& x) {
        //因AOL算子限制，不支持64位int之外的int类型
        auto shape = x.shape;
        aclDataType aclType = x.aclDtype;
        if (x.aclDtype == ACL_BOOL || x.aclDtype == ACL_INT64){
            aclType = ACL_DOUBLE;
        }
        auto result = NPUArray(shape, aclType);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnExpm1GetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnExpm1(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnExpm1 error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Exp2(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        auto result = NPUArray(shape, aclType);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnExp2GetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnExp2(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnExp2 error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Log(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        auto result = NPUArray(shape, aclType);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnLogGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnLog(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnLog error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Log10(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        auto result = NPUArray(shape, aclType);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnLog10GetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnLog10(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnLog10 error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Log2(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        auto result = NPUArray(shape, aclType);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnLog2GetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnLog2(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnLog2 error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Log1p(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        auto result = NPUArray(shape, aclType);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnLog1pGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnLog1p(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnLog1p error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Logaddexp(const NPUArray& x1, const NPUArray& x2) {
        auto broadcast = GetBroadcastShape(x1, x2);
        auto result = NPUArray(broadcast, ACL_FLOAT);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnLogAddExpGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnLogAddExp(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnLogAddExp error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Logaddexp2(const NPUArray& x1, const NPUArray& x2) {
        auto broadcast = GetBroadcastShape(x1, x2);
        auto result = NPUArray(broadcast, ACL_FLOAT);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnLogAddExp2GetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnLogAddExp2(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnLogAddExp2 error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }
}
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
#include <asnumpy/utils/acl_executor.hpp>

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

#include <fmt/core.h>
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
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnExpGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnExp(workspace, workspaceSize, executor, nullptr);
            },
            "Exp"                                       
        );
    }

    NPUArray Expm1(const NPUArray& x) {
        //因AOL算子限制，不支持64位int之外的int类型
        auto shape = x.shape;
        aclDataType aclType = x.aclDtype;
        if (x.aclDtype == ACL_BOOL || x.aclDtype == ACL_INT64){
            aclType = ACL_DOUBLE;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnExpm1GetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnExpm1(workspace, workspaceSize, executor, nullptr);
            },
            "Expm1"                                       
        );
    }

    NPUArray Exp2(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnExp2GetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnExp2(workspace, workspaceSize, executor, nullptr);
            },
            "Exp2"                                       
        );
    }

    NPUArray Log(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || 
            x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnLogGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnLog(workspace, workspaceSize, executor, nullptr);
            },
            "Log"                                       
        );
    }

    NPUArray Log10(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || 
            x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnLog10GetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnLog10(workspace, workspaceSize, executor, nullptr);
            },
            "Log10"                                       
        );
    }

    NPUArray Log2(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || 
            x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnLog2GetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnLog2(workspace, workspaceSize, executor, nullptr);
            },
            "Log2"                                       
        );
    }

    NPUArray Log1p(const NPUArray& x) {
        auto shape = x.shape;
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnLog1pGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnLog1p(workspace, workspaceSize, executor, nullptr);
            },
            "Log1p"                                       
        );
    }

    NPUArray Logaddexp(const NPUArray& x1, const NPUArray& x2) {
        py::dtype dtype = NPUArray::GetPyDtype(ACL_FLOAT);
        return ExecuteBinaryOp(
            x1,
            x2,                                           
            dtype,                                     
            [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnLogAddExpGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnLogAddExp(workspace, workspaceSize, executor, nullptr);
            },
            "Logaddexp"                                       
        );
    }

    NPUArray Logaddexp2(const NPUArray& x1, const NPUArray& x2) {
        py::dtype dtype = NPUArray::GetPyDtype(ACL_FLOAT);
        return ExecuteBinaryOp(
            x1,
            x2,                                           
            dtype,                                     
            [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnLogAddExp2GetWorkspaceSize(in1, in2, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnLogAddExp2(workspace, workspaceSize, executor, nullptr);
            },
            "Logaddexp2"                                       
        );
    }
}
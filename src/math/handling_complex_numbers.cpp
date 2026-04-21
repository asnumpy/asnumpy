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


#include <asnumpy/math/handling_complex_numbers.hpp>
#include <asnumpy/utils/acl_executor.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_real.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy{
    NPUArray Real(const NPUArray& val) {
        auto shape = val.shape;
        auto aclType = val.aclDtype;
        if (val.aclDtype == ACL_COMPLEX64 || val.aclDtype == ACL_COMPLEX128){
            aclType = ACL_FLOAT;
        }
        ACL_DTYPE_WARN(val.aclDtype, aclType, __func__);
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return EXECUTE_UNARY_OP(
            val,
            dtype,
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnRealGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnReal(workspace, workspaceSize, executor, nullptr);
            },
            "Real",
            "aclnnReal"
        );
    }
}
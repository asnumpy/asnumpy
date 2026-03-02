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


#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_sinc.h>
#include <asnumpy/math/other_special_functions.hpp>
#include <asnumpy/utils/status_handler.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

/**
 * @brief Element-wise sinc function using aclnnSinc.
 */
NPUArray Sinc(const NPUArray& x, std::optional<py::dtype> dtype) {
    py::dtype out_py_dtype = NPUArray::GetPyDtype(ACL_DOUBLE);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
    }
    return ExecuteUnaryOp(
        x,                                           
        out_py_dtype,                                      
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnSincGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnSinc(workspace, workspaceSize, executor, nullptr);
        },
        "Sinc"
    );
}

}
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

#include <asnumpy/nn/activation.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_softmax.h>

#include <cstdint>
#include <cstdio>
#include <fmt/base.h>
#include <fmt/format.h>
#include <optional>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace asnumpy {
    NPUArray Softmax(const NPUArray& x, int64_t axis, std::optional<py::dtype> dtype) {
        py::dtype outDtype = dtype.has_value() ? dtype.value() : x.dtype;
        auto shape = x.shape;
        
        // Normalize axis
        int64_t ax = axis;
        if (axis < 0) {
            ax = shape.size() + axis;
        }
        
        // Output has the same shape as input
        auto result = NPUArray(shape, outDtype);
        
        // Call CANN softmax operator
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnSoftmaxGetWorkspaceSize(x.tensorPtr, ax, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        
        error = aclnnSoftmax(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnSoftmax error");
        
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        
        if(workspaceSize != 0ULL) {
            aclrtFree(workspaceAddr);
        }
        
        return result;
    }
}


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
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_softmax.h>

#include <cstdint>
#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>
#include <optional>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace asnumpy {
    NPUArray Softmax(const NPUArray& x, int64_t axis, std::optional<py::dtype> dtype) {
        LOG_DEBUG("aclnnSoftmax start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype), axis);
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
        ACLNN_CHECK(error, "aclnnSoftmaxGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);

        error = aclnnSoftmax(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnSoftmax");

        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
        LOG_INFO("aclnnSoftmax completed");
        return result;
    }
}


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


#include <asnumpy/sorting/sorting.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_sort.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

NPUArray Sort(const NPUArray& a, int axis, bool stable) {
    LOG_DEBUG("aclnnSort start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis);
    auto shape = a.shape;
    auto result = NPUArray(shape, a.aclDtype);
    auto indices = NPUArray(shape, ACL_INT64);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnSortGetWorkspaceSize(a.tensorPtr, stable, axis, false,
        result.tensorPtr, indices.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnSortGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnSort(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnSort");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnSort completed");
    return result;
}

}
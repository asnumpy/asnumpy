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
 ******************************************************************************/

#include <asnumpy/utils/cast.hpp>

#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_cast.h>

namespace asnumpy {

NPUArray CastTo(const NPUArray& input, aclDataType targetDtype) {
    if (input.aclDtype == targetDtype)
        return NPUArray(input); // deep copy; keeps callers free to cast unconditionally

    LOG_DEBUG("aclnnCast start: tensorSize={}, from={}, to={}", input.tensorSize, AclDtypeName(input.aclDtype),
              AclDtypeName(targetDtype));

    auto result = NPUArray(input.shape, targetDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnCastGetWorkspaceSize(input.tensorPtr, targetDtype, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnCastGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);
    error = aclnnCast(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnCast");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclnnCast: aclrtSynchronizeDevice");

    LOG_INFO("aclnnCast completed");
    return result;
}

} // namespace asnumpy

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

#include <fmt/format.h>
#include <stdexcept>
#include <string>
#include "acl/acl.h"
#include "aclnn/acl_meta.h" // aclnnStatus definition

/*
 * How to use:
 * 1   auto error = AclnnGetWorkspaceSizeFunc(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
 * 2   CheckAclnnStatus(error, "Failed to allocate workspace.");
 * or:
 * 1   auto error = AclnnGetWorkspaceSizeFunc(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
 * 2   CheckGetWorkspaceSizeAclnnStatus(error);
 *
 * output:
 *       RuntimeError: Failed to allocate workspace. error code 161001. Details: xxx
 *
 *       (if GetRecentErrMsg is not nullptr, show Details: xxx)
 */

/**
 * @brief Check aclnnStatus, and throw an exception with detailed error information if it fails
 *
 * @param status ACLNN status code
 * @param context Optional context information, used to describe the operation content
 * @throw std::runtime_error Thrown when status is not ACL_SUCCESS
 */
void CheckAclnnStatus(aclnnStatus status, const std::string& context);


// Directly call the corresponding error function below for commonly used APIs in the operator
// to avoid writing error operation information yourself

// aclnnXXXGetWorkspaceSize
void CheckGetWorkspaceSizeAclnnStatus(aclnnStatus status);
// aclrtMalloc
void CheckMallocAclnnStatus(aclnnStatus status);
// aclrtSynchronizeDevice
void CheckSynchronizeDeviceAclnnStatus(aclnnStatus status);

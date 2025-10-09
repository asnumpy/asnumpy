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


#include <asnumpy/utils/status_handler.hpp>




void CheckAclnnStatus(aclnnStatus status, const std::string& context)
{
    if (status != ACL_SUCCESS) {

        const char* error_msg = aclGetRecentErrMsg();
        std::string full_error_msg;
        
        if (!context.empty()){
            full_error_msg = fmt::format("{} error code: {}. ",context,status);
        }
        else{
            full_error_msg = fmt::format("error code: {}. ",status);
        }

        if (error_msg != nullptr) {
            full_error_msg += fmt::format("Details: {}", error_msg);
        }
        
        throw std::runtime_error(full_error_msg);
    }
}


void CheckGetWorkspaceSizeAclnnStatus(aclnnStatus status){
    CheckAclnnStatus(status, "Failed to allocate workspace.");
}

void CheckMallocAclnnStatus(aclnnStatus status){
    CheckAclnnStatus(status, "Failed to allocate workspace memory.");
}


void CheckSynchronizeDeviceAclnnStatus(aclnnStatus status){
    CheckAclnnStatus(status, "Failed to synchronize device.");
}
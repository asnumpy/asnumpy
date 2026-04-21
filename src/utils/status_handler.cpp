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
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <stdexcept>
#include <cstring>

namespace asnumpy {

namespace {

// Extract basename from full path: "/path/to/file.cpp" -> "file.cpp"
const char* basename(const char* path) {
    if (!path) return "unknown";
    const char* last_slash = std::strrchr(path, '/');
    return last_slash ? last_slash + 1 : path;
}

// Safely get ACL error detail message
std::string get_error_detail() {
    const char* msg = aclGetRecentErrMsg();
    if (msg && std::strlen(msg) > 0) {
        return std::string(" - ") + msg;
    }
    return "";
}

} // anonymous namespace

void CheckAclnnStatus(aclnnStatus status, const char* file, const char* func, const char* api_name) {
    CheckAclnnStatus(status, file, func, std::string(api_name));
}

void CheckAclRuntimeStatus(aclError status, const char* file, const char* func, const char* api_name) {
    CheckAclRuntimeStatus(status, file, func, std::string(api_name));
}

void CheckAclnnStatus(aclnnStatus status, const char* file, const char* func, const std::string& api_name) {
    if (status != ACL_SUCCESS) {
        auto detail = get_error_detail();
        auto error_msg = fmt::format("[{}]({}) {} error = {}{}",
                                     basename(file), func, api_name, status, detail);
        spdlog::error(error_msg);
        throw std::runtime_error(error_msg);
    }
}

void CheckAclRuntimeStatus(aclError status, const char* file, const char* func, const std::string& api_name) {
    if (status != ACL_SUCCESS) {
        auto detail = get_error_detail();
        auto error_msg = fmt::format("[{}]({}) {} error = {}{}",
                                     basename(file), func, api_name, status, detail);
        spdlog::error(error_msg);
        throw std::runtime_error(error_msg);
    }
}

} // namespace asnumpy

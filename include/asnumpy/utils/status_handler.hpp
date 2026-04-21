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

#pragma once

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstring>
#include <string>

// Forward declaration for spdlog (header-only inline usage)
#include <spdlog/spdlog.h>

namespace asnumpy {

namespace detail {

// Extract basename from full path: "/path/to/file.cpp" -> "file.cpp"
inline const char* LogBasename(const char* path) {
    if (!path) return "unknown";
    const char* last_slash = std::strrchr(path, '/');
    return last_slash ? last_slash + 1 : path;
}

} // namespace detail

/**
 * @brief Check aclnnStatus with full source location context
 *
 * Message format: [filename](function) api_name error = <code> - <detail>
 *
 * @param status ACLNN status code
 * @param file Source file path (use __FILE__)
 * @param func Function name (use __func__)
 * @param api_name ACL API name (e.g., "aclnnSinGetWorkspaceSize")
 * @throw std::runtime_error Thrown when status is not ACL_SUCCESS
 */
void CheckAclnnStatus(aclnnStatus status, const char* file, const char* func, const char* api_name);

/**
 * @brief Check aclnnStatus with full source location context (string api_name overload)
 */
void CheckAclnnStatus(aclnnStatus status, const char* file, const char* func, const std::string& api_name);

/**
 * @brief Check aclError (runtime API) with full source location context
 *
 * Message format: [filename](function) api_name error = <code> - <detail>
 *
 * @param status ACL runtime error code
 * @param file Source file path (use __FILE__)
 * @param func Function name (use __func__)
 * @param api_name ACL API name (e.g., "aclrtMalloc")
 * @throw std::runtime_error Thrown when status is not ACL_SUCCESS
 */
void CheckAclRuntimeStatus(aclError status, const char* file, const char* func, const char* api_name);

/**
 * @brief Check aclError with full source location context (string api_name overload)
 */
void CheckAclRuntimeStatus(aclError status, const char* file, const char* func, const std::string& api_name);

/**
 * @brief Convert aclDataType enum to human-readable string
 *
 * Maps CANN aclDataType enum values to NumPy-compatible dtype names
 * for use in log messages and warnings.
 *
 * @param dtype ACL data type enum value
 * @return Human-readable type name string (e.g., "float32", "int64")
 */
inline const char* AclDtypeName(aclDataType dtype) {
    switch (dtype) {
        case ACL_FLOAT:   return "float32";
        case ACL_FLOAT16: return "float16";
        case ACL_DOUBLE:  return "float64";
        case ACL_INT8:    return "int8";
        case ACL_INT16:   return "int16";
        case ACL_INT32:   return "int32";
        case ACL_INT64:   return "int64";
        case ACL_UINT8:   return "uint8";
        case ACL_UINT16:  return "uint16";
        case ACL_UINT32:  return "uint32";
        case ACL_UINT64:  return "uint64";
        case ACL_BOOL:    return "bool";
        case ACL_COMPLEX64:  return "complex64";
        case ACL_COMPLEX128: return "complex128";
        case ACL_BF16:    return "bfloat16";
        default:          return "unknown";
    }
}

/**
 * @brief Log a warning when input dtype is silently overridden
 *
 * Call this at the point where dtype auto-conversion happens:
 *   ACL_DTYPE_WARN(x.aclDtype, aclType, __func__);
 *
 * @param actual_dtype The original input dtype
 * @param resolved_dtype The dtype actually used (may differ)
 * @param func Caller function name (typically __func__)
 */
#define ACL_DTYPE_WARN(actual_dtype, resolved_dtype, func) \
    do { \
        if ((actual_dtype) != (resolved_dtype)) { \
            spdlog::warn("[{}]({}) dtype auto-conversion: {} -> {}", \
                          ::asnumpy::detail::LogBasename(__FILE__), (func), \
                          ::asnumpy::AclDtypeName(actual_dtype), ::asnumpy::AclDtypeName(resolved_dtype)); \
        } \
    } while (0)

} // namespace asnumpy

// Convenience macros - automatically capture source location
#define ACLNN_CHECK(status, api_name) \
    ::asnumpy::CheckAclnnStatus(status, __FILE__, __func__, api_name)

#define ACL_RT_CHECK(status, api_name) \
    ::asnumpy::CheckAclRuntimeStatus(status, __FILE__, __func__, api_name)

// Logging macros with automatic [filename](function) prefix
// Usage: LOG_DEBUG("{} start: shape={}", op_name, shape);
// Output: [acl_executor.hpp](Sin) Sin start: shape=3x3
#define LOG_DEBUG(fmt_str, ...) \
    spdlog::debug("[{}]({}) " fmt_str, \
                   ::asnumpy::detail::LogBasename(__FILE__), __func__ __VA_OPT__(,) __VA_ARGS__)

#define LOG_INFO(fmt_str, ...) \
    spdlog::info("[{}]({}) " fmt_str, \
                  ::asnumpy::detail::LogBasename(__FILE__), __func__ __VA_OPT__(,) __VA_ARGS__)

#define LOG_WARN(fmt_str, ...) \
    spdlog::warn("[{}]({}) " fmt_str, \
                  ::asnumpy::detail::LogBasename(__FILE__), __func__ __VA_OPT__(,) __VA_ARGS__)

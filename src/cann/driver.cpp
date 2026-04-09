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


#include "asnumpy/cann/driver.hpp"
#include "fmt/core.h"
#include <stdexcept>

namespace {
aclrtStream g_stream = nullptr;
}

void asnumpy::cann::init() {
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        auto message = aclGetRecentErrMsg();
        throw std::runtime_error(fmt::format("aclInit failed ({}): {}",
            ret, message ? message : "unknown error"));
    }

    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        auto message = aclGetRecentErrMsg();
        throw std::runtime_error(fmt::format("aclrtSetDevice(0) failed ({}): {}",
            ret, message ? message : "unknown error"));
    }

    if (g_stream == nullptr) {
        ret = aclrtCreateStream(&g_stream);
        if (ret != ACL_SUCCESS || g_stream == nullptr) {
            auto message = aclGetRecentErrMsg();
            throw std::runtime_error(fmt::format("aclrtCreateStream failed ({}): {}",
                ret, message ? message : "unknown error"));
        }
    }
}

aclrtStream asnumpy::cann::get_stream() {
    return g_stream;
}

void asnumpy::cann::finalize() {
    if (g_stream != nullptr) {
        aclrtDestroyStream(g_stream);
        g_stream = nullptr;
    }
    auto ret = aclFinalize();
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_FINALIZE) {
        auto message = aclGetRecentErrMsg();
        throw std::runtime_error(fmt::format("aclFinalize failed ({}): {}",
            ret, message ? message : "unknown error"));
    }
}

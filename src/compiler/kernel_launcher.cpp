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

#include "asnumpy/compiler/kernel_launcher.hpp"

#include <acl/acl.h>
#include <fmt/core.h>
#include <stdexcept>

namespace asnumpy {
namespace compiler {

// ---- Internal helpers --------------------------------------------------------

namespace {

void check_acl(aclError ret, const std::string& context) {
    if (ret != ACL_SUCCESS) {
        const char* detail = aclGetRecentErrMsg();
        throw std::runtime_error(fmt::format(
            "{}: error code={}. {}",
            context, static_cast<int>(ret),
            (detail != nullptr ? detail : "(no detail)")
        ));
    }
}

}  // anonymous namespace

// ---- Binary lifecycle -------------------------------------------------------

BinHandle load_binary(const std::string& file_path) {
    aclrtBinHandle handle = nullptr;
    check_acl(
        aclrtBinaryLoadFromFile(file_path.c_str(), nullptr, &handle),
        "aclrtBinaryLoadFromFile failed"
    );
    return reinterpret_cast<BinHandle>(handle);
}

void unload_binary(BinHandle bin_handle) {
    aclError ret = aclrtBinaryUnLoad(reinterpret_cast<aclrtBinHandle>(bin_handle));
    if (ret != ACL_SUCCESS) {
        // Log warning but don't throw -- may be called during atexit
        // when CANN has already been finalized.
        const char* detail = aclGetRecentErrMsg();
        fmt::print(stderr, "[asnumpy] aclrtBinaryUnLoad warning: code={}. {}\n",
                   static_cast<int>(ret),
                   (detail != nullptr ? detail : "(no detail)"));
    }
}

// ---- Function lookup ---------------------------------------------------------

FuncHandle get_function(BinHandle bin_handle, const std::string& kernel_name) {
    aclrtFuncHandle func = nullptr;
    check_acl(
        aclrtBinaryGetFunction(
            reinterpret_cast<aclrtBinHandle>(bin_handle),
            kernel_name.c_str(),
            &func
        ),
        fmt::format("aclrtBinaryGetFunction failed for '{}'", kernel_name)
    );
    return reinterpret_cast<FuncHandle>(func);
}

// ---- Kernel launch -----------------------------------------------------------

void launch_kernel(FuncHandle func_handle,
                   std::uint32_t block_dim,
                   std::vector<py::bytes> packed_args,
                   StreamPtr stream)
{
    auto* hfunc   = reinterpret_cast<aclrtFuncHandle>(func_handle);
    auto* hstream = reinterpret_cast<aclrtStream>(stream);

    // 1. Init kernel args handle
    aclrtArgsHandle args_handle = nullptr;
    check_acl(aclrtKernelArgsInit(hfunc, &args_handle),
              "aclrtKernelArgsInit failed");

    // 2. Append each argument
    for (auto& arg_bytes : packed_args) {
        char* data = PyBytes_AS_STRING(arg_bytes.ptr());
        Py_ssize_t size = PyBytes_GET_SIZE(arg_bytes.ptr());
        aclrtParamHandle param_handle = nullptr;
        check_acl(
            aclrtKernelArgsAppend(args_handle, data,
                                  static_cast<size_t>(size), &param_handle),
            "aclrtKernelArgsAppend failed"
        );
    }

    // 3. Launch the kernel
    // aclrtLaunchKernel (simpler API) takes l2ctrl params, not args handle.
    // Use aclrtLaunchKernelWithConfig with default config instead.
    check_acl(
        aclrtLaunchKernelWithConfig(hfunc, block_dim, hstream,
                                    nullptr,  // default config
                                    args_handle,
                                    nullptr), // reserve
        "aclrtLaunchKernelWithConfig failed"
    );

    // 4. Release args handle resources
    aclrtKernelArgsFinalize(args_handle);
}

// ---- Timing ------------------------------------------------------------------

EventPtr create_event() {
    aclrtEvent event = nullptr;
    check_acl(aclrtCreateEvent(&event), "aclrtCreateEvent failed");
    return reinterpret_cast<EventPtr>(event);
}

void destroy_event(EventPtr event) {
    aclrtDestroyEvent(reinterpret_cast<aclrtEvent>(event));
}

void record_event(EventPtr event, StreamPtr stream) {
    check_acl(
        aclrtRecordEvent(reinterpret_cast<aclrtEvent>(event),
                         reinterpret_cast<aclrtStream>(stream)),
        "aclrtRecordEvent failed"
    );
}

void synchronize_event(EventPtr event) {
    check_acl(
        aclrtSynchronizeEvent(reinterpret_cast<aclrtEvent>(event)),
        "aclrtSynchronizeEvent failed"
    );
}

float elapsed_time_between(EventPtr start, EventPtr end) {
    float ms = 0.0f;
    check_acl(
        aclrtEventElapsedTime(&ms,
                              reinterpret_cast<aclrtEvent>(start),
                              reinterpret_cast<aclrtEvent>(end)),
        "aclrtEventElapsedTime failed"
    );
    return ms;
}

// ---- Stream ------------------------------------------------------------------

StreamPtr create_stream() {
    aclrtStream stream = nullptr;
    check_acl(aclrtCreateStream(&stream), "aclrtCreateStream failed");
    return reinterpret_cast<StreamPtr>(stream);
}

void destroy_stream(StreamPtr stream) {
    aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream));
}

void synchronize_stream(StreamPtr stream) {
    check_acl(
        aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)),
        "aclrtSynchronizeStream failed"
    );
}

// ---- Error -------------------------------------------------------------------

std::string get_last_error() {
    const char* msg = aclGetRecentErrMsg();
    return (msg != nullptr) ? std::string(msg) : "";
}

}  // namespace compiler
}  // namespace asnumpy

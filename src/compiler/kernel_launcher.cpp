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
#include <runtime/runtime/kernel.h>
#include <fmt/core.h>
#include <stdexcept>
#include <cstring>
#include <unordered_map>
#include <vector>

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

void check_rt(rtError_t ret, const std::string& context) {
    if (ret != RT_ERROR_NONE) {
        throw std::runtime_error(fmt::format(
            "{}: rtError={}", context, static_cast<int>(ret)
        ));
    }
}

}  // anonymous namespace

// ---- Binary lifecycle (ACL path, for backward compatibility) ---------------

BinHandle load_binary(const std::string& file_path) {
    aclrtBinHandle handle = nullptr;

    // Pass nullptr for default loading options.  CANN 9.x handles bisheng-
    // compiled ELF .o files (``-shared -fPIC``) with the automatic detection
    // of the ``.aicore_binary`` section.
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

// ---- Binary lifecycle (RTS path) --------------------------------------------

// Magic value matching bisheng --cce-soc-core-type=VecCore output.
// The .aicore_binary section contains an ELF for AI Vector Core.
static constexpr uint32_t kAivecElfMagic = 0x41415246U;  // RT_DEV_BINARY_MAGIC_ELF_AIVEC

namespace {

// Some CANN runtime versions reference the binary data lazily, so we must
// keep the buffer alive after rtDevBinaryRegister returns.  The vector is
// indexed by BinHandle.
std::unordered_map<BinHandle, std::vector<char>> g_binary_data;

// rtFunctionRegister may store the stubName pointer directly (rather than
// copying the string).  Keep kernel name strings alive indexed by handle.
std::unordered_map<BinHandle, std::vector<std::string>> g_kernel_names;

}  // anonymous namespace

BinHandle register_binary(const py::bytes& data) {
    // Copy the Python bytes into a persistent heap buffer.  The runtime
    // may access this data lazily (after register_binary returns).
    Py_ssize_t len = PyBytes_GET_SIZE(data.ptr());
    std::vector<char> persistent_buf(PyBytes_AS_STRING(data.ptr()),
                                     PyBytes_AS_STRING(data.ptr()) + len);

    rtDevBinary_t dev_bin;
    dev_bin.magic   = kAivecElfMagic;
    dev_bin.version = 0;
    dev_bin.data    = static_cast<const void*>(persistent_buf.data());
    dev_bin.length  = static_cast<uint64_t>(len);

    void* handle = nullptr;
    check_rt(
        rtDevBinaryRegister(&dev_bin, &handle),
        "rtDevBinaryRegister failed"
    );
    BinHandle bh = reinterpret_cast<BinHandle>(handle);
    g_binary_data[bh] = std::move(persistent_buf);
    return bh;
}

void register_function(BinHandle bin_handle, const std::string& kernel_name) {
    // In the RTS API, stubFunc / stubName / kernelInfoExt are all the kernel
    // name string.  The runtime uses the name as a lookup key.
    //
    // IMPORTANT: some CANN versions store these pointers directly rather
    // than copying the strings.  We persist a copy in g_kernel_names so the
    // pointer remains valid for the lifetime of the registration.
    auto& names = g_kernel_names[bin_handle];
    names.push_back(kernel_name);
    const char* name = names.back().c_str();

    check_rt(
        rtFunctionRegister(
            reinterpret_cast<void*>(bin_handle),
            static_cast<const void*>(name),   // stubFunc
            name,                              // stubName
            static_cast<const void*>(name),   // kernelInfoExt
            FUNC_MODE_NORMAL                   // 0
        ),
        fmt::format("rtFunctionRegister failed for '{}'", kernel_name)
    );
}

void unregister_binary(BinHandle bin_handle) {
    rtError_t ret = rtDevBinaryUnRegister(reinterpret_cast<void*>(bin_handle));
    if (ret != RT_ERROR_NONE) {
        fmt::print(stderr, "[asnumpy] rtDevBinaryUnRegister warning: rtError={}\n",
                   static_cast<int>(ret));
    }
    // Release the persistent binary data and kernel name buffers
    g_binary_data.erase(bin_handle);
    g_kernel_names.erase(bin_handle);
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

// ---- Kernel launch (ACL path) ------------------------------------------------

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

    // 3. Finalize args handle (required before launch in CANN 9.x)
    check_acl(
        aclrtKernelArgsFinalize(args_handle),
        "aclrtKernelArgsFinalize failed"
    );

    // 4. Launch the kernel
    check_acl(
        aclrtLaunchKernelWithConfig(hfunc, block_dim, hstream,
                                    nullptr,  // default config
                                    args_handle,
                                    nullptr), // reserve
        "aclrtLaunchKernelWithConfig failed"
    );

    // 5. Synchronize device to surface kernel errors
    check_acl(
        aclrtSynchronizeDevice(),
        "aclrtSynchronizeDevice failed"
    );
}

// ---- Kernel launch (RTS path) ------------------------------------------------

void launch_kernel_rts(const std::string& kernel_name,
                       std::uint32_t block_dim,
                       std::vector<py::bytes> packed_args,
                       StreamPtr stream)
{
    auto* hstream = reinterpret_cast<rtStream_t>(stream);

    // Flatten packed_args into a single contiguous buffer
    size_t total_size = 0;
    for (auto& arg : packed_args) {
        total_size += static_cast<size_t>(PyBytes_GET_SIZE(arg.ptr()));
    }

    // Use a stable buffer (not a vector that may reallocate)
    auto flat_buf = std::make_unique<char[]>(total_size);
    size_t offset = 0;
    for (auto& arg : packed_args) {
        size_t sz = static_cast<size_t>(PyBytes_GET_SIZE(arg.ptr()));
        std::memcpy(flat_buf.get() + offset, PyBytes_AS_STRING(arg.ptr()), sz);
        offset += sz;
    }

    check_rt(
        rtKernelLaunch(
            static_cast<const void*>(kernel_name.c_str()),
            block_dim,
            flat_buf.get(),
            static_cast<uint32_t>(total_size),
            nullptr,   // smDesc (default)
            hstream
        ),
        fmt::format("rtKernelLaunch failed for '{}'", kernel_name)
    );

    // Synchronize stream to surface kernel errors
    check_acl(
        aclrtSynchronizeStream(hstream),
        fmt::format("aclrtSynchronizeStream failed after '{}'", kernel_name)
    );
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

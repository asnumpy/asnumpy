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

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace asnumpy {
namespace compiler {

using BinHandle  = std::uintptr_t;
using FuncHandle = std::uintptr_t;
using StreamPtr  = std::uintptr_t;
using EventPtr   = std::uintptr_t;

// ---- Binary lifecycle ----
BinHandle  load_binary(const std::string& file_path);
void       unload_binary(BinHandle bin_handle);

// ---- Function lookup ----
FuncHandle get_function(BinHandle bin_handle, const std::string& kernel_name);

// ---- Kernel launch ----
// Each element of `packed_args` is a py::bytes holding the raw argument data.
// For device pointers (__gm__ float* etc), this is a little-endian uint64.
// For scalars, this is the raw value bytes (e.g. 4 bytes for int32).
void launch_kernel(FuncHandle func_handle,
                   std::uint32_t block_dim,
                   std::vector<py::bytes> packed_args,
                   StreamPtr stream = 0);

// ---- Timing helpers ----
EventPtr  create_event();
void      destroy_event(EventPtr event);
void      record_event(EventPtr event, StreamPtr stream);
void      synchronize_event(EventPtr event);
float     elapsed_time_between(EventPtr start, EventPtr end);

// ---- Stream helpers ----
StreamPtr create_stream();
void      destroy_stream(StreamPtr stream);
void      synchronize_stream(StreamPtr stream);

// ---- Error ----
std::string get_last_error();

}  // namespace compiler
}  // namespace asnumpy

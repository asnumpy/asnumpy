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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <asnumpy/compiler/kernel_launcher.hpp>

namespace py = pybind11;

void bind_compiler(py::module_& compiler) {
    compiler.doc() = "compiler module - JIT Ascend C kernel compilation and launching";

    // ---- Binary lifecycle ----
    compiler.def("load_binary", &asnumpy::compiler::load_binary,
                 py::arg("file_path"),
                 "Load a compiled kernel binary (.o file) and return a bin handle.");

    compiler.def("unload_binary", &asnumpy::compiler::unload_binary,
                 py::arg("bin_handle"),
                 "Unload a kernel binary.");

    // ---- Binary lifecycle (RTS path for CANN 9.x) ----
    compiler.def("register_binary", &asnumpy::compiler::register_binary,
                 py::arg("data"),
                 "Register a raw kernel binary (inner ELF from .aicore_binary section) "
                 "via rtDevBinaryRegister. Returns a bin handle.");

    compiler.def("register_function", &asnumpy::compiler::register_function,
                 py::arg("bin_handle"), py::arg("kernel_name"),
                 "Register a kernel function by name via rtFunctionRegister.");

    compiler.def("unregister_binary", &asnumpy::compiler::unregister_binary,
                 py::arg("bin_handle"),
                 "Unregister a kernel binary via rtDevBinaryUnRegister.");

    // ---- Function lookup ----
    compiler.def("get_function", &asnumpy::compiler::get_function,
                 py::arg("bin_handle"), py::arg("kernel_name"),
                 "Get a function handle for a kernel by name.");

    // ---- Kernel launch ----
    compiler.def("launch_kernel", &asnumpy::compiler::launch_kernel,
                 py::arg("func_handle"), py::arg("block_dim"),
                 py::arg("packed_args"), py::arg("stream") = 0,
                 "Launch a kernel on the NPU (ACL path).");

    compiler.def("launch_kernel_rts", &asnumpy::compiler::launch_kernel_rts,
                 py::arg("kernel_name"), py::arg("block_dim"),
                 py::arg("packed_args"), py::arg("stream") = 0,
                 "Launch a kernel on the NPU (RTS path via rtKernelLaunch).");

    // ---- Timing helpers ----
    compiler.def("create_event", &asnumpy::compiler::create_event,
                 "Create an ACL event for timing.");
    compiler.def("destroy_event", &asnumpy::compiler::destroy_event,
                 py::arg("event"));
    compiler.def("record_event", &asnumpy::compiler::record_event,
                 py::arg("event"), py::arg("stream"));
    compiler.def("synchronize_event", &asnumpy::compiler::synchronize_event,
                 py::arg("event"));
    compiler.def("elapsed_time_between", &asnumpy::compiler::elapsed_time_between,
                 py::arg("start"), py::arg("end"),
                 "Return elapsed time in milliseconds between two events.");

    // ---- Stream helpers ----
    compiler.def("create_stream", &asnumpy::compiler::create_stream,
                 "Create an ACL stream.");
    compiler.def("destroy_stream", &asnumpy::compiler::destroy_stream,
                 py::arg("stream"));
    compiler.def("synchronize_stream", &asnumpy::compiler::synchronize_stream,
                 py::arg("stream"));

    // ---- Error ----
    compiler.def("get_last_error", &asnumpy::compiler::get_last_error,
                 "Get the last ACL error message.");
}

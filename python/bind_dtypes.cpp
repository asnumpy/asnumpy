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
#include <pybind11/numpy.h>
#include <cstdint>
namespace py = pybind11;
void bind_dtypes(py::module_& dtypes){
    dtypes.doc() = "dtypes module of asnumpy";
    dtypes.attr("int32") = py::dtype::of<int32_t>();
    // dtypes.def("float8_e5m2", &float8_e5m2);
    // dtypes.def("float8_e4m3fn", &float8_e4m3fn);
    // dtypes.def("float8_e8m0", &float8_e8m0);
    // dtypes.def("bfloat16", &bfloat16);
    // dtypes.def("float6_e2m3fn", &float6_e2m3fn);
    // dtypes.def("float6_e3m2fn", &float6_e3m2fn);
    // dtypes.def("float4_e2m1fn", &float4_e2m1fn);
    // dtypes.def("float4_e1m2fn", &float4_e1m2fn);
    // dtypes.def("int4", &int4);
    // dtypes.def("uint1", &uint1);
} 
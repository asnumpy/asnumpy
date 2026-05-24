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

#include <asnumpy/dtypes/acl_float_reg.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
// forward declaration to avoid extra header
namespace asnumpy {
namespace dtypes {
void InitAndRegisterDtypes();
}
} // namespace asnumpy

using namespace asnumpy::dtypes;

void bind_dtypes(pybind11::module_& dtypes) {
    dtypes.doc() = "ACL custom dtypes for NumPy";

    // initialize and register all dtypes (idempotent; imports NumPy C API once)
    asnumpy::dtypes::InitAndRegisterDtypes();

    // bind all registered float type objects to the Python module
    if (ACLFloatManager<float8_e5m2>::type_ptr != nullptr) {
        dtypes.attr("float8_e5m2") =
            pybind11::reinterpret_borrow<pybind11::object>(ACLFloatManager<float8_e5m2>::type_ptr);
    }

    if (ACLFloatManager<float8_e4m3fn>::type_ptr != nullptr) {
        dtypes.attr("float8_e4m3fn") =
            pybind11::reinterpret_borrow<pybind11::object>(ACLFloatManager<float8_e4m3fn>::type_ptr);
    }

    if (ACLFloatManager<float8_e8m0>::type_ptr != nullptr) {
        dtypes.attr("float8_e8m0") =
            pybind11::reinterpret_borrow<pybind11::object>(ACLFloatManager<float8_e8m0>::type_ptr);
    }

    if (ACLFloatManager<bfloat16>::type_ptr != nullptr) {
        dtypes.attr("bfloat16") = pybind11::reinterpret_borrow<pybind11::object>(ACLFloatManager<bfloat16>::type_ptr);
    }

    if (ACLFloatManager<float6_e2m3fn>::type_ptr != nullptr) {
        dtypes.attr("float6_e2m3fn") =
            pybind11::reinterpret_borrow<pybind11::object>(ACLFloatManager<float6_e2m3fn>::type_ptr);
    }

    if (ACLFloatManager<float6_e3m2fn>::type_ptr != nullptr) {
        dtypes.attr("float6_e3m2fn") =
            pybind11::reinterpret_borrow<pybind11::object>(ACLFloatManager<float6_e3m2fn>::type_ptr);
    }

    if (ACLFloatManager<float4_e2m1fn>::type_ptr != nullptr) {
        dtypes.attr("float4_e2m1fn") =
            pybind11::reinterpret_borrow<pybind11::object>(ACLFloatManager<float4_e2m1fn>::type_ptr);
    }
}

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

<<<<<<< HEAD:bindings/python/bind_dtypes.cpp
#include <asnumpy/dtypes/acl_float_reg.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
// forward declaration to avoid extra header
namespace asnumpy {
namespace dtypes {
void InitAndRegisterDtypes();
}
} // namespace asnumpy
=======
#include <pybind11/pybind11.h>
>>>>>>> f71a1a9 (feat(dtypes): integrate new dtype module into core code):python/bind_dtypes.cpp

#include <asnumpy/dtypes/acl_float_reg.hpp>
#include <asnumpy/dtypes/acl_int_reg.hpp>

namespace py = pybind11;

// 前向声明，避免额外头文件
namespace asnumpy {
namespace dtypes {
void InitAndRegisterDtypes();
}  // namespace dtypes
}  // namespace asnumpy

namespace {

template <template <typename> class Manager, typename T>
void bind_one_dtype(py::module_& m, const char* attr_name) {
    if (Manager<T>::type_ptr == nullptr) {
        return;
    }
    m.attr(attr_name) = py::reinterpret_borrow<py::object>(Manager<T>::type_ptr);
}

// 单一数据源：此列表应与 src/dtypes/reg.cpp 中 InitAndRegisterDtypes() 注册的类型保持一致。
#define ASNUMPY_DTYPE_BIND_LIST(X)                                                              \
    /* float */                                                                                 \
    X("float8_e5m2", asnumpy::dtypes::ACLFloatManager, asnumpy::dtypes::float8_e5m2)            \
    X("float8_e4m3fn", asnumpy::dtypes::ACLFloatManager, asnumpy::dtypes::float8_e4m3fn)        \
    X("float8_e8m0", asnumpy::dtypes::ACLFloatManager, asnumpy::dtypes::float8_e8m0)            \
    X("bfloat16", asnumpy::dtypes::ACLFloatManager, asnumpy::dtypes::bfloat16)                  \
    X("float6_e2m3fn", asnumpy::dtypes::ACLFloatManager, asnumpy::dtypes::float6_e2m3fn)        \
    X("float6_e3m2fn", asnumpy::dtypes::ACLFloatManager, asnumpy::dtypes::float6_e3m2fn)        \
    X("float4_e2m1fn", asnumpy::dtypes::ACLFloatManager, asnumpy::dtypes::float4_e2m1fn)        \
    X("float4_e1m2fn", asnumpy::dtypes::ACLFloatManager, asnumpy::dtypes::float4_e1m2fn)        \
    /* int */                                                                                   \
    X("int4", asnumpy::dtypes::ACLIntManager, asnumpy::dtypes::int4)                             \
    X("uint1", asnumpy::dtypes::ACLIntManager, asnumpy::dtypes::uint1)

}  // namespace

void bind_dtypes(pybind11::module_& dtypes) {
    dtypes.doc() = "ACL custom dtypes for NumPy";

<<<<<<< HEAD:bindings/python/bind_dtypes.cpp
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
=======
    // 初始化并注册所有 dtypes（幂等且仅导入一次 NumPy C API）
    asnumpy::dtypes::InitAndRegisterDtypes();

    // 将所有已注册 dtype 的 Python 类型对象挂到 asnumpy_core.dtypes 下
#define ASNUMPY_BIND_ONE(attr_name, Manager, Type) bind_one_dtype<Manager, Type>(dtypes, attr_name);
    ASNUMPY_DTYPE_BIND_LIST(ASNUMPY_BIND_ONE)
#undef ASNUMPY_BIND_ONE
}

#undef ASNUMPY_DTYPE_BIND_LIST
>>>>>>> f71a1a9 (feat(dtypes): integrate new dtype module into core code):python/bind_dtypes.cpp

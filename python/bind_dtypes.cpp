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

#include <asnumpy/dtypes/np_import.hpp>
#include <asnumpy/dtypes/reg.hpp>
#include <asnumpy/dtypes/float_types.hpp>

namespace py = pybind11;
void bind_dtypes(py::module_& dtypes){
    dtypes.doc() = "dtypes module of asnumpy";
    
    // 1. 初始化并注册所有 dtype（包括 float8_e5m2 和 bfloat16）
    asnumpy::dtypes::InitAndRegisterDtypes();
    
    // 2. 检查注册是否成功
    bool is_registered = asnumpy::dtypes::AreAllACLFloatTypesRegistered();
    if (!is_registered) {
        throw std::runtime_error("ACL float types registration failed");
    }
    
    // 3. 绑定标准类型
    dtypes.attr("int32") = py::dtype::of<int32_t>();
    
    // 4. 绑定 float8_e5m2
    PyObject* float8_e5m2_type = asnumpy::dtypes::GetACLFloatTypeObject<asnumpy::dtypes::float8_e5m2>();
    if (float8_e5m2_type != nullptr) {
        dtypes.attr("float8_e5m2") = py::reinterpret_borrow<py::object>(float8_e5m2_type);
    } else {
        throw std::runtime_error("float8_e5m2 type object is null");
    }
    
    // 5. 绑定 bfloat16
    PyObject* bfloat16_type = asnumpy::dtypes::GetACLFloatTypeObject<asnumpy::dtypes::bfloat16>();
    if (bfloat16_type != nullptr) {
        dtypes.attr("bfloat16") = py::reinterpret_borrow<py::object>(bfloat16_type);
    } else {
        throw std::runtime_error("bfloat16 type object is null");
    }
    
    // 6. 添加检查注册状态的公共函数（用于测试）
    dtypes.def("check_float8_e5m2_registered", []() {
        return asnumpy::dtypes::AreAllACLFloatTypesRegistered();
    });
    
    dtypes.def("get_float8_e5m2_type_num", []() {
        return asnumpy::dtypes::GetACLFloatTypeNum<asnumpy::dtypes::float8_e5m2>();
    });
    
    dtypes.def("check_bfloat16_registered", []() {
        return asnumpy::dtypes::GetACLFloatTypeNum<asnumpy::dtypes::bfloat16>() != NPY_NOTYPE;
    });
    
    dtypes.def("get_bfloat16_type_num", []() {
        return asnumpy::dtypes::GetACLFloatTypeNum<asnumpy::dtypes::bfloat16>();
    });
    dtypes.def("test_numpy_dtype_str", []() {
        asnumpy::dtypes::ImportNumpy();
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
        if (descr == nullptr) {
            throw py::error_already_set();
        }
        py::object dtype_obj = py::reinterpret_steal<py::object>(reinterpret_cast<PyObject*>(descr));
        return py::str(dtype_obj);
    });
    dtypes.def("test_numpy_create_array", []() {
        asnumpy::dtypes::ImportNumpy();
        npy_intp dims[1] = {3};
        PyObject* arr = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (arr == nullptr) {
            throw py::error_already_set();
        }
        return py::reinterpret_steal<py::array>(arr);
    });
} 
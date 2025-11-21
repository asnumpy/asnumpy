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
namespace py = pybind11;
void bind_dtypes(py::module_& dtypes){
    dtypes.doc() = "dtypes module of asnumpy";
    dtypes.attr("int32") = py::dtype::of<int32_t>();
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
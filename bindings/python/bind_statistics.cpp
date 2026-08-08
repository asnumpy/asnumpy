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

#include <asnumpy/statistics/averages_and_variances.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

namespace py = pybind11;

namespace asnumpy {

void bind_statistics(py::module_& statistics) {
    statistics.doc() = "statistics module of asnumpy";
    statistics.def(
        "mean",
        [](const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims, py::dtype compute_dtype,
           py::dtype result_dtype, const py::object& out) -> py::object {
            if (out.is_none()) {
                return py::cast(
                    Mean(a, axes, keepdims, std::move(compute_dtype), std::move(result_dtype)));
            }

            // Keep and return the exact Python object supplied by the caller.
            // This matters for a public `asnumpy.ndarray`, which is a Python
            // subclass of the bound NPUArray base class.
            NPUArray& out_array = out.cast<NPUArray&>();
            MeanOut(a, axes, keepdims, std::move(compute_dtype), out_array);
            return out;
        },
        py::arg("a"), py::arg("axes"), py::arg("keepdims"), py::arg("compute_dtype"),
        py::arg("result_dtype"), py::arg("out") = py::none(),
        "Compute a mean over normalized axes, optionally into an existing array.");
}

} // namespace asnumpy

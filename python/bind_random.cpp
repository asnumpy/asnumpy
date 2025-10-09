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
#include <asnumpy/random/distributions.hpp>

using namespace asnumpy;

void bind_random(pybind11::module_& random) {
    random.doc() = "random module of asnumpy";

    random.def("pareto", &Generator_Pareto, py::arg("a"), py::arg("size"));
    random.def("rayleigh", &Generator_Rayleigh, py::arg("scale"), py::arg("size"));
    random.def("normal", &Generator_Normal, py::arg("loc"), py::arg("scale"), py::arg("size"));
    random.def("uniform", &Generator_Uniform, py::arg("low"), py::arg("high"), py::arg("size"));
    random.def("standard_normal", &Generator_Standard_normal, py::arg("size"));
    random.def("standard_cauchy", &Generator_Standard_cauchy, py::arg("size"));
    random.def("weibull", &Generator_Weibull, py::arg("a"), py::arg("size"));
    random.def("binomial", &Binomial, py::arg("n"), py::arg("p"), py::arg("size"));
    random.def("exponential", &Exponential, py::arg("scale"), py::arg("size"));
    random.def("geometric", &Geometric, py::arg("p"), py::arg("size"));
    random.def("gumbel", &Gumbel, py::arg("loc"), py::arg("scale"), py::arg("size"));
    random.def("laplace", &Laplace, py::arg("loc"), py::arg("scale"), py::arg("size"));
    random.def("logistic", &Logistic, py::arg("loc"), py::arg("scale"), py::arg("size"));
    random.def("lognormal", &Lognormal, py::arg("mean"), py::arg("sigma"), py::arg("size"));
}
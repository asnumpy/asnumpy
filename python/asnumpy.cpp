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

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

void bind_array(pybind11::module_& array);
void bind_cann(pybind11::module_& cann);
void bind_dtypes(pybind11::module_& dtypes);
void bind_fft(pybind11::module_& fft);
void bind_linalg(pybind11::module_& linalg);
void bind_linalg_no_submodule(pybind11::module_& m);
void bind_math(pybind11::module_& math);
void bind_logic(pybind11::module_& logic);
void bind_random(pybind11::module_& random);
void bind_sorting(pybind11::module_& sorting);
void bind_testing(pybind11::module_& testing);
void bind_utils(pybind11::module_& utils);
void bind_version(pybind11::module_& version);

namespace asnumpy {
void bind_statistics(pybind11::module_& statistics);
void bind_nn(pybind11::module_& nn);
}


PYBIND11_MODULE(asnumpy_core, module) {
    module.doc() = "*** AsNumpy Core ***";
    auto array = module.def_submodule("array");
    auto cann = module.def_submodule("cann");
    auto dtypes = module.def_submodule("dtypes");
    auto fft = module.def_submodule("fft");
    auto linalg = module.def_submodule("linalg");
    auto math = module.def_submodule("math");
    auto logic = module.def_submodule("logic");
    auto random = module.def_submodule("random");
    auto sorting = module.def_submodule("sorting");
    auto statistics = module.def_submodule("statistics");
    auto nn = module.def_submodule("nn");
    auto testing = module.def_submodule("testing");
    // auto utils = module.def_submodule("utils");
    auto version = module.def_submodule("version");
    

    bind_array(array);
    bind_cann(cann);
    // bind_dtypes(dtypes);
    bind_fft(fft);
    bind_linalg(linalg);
    bind_linalg_no_submodule(module);
    bind_math(math);
    bind_logic(logic);
    bind_random(random);
    bind_sorting(sorting);
    asnumpy::bind_statistics(statistics);
    asnumpy::bind_nn(nn);
    bind_testing(testing);
    bind_utils(module);
    bind_version(version);
}
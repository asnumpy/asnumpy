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
 *****************************************************************************/

#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

namespace asnumpy {

//NPUArray Convolve(const NPUArray& a, const NPUArray& v);

NPUArray Clip(const NPUArray& a, const NPUArray& a_min, const NPUArray& a_max);
NPUArray Clip(const NPUArray& a, const NPUArray& a_min, float a_max);
NPUArray Clip(const NPUArray& a, float a_min, float a_max);
NPUArray Clip(const NPUArray& a, float a_min, const NPUArray& a_max);

NPUArray Square(const NPUArray& x);

NPUArray Absolute(const NPUArray& x);

NPUArray Fabs(const NPUArray& x);

NPUArray Nan_to_num(const NPUArray& x, float nan, py::object posinf, py::object neginf);

NPUArray Sign(const NPUArray& x);

NPUArray Heaviside(const NPUArray& x1, const NPUArray& x2);

NPUArray Maximum(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Minimum(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Fmax(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Fmin(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

}
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

NPUArray Generator_Pareto(float a, const std::vector<int64_t>& size);

NPUArray Generator_Rayleigh(float scale, const std::vector<int64_t>& size);

NPUArray Generator_Normal(float loc, float scale, const std::vector<int64_t>& size);

NPUArray Generator_Uniform(double low, double high, const std::vector<int64_t>& size);

NPUArray Generator_Standard_normal(const std::vector<int64_t>& size);

NPUArray Generator_Standard_cauchy(const std::vector<int64_t>& size);

NPUArray Generator_Weibull(float a, const std::vector<int64_t>& size);

NPUArray Binomial(int n, float p, const std::vector<int64_t>& size);

NPUArray Exponential(float scale, const std::vector<int64_t>& size);

NPUArray Geometric(float p, const std::vector<int64_t>& size);

NPUArray Gumbel(double loc, double scale, const std::vector<int64_t>& size);

NPUArray Laplace(double loc, double scale, const std::vector<int64_t>& size);

NPUArray Logistic(double loc, double scale, const std::vector<int64_t>& size);

NPUArray Lognormal(float mean, float sigma, const std::vector<int64_t>& size);

//NPUArray Multinomial(int64_t n, const NPUArray& pvals, const std::vector<int64_t>& size);

}
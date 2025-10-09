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
/**
 * @brief Compute element-wise hyperbolic sine.
 * 
 * Equivalent to numpy.sinh(x), calculates sinh(x) = (e^x - e^(-x))/2 for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic sine of x
 */
NPUArray Sinh(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise hyperbolic cosine.
 * 
 * Equivalent to numpy.cosh(x), calculates cosh(x) = (e^x + e^(-x))/2 for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic cosine of x
 */
NPUArray Cosh(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise hyperbolic tangent.
 * 
 * Equivalent to numpy.tanh(x), calculates tanh(x) = sinh(x)/cosh(x) for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic tangent of x
 */
NPUArray Tanh(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise inverse hyperbolic sine.
 * 
 * Equivalent to numpy.arcsinh(x), calculates arcsinh(x) = ln(x + √(x² + 1)) for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise inverse hyperbolic sine of x
 */
NPUArray Arcsinh(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise inverse hyperbolic cosine.
 * 
 * Equivalent to numpy.arccosh(x), calculates arccosh(x) = ln(x + √(x² - 1)) for x ≥ 1.
 * 
 * @param x NPUArray, input array (must contain values ≥ 1)
 * @return NPUArray Element-wise inverse hyperbolic cosine of x
 */
NPUArray Arccosh(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise inverse hyperbolic tangent.
 * 
 * Equivalent to numpy.arctanh(x), calculates arctanh(x) = 0.5*ln((1+x)/(1-x)) for |x| < 1.
 * 
 * @param x NPUArray, input array (must contain values with absolute value < 1)
 * @return NPUArray Element-wise inverse hyperbolic tangent of x
 */
NPUArray Arctanh(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

}
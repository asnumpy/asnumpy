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
 * @brief Compute element-wise sign bit check.
 * 
 * Equivalent to numpy.signbit(x), returns a boolean array indicating whether the sign bit is set (negative values).
 * 
 * @param x NPUArray, input array (numeric type)
 * @return NPUArray Boolean array where True indicates negative elements (sign bit set)
 */
NPUArray Signbit(const NPUArray& x);

/**
 * @brief Compute element-wise ldexp operation.
 * 
 * Equivalent to numpy.ldexp(x1, x2), computes x1 * (2 ** x2) element-wise.
 * 
 * @param x1 NPUArray, base array (numeric type)
 * @param x2 NPUArray, exponent array (numeric type)
 * @return NPUArray Resulting array after applying ldexp
 */
NPUArray Ldexp(const NPUArray& x1, const NPUArray& x2);

/**
 * @brief Compute element-wise copysign operation.
 * 
 * Equivalent to numpy.copysign(x1, x2), returns an array with the magnitude of x1 and the sign of x2.
 * 
 * @param x1 NPUArray, magnitude array (numeric type)
 * @param x2 NPUArray, sign array (numeric type)
 * @return NPUArray Resulting array after applying copysign
 */
NPUArray Copysign(const NPUArray& x1, const NPUArray& x2);

}
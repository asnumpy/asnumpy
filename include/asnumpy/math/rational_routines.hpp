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
 * @brief Compute element-wise least common multiple (LCM).
 * 
 * Equivalent to numpy.lcm(x1, x2), returns the smallest positive integer divisible by both x1 and x2.
 * Implemented using the relationship: LCM(a, b) = |a * b| / GCD(a, b)
 * 
 * @param x1 NPUArray, input array (integer type)
 * @param x2 NPUArray, input array (integer type)
 * @return NPUArray Element-wise LCM of x1 and x2
 */
NPUArray Lcm(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise greatest common divisor (GCD).
 * 
 * Equivalent to numpy.gcd(x1, x2), returns the largest positive integer dividing both x1 and x2.
 * 
 * @param x1 NPUArray, input array (integer type)
 * @param x2 NPUArray, input array (integer type)
 * @return NPUArray Element-wise GCD of x1 and x2
 */
NPUArray Gcd(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

}
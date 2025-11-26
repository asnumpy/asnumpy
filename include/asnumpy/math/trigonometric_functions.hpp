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
    * @brief Compute the sine of each element in the input array.
    * 
    * Calculates element-wise sine values on NPU by calling aclnnSin.
    * 
    * @param x Input array.
    * @return NPUArray Array with element-wise sine values.
    * @throws std::runtime_error If ACL operation fails.
    */
    NPUArray Sin(const NPUArray& x);

    /**
    * @brief Compute the cosine of each element in the input array.
    * 
    * Calculates element-wise cosine values on NPU by calling aclnnCos.
    * 
    * @param x Input array.
    * @return NPUArray Array with element-wise cosine values.
    * @throws std::runtime_error If ACL operation fails.
    */
    NPUArray Cos(const NPUArray& x);

    /**
    * @brief Compute the tangent of each element in the input array.
    * 
    * Calculates element-wise tangent values on NPU by calling aclnnTan.
    * 
    * @param x Input array.
    * @return NPUArray Array with element-wise tangent values.
    * @throws std::runtime_error If ACL operation fails.
    */
    NPUArray Tan(const NPUArray& x);

    /**
    * @brief Compute the inverse sine (arcsin) of each element in the input array.
    *
    * Calculates element-wise arcsin values on NPU by calling aclnnAsin.
    *
    * @param x Input array.
    * @return NPUArray Array with element-wise arcsin values.
    * @throws std::runtime_error If ACL operation fails.
    */
    NPUArray Arcsin(const NPUArray& x);

    /**
    * @brief Compute the inverse cosine (arccos) of each element in the input array.
    *
    * Calculates element-wise arccos values on NPU by calling aclnnAcos.
    *
    * @param x Input array.
    * @return NPUArray Array with element-wise arccos values.
    * @throws std::runtime_error If ACL operation fails.
    */
    NPUArray Arccos(const NPUArray& x);

    /**
    * @brief Compute the element-wise arc tangent of input array.
    * 
    * Applies the inverse tangent function to each element of the input array.
    * 
    * @param x Input NPUArray.
    * @return NPUArray Array where each element is the arctangent of the corresponding input element.
    * @throws std::runtime_error If ACL operation returns an error.
    */
    NPUArray Arctan(const NPUArray& x);

    /**
    * @brief Compute the hypotenuse of two arrays element-wise.
    * 
    * This function calculates the hypotenuse for each pair of elements from the input arrays `a` and `b`
    * using the formula: hypot(a, b) = sqrt(a^2 + b^2). It supports broadcasting of input shapes.
    * 
    * @param a NPUArray, first input array
    * @param b NPUArray, second input array
    * @return NPUArray Resulting array containing the hypotenuse values
    * @throws std::invalid_argument If input shapes are not broadcastable
    * @throws std::runtime_error If ACL operations encounter errors
    */
    NPUArray Hypot(const NPUArray& a, const NPUArray& b);

    /**
    * @brief Compute the element-wise arc tangent of y/x considering the quadrant.
    * 
    * This function computes the angle (in radians) between the positive x-axis and the point (x, y)
    * for each pair of elements from the input arrays `y` and `x`. It uses the aclnnAtan2 operator
    * to handle the quadrant correctly.
    * 
    * @param y NPUArray, input array representing the y-coordinates
    * @param x NPUArray, input array representing the x-coordinates
    * @return NPUArray Array where each element is the angle in radians corresponding to atan2(y, x)
    * @throws std::invalid_argument If input shapes are not broadcastable
    * @throws std::runtime_error If ACL operations encounter errors
    */
    NPUArray Arctan2(const NPUArray& y, const NPUArray& x);

    /**
    * @brief Convert angles from degrees to radians.
    * 
    * This function converts each element in the input array from degrees to radians
    * using the formula: radians = degrees * (π / 180).
    * 
    * @param x NPUArray, input array with angles in degrees
    * @return NPUArray Array with angles converted to radians
    * @throws std::runtime_error If ACL operations encounter errors
    */
    NPUArray Radians(const NPUArray& x);

    /**
    * @brief Convert angles from radians to degrees for each element in the input array.
    * 
    * Computes element-wise angle conversion using the formula:
    *      degrees = radians * (180 / π)
    * 
    * @param x Input array containing angles in radians.
    * @return NPUArray Array with element-wise angles in degrees.
    * @throws std::runtime_error If ACL operation fails or unsupported dtype.
    */
    NPUArray Degrees(const NPUArray& x);
}
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

#include "../utils/npu_array.hpp"
#include <utility>
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_cumprod.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_sinh.h>
#include <aclnnop/aclnn_cosh.h>
#include <aclnnop/aclnn_tanh.h>
#include <aclnnop/aclnn_logaddexp.h>
#include <aclnnop/aclnn_logaddexp2.h>
#include <aclnnop/aclnn_sinc.h>
#include <aclnnop/aclnn_real.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_log10.h>
#include <aclnnop/aclnn_log2.h>
#include <aclnnop/aclnn_log1p.h>
#include <aclnnop/aclnn_linalg_cross.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_expm1.h>
#include <aclnnop/aclnn_exp2.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_atan2.h>
#include <aclnnop/aclnn_prod.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_sum.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_asinh.h>
#include <aclnnop/aclnn_acosh.h>
#include <aclnnop/aclnn_atanh.h>
#include <aclnnop/aclnn_ceil.h>
#include <aclnnop/aclnn_trunc.h>
#include <aclnnop/aclnn_signbit.h>
#include <aclnnop/aclnn_gcd.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_tan.h>
#include <aclnnop/aclnn_asin.h>
#include <aclnnop/aclnn_acos.h>
#include <aclnnop/aclnn_atan.h>
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_reciprocal.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_floor_divide.h>


NPUArray Cumprod(const NPUArray& a, int64_t axis, py::dtype dtype);

NPUArray Cumsum(const NPUArray& a, int64_t axis, py::dtype dtype);

NPUArray Nancumprod(const NPUArray& a, int64_t axis, py::dtype dtype);

NPUArray Nancumsum(const NPUArray& a, int64_t axis, py::dtype dtype);

NPUArray Logaddexp(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

NPUArray Logaddexp2(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

NPUArray Sinc(const NPUArray& x);

NPUArray Real(const NPUArray& val);

NPUArray Log(const NPUArray& x, py::dtype dtype);

NPUArray Log10(const NPUArray& x, py::dtype dtype);

NPUArray Log2(const NPUArray& x, py::dtype dtype);

NPUArray Log1p(const NPUArray& x, py::dtype dtype);

NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axisa, int64_t axisb, int64_t axisc, int64_t axis);

NPUArray Exp(const NPUArray& x, py::dtype dtype);

NPUArray Expm1(const NPUArray& x, py::dtype dtype);

NPUArray Exp2(const NPUArray& x, py::dtype dtype);

NPUArray Prod(const NPUArray& a, int64_t axis, py::dtype dtype, bool keepdims);
NPUArray Prod(const NPUArray& a, py::dtype dtype);

NPUArray Sum(const NPUArray& a, const std::vector<int64_t>& axis, py::dtype dtype, bool keepdims);
NPUArray Sum(const NPUArray& a, py::dtype dtype);
    
NPUArray Nanprod(const NPUArray& a, int64_t axis, py::dtype dtype, bool keepdims);
NPUArray Nanprod(const NPUArray& a, py::dtype dtype);

NPUArray Nansum(const NPUArray& a, const std::vector<int64_t>& axis, py::dtype dtype, bool keepdims);
NPUArray Nansum(const NPUArray& a, py::dtype dtype);

NPUArray Hypot(const NPUArray& a, const NPUArray& b);

NPUArray Arctan2(const NPUArray& y, const NPUArray& x);

NPUArray Radians(const NPUArray& x);

NPUArray Sinh(const NPUArray& x);

NPUArray Cosh(const NPUArray& x);

NPUArray Tanh(const NPUArray& x);

NPUArray Arcsinh(const NPUArray& x);

NPUArray Arccosh(const NPUArray& x);

NPUArray Arctanh(const NPUArray& x);

NPUArray Ceil(const NPUArray& x);

NPUArray Trunc(const NPUArray& x);

NPUArray Signbit(const NPUArray& x);

NPUArray Lcm(const NPUArray& x1, const NPUArray& x2);

NPUArray Gcd(const NPUArray& x1, const NPUArray& x2);

NPUArray FloatPower(const NPUArray& x1, const NPUArray& x2);

NPUArray Fmod(const NPUArray& x1, const NPUArray& x2);

NPUArray Mod(const NPUArray& x1, const NPUArray& x2);

std::pair<NPUArray, NPUArray> Modf(const NPUArray& x);

NPUArray Remainder(const NPUArray& x1, const NPUArray& x2);

std::pair<NPUArray, NPUArray> Divmod(const NPUArray& x1, const NPUArray& x2);

/**
 * @brief Compute the sine of each element in the input array.
 * 
 * Calculates element-wise sine values on NPU by calling aclnnSin.
 * 
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise sine values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray sin(const NPUArray& x, py::dtype dtype);

/**
 * @brief Compute the cosine of each element in the input array.
 * 
 * Calculates element-wise cosine values on NPU by calling aclnnCos.
 * 
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise cosine values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray cos(const NPUArray& x, py::dtype dtype);

/**
 * @brief Compute the tangent of each element in the input array.
 * 
 * Calculates element-wise tangent values on NPU by calling aclnnTan.
 * 
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise tangent values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray tan(const NPUArray& x, py::dtype dtype);

/**
 * @brief Compute the inverse sine (arcsin) of each element in the input array.
 *
 * Calculates element-wise arcsin values on NPU by calling aclnnAsin.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise arcsin values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray arcsin(const NPUArray& x, py::dtype dtype);

/**
 * @brief Compute the inverse cosine (arccos) of each element in the input array.
 *
 * Calculates element-wise arccos values on NPU by calling aclnnAcos.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise arccos values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray arccos(const NPUArray& x, py::dtype dtype);

/**
 * @brief Compute the element-wise arc tangent of input array.
 * 
 * Applies the inverse tangent function to each element of the input array.
 * 
 * @param x Input NPUArray.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array where each element is the arctangent of the corresponding input element.
 * @throws std::runtime_error If ACL operation returns an error.
 */
NPUArray Arctan(const NPUArray& x, py::dtype dtype);

/**
 * @brief Round elements of the array to the given number of decimals.
 *
 * Uses aclnnRoundDecimals to round each element to the specified number of decimal places.
 *
 * @param a Input array.
 * @param decimals Number of decimal places to round to (default 0).
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the specified decimals.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray around(const NPUArray& a, int decimals, py::dtype dtype);

/**
 * @brief Round elements of the array to the given number of decimals.
 *
 * Equivalent to around(). Provided for API compatibility with NumPy.
 *
 * @param a Input array.
 * @param decimals Number of decimal places to round to (default 0).
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the specified decimals.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray round_(const NPUArray& a, int decimals, py::dtype dtype);

/**
 * @brief Round elements of the array to the nearest integer.
 *
 * Calculates element-wise nearest integers on NPU by calling aclnnRound.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the nearest integer.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray rint(const NPUArray& x, py::dtype dtype);

/**
 * @brief Truncate elements of the array towards zero.
 *
 * Applies element-wise truncation on NPU by calling aclnnTrunc.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with truncated values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray fix(const NPUArray& x, py::dtype dtype);

/**
 * @brief Compute the floor of each element in the input array.
 *
 * Applies element-wise floor operation on NPU by calling aclnnFloor.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with floored values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray floor(const NPUArray& x, py::dtype dtype);

/**
 * @brief Element-wise addition of two arrays.
 *
 * Computes x1 + x2 on NPU by calling aclnnAdd.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise sums.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray add(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Compute the reciprocal (1/x) of each element in the input array.
 *
 * Applies element-wise reciprocal on NPU by calling aclnnReciprocal.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise reciprocals.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray reciprocal(const NPUArray& x, py::dtype dtype);

/**
 * @brief Return the input array itself (with optional dtype conversion).
 *
 * Equivalent to applying the unary plus operator. No numerical change occurs.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Same array values, possibly with a new dtype.
 */
NPUArray positive(const NPUArray& x, py::dtype dtype);

/**
 * @brief Compute the numerical negative of each element in the input array.
 *
 * Applies element-wise negation on NPU by calling aclnnNeg.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise negated values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray negative(const NPUArray& x, py::dtype dtype);

/**
 * @brief Element-wise multiplication of two arrays.
 *
 * Computes x1 * x2 on NPU by calling aclnnMul.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise products.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray multiply(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Element-wise true division of two arrays.
 *
 * Computes x1 / x2 on NPU by calling aclnnDiv.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise quotients.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray divide(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Element-wise subtraction of two arrays.
 *
 * Computes x1 - x2 on NPU by calling aclnnSub.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise differences.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray subtract(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Element-wise true division of two arrays (alias of divide).
 *
 * Equivalent to divide(). Provided for NumPy API compatibility.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise quotients.
 */
NPUArray true_divide(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Element-wise floor division of two arrays.
 *
 * Computes floor(x1 / x2) on NPU by calling aclnnFloorDivide.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise floor division results.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray floor_divide(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);
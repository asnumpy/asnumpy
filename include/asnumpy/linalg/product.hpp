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

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_einsum.h>
#include <aclnnop/aclnn_eye.h>
#include <aclnnop/aclnn_inverse.h>
#include <aclnnop/aclnn_matmul.h>
#include <utility>
#include "../utils/npu_array.hpp"

NPUArray Matmul(const NPUArray& x1, const NPUArray& x2);

NPUArray Einsum(const char* subscripts, const std::vector<NPUArray>& operands);

NPUArray Matrix_power(const NPUArray& a, int64_t n);

/**
 * @brief Compute the dot product of two arrays.
 *
 * - If both inputs are scalars: returns scalar multiplication.
 * - If both are 1D arrays: returns inner product (scalar).
 * - If both are 2D arrays: returns matrix multiplication.
 * - Otherwise: performs generalized dot product on the last axis of `a` and the second-to-last axis of `b`.
 *
 * Uses aclnnDot for 1D dot products and aclnnMm for 2D matrix multiplication.
 *
 * @param a First input array.
 * @param b Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Result of the dot product.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray dot(const NPUArray& a, const NPUArray& b);

/**
 * @brief Compute the dot product of two arrays after flattening.
 *
 * Both input arrays are flattened using aclnnFlatten(axis=0) to get (1, N),
 * then reshaped into (N,) (requires reshape support). Finally, aclnnDot
 * is used to compute the inner product.
 *
 * Note: The project does not yet implement reshape. This code assumes
 * reshape exists; implementation will be added later.
 *
 * @param a First input array.
 * @param b Second input array.
 * @param dtype Target numpy dtype for the output.
 * @return NPUArray A scalar (0D array) containing the dot product.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray vdot(const NPUArray& a, const NPUArray& b);

/**
 * @brief Compute the inner product of two arrays.
 *
 * - If both inputs are 1D arrays: returns scalar dot product.
 * - If both inputs are 2D arrays: returns matrix product (m × k) · (k × n) → (m × n).
 * - For higher dimensions: currently not supported, will throw exception.
 *
 * Uses aclnnDot for 1D and aclnnMm for 2D.
 *
 * @param a First input array.
 * @param b Second input array.
 * @param dtype Target numpy dtype for the output.
 * @return NPUArray Result of the inner product.
 * @throws std::runtime_error If ACL operation fails or unsupported input dims.
 */
NPUArray inner(const NPUArray& a, const NPUArray& b);

/**
 * @brief Compute the outer product of two arrays.
 *
 * Both input arrays are flattened to 2D using aclnnFlatten:
 * - 'a' transformed to shape (m, 1) using axis = 1
 * - 'b' transformed to shape (1, n) using axis = 0
 * Then, element-wise multiplication yields an output of shape (m, n).
 *
 * @param a First input array.
 * @param b Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Result of the outer product with shape (a.size, b.size).
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray outer(const NPUArray& a, const NPUArray& b);

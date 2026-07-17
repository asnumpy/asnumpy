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
#include <optional>

namespace asnumpy {

/**
 * @brief NumPy-exact type promotion for two ACL dtypes.
 *
 * Delegates to numpy.result_type rather than reimplementing the promotion lattice, so asnumpy
 * cannot drift from NumPy. tests/asnumpy_tests/interop_tests/test_numpy_baseline.py pins
 * numpy.result_type as the oracle for exactly this reason.
 *
 * Results are memoized per (a, b) pair, so the Python call happens once per dtype combination
 * rather than once per operation.
 *
 * Only array-array promotion is modelled. NEP 50 weak scalar semantics (a Python int or float
 * operand binding weakly) are not implemented yet; scalar overloads still take the array's dtype.
 *
 * @param a First ACL dtype.
 * @param b Second ACL dtype.
 * @return The promoted ACL dtype.
 * @throws std::invalid_argument If either input, or the promoted result, has no NumPy equivalent.
 */
aclDataType ResultType(aclDataType a, aclDataType b);

/**
 * @brief Two binary operands promoted to their common dtype.
 *
 * Casts an operand only when its dtype differs from the common type, so the overwhelmingly common
 * same-dtype case costs nothing beyond a pointer. Any cast result is owned by this object, so it
 * must outlive the references handed out by x1() and x2().
 *
 * Non-copyable and non-movable: the accessors return references into internal storage.
 */
class PromotedOperands {
  public:
    /// Promote to result_type(x1, x2).
    PromotedOperands(const NPUArray& x1, const NPUArray& x2);

    /// Promote to an explicitly requested common dtype (used when a caller forces the operand type).
    PromotedOperands(const NPUArray& x1, const NPUArray& x2, aclDataType common);

    PromotedOperands(const PromotedOperands&) = delete;
    PromotedOperands& operator=(const PromotedOperands&) = delete;
    PromotedOperands(PromotedOperands&&) = delete;
    PromotedOperands& operator=(PromotedOperands&&) = delete;

    const NPUArray& x1() const { return *x1_; }
    const NPUArray& x2() const { return *x2_; }

    /// The dtype both operands now share.
    aclDataType common() const { return common_; }

  private:
    void Materialize(const NPUArray& x1, const NPUArray& x2);

    std::optional<NPUArray> x1_storage_;
    std::optional<NPUArray> x2_storage_;
    const NPUArray* x1_ = nullptr;
    const NPUArray* x2_ = nullptr;
    aclDataType common_;
};

} // namespace asnumpy

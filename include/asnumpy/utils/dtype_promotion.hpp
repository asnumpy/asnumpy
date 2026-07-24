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

namespace asnumpy {

/** True for float16 / float / double / complex floating ACL dtypes. */
bool IsFloatingAclDtype(aclDataType t);

/**
 * NumPy-like unary floating promotion for transcendental ops (exp/log/sinh/...):
 * keep floating/complex input dtype; otherwise promote to float64.
 */
aclDataType PromoteUnaryFloating(aclDataType in);

/**
 * NumPy-like binary floating promotion (e.g. logaddexp):
 * integers/bool promote toward float64; among floats pick the wider type.
 */
aclDataType PromoteBinaryFloating(aclDataType a, aclDataType b);

/** Cast device array to target ACL dtype via aclnnCast (no-op copy when already matching). */
NPUArray CastToDtype(const NPUArray& input, aclDataType targetDtype);

/** Return input unchanged (by value copy ctor path) if dtype matches; otherwise cast. */
NPUArray EnsureAclDtype(const NPUArray& input, aclDataType targetDtype);

/**
 * If desired output is float64 but the ACL op only supports float32/float16,
 * compute in float32 then cast the result to float64 so NumPy dtypes still match.
 */
inline aclDataType AclComputeFloatingDtype(aclDataType desired, bool supports_float64) {
    if (desired == ACL_DOUBLE && !supports_float64) {
        return ACL_FLOAT;
    }
    return desired;
}

} // namespace asnumpy

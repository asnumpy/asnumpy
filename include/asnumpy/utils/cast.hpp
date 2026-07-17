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

/**
 * @brief Cast an array to `targetDtype` on device, via aclnnCast.
 *
 * Returns a deep copy unchanged when the array is already `targetDtype`, so callers can invoke it
 * unconditionally. Takes an aclDataType rather than a py::dtype so no NumPy round trip is involved.
 *
 * This is the single cast primitive: the promotion layer, `astype`, and the ops that need to widen
 * an operand all route through it.
 *
 * @param input Array to cast.
 * @param targetDtype Desired ACL element type.
 * @return NPUArray of the same shape with element type `targetDtype`.
 * @throws std::runtime_error If the ACL cast fails (e.g. the kernel lacks this dtype pair).
 */
NPUArray CastTo(const NPUArray& input, aclDataType targetDtype);

} // namespace asnumpy

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
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace asnumpy::dtypes {

/**
 * Single source of truth for the NumPy <-> ACL dtype correspondence.
 *
 * AclFromNumpy and NumpyFromAcl are exact inverses over the supported set. That property is what
 * the operator layer relies on: an op resolves an aclDataType, converts it with NumpyFromAcl, and
 * hands the result to NPUArray, which converts back with AclFromNumpy. If the round trip were
 * lossy the op's chosen type would be silently rewritten.
 *
 * The supported set is the 14 dtypes ACL and NumPy both represent natively. ACL types with no
 * NumPy equivalent (bf16, fp8/fp6/fp4, int4, uint1, complex32) are deliberately absent: an array
 * of such a type cannot be handed to Python without either a lie or a dependency such as
 * ml_dtypes, so NumpyFromAcl rejects them rather than widening them.
 */

/// True if `acl` has an exact NumPy equivalent.
bool IsSupported(aclDataType acl);

/// True if `acl` is a floating or complex type (NumPy's "inexact" kinds, 'f' and 'c').
///
/// Ops whose NumPy signature has no integer loop -- true_divide ('ee->e','ff->f','dd->d','FF->F',
/// 'DD->D') and ldexp ('ei->e','fi->f','di->d') -- use this to widen integer operands to float64
/// instead of truncating them.
bool IsInexact(aclDataType acl);

/// NumPy dtype -> aclDataType.
/// Equivalent types with different type numbers (int64 vs longlong) resolve identically.
/// @throws std::invalid_argument if `dtype` is big-endian, structured, or has no ACL equivalent.
aclDataType AclFromNumpy(const py::dtype& dtype);

/// aclDataType -> NumPy dtype. Exact inverse of AclFromNumpy over the supported set.
/// @throws std::invalid_argument if `acl` has no NumPy equivalent.
py::dtype NumpyFromAcl(aclDataType acl);

/// Byte size of one element of `acl`. Defined for every ACL type asnumpy may encounter,
/// including unsupported ones, so byte-size math stays available for diagnostics.
/// @throws std::invalid_argument if `acl` is unknown.
int64_t ItemSize(aclDataType acl);

/// Human-readable name of `acl`, for error messages. Never throws; returns "<unknown>" if unknown.
const char* Name(aclDataType acl);

} // namespace asnumpy::dtypes

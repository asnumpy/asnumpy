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

/**
 * @file registry.hpp
 * @brief Public APIs for AsNumpy custom dtype registration and lookup.
 */
#pragma once

#include <acl/acl.h>
#include <asnumpy/dtypes/np_import.hpp>

namespace asnumpy {
namespace dtypes {

/**
 * @brief Initialize NumPy C-API (if needed) and register all custom dtypes.
 *
 * This function is designed to be idempotent.
 */
void InitAndRegisterDtypes();

/**
 * @brief Lookup a registered NumPy dtype descriptor for a given ACL dtype.
 *
 * @return Descriptor pointer if registered; otherwise nullptr.
 *
 * Note: The returned pointer is owned by NumPy. Callers should INCREF if they
 * need to hold onto it.
 */
PyArray_Descr* RegisteredArrayDescrForAclType(aclDataType acl_type);

/**
 * @brief Try to map a NumPy dtype descriptor to an ACL dtype.
 *
 * This recognizes AsNumpy-registered custom dtypes (float8/bf16/int4/etc).
 * Returns true on success and writes to out_acl_type; otherwise returns false.
 *
 * Note: Callers should pass a valid `PyArray_Descr*` pointer.
 */
bool TryGetAclTypeFromArrayDescr(PyArray_Descr* descr, aclDataType& out_acl_type);

/**
 * @brief Check whether all custom float dtypes have been registered.
 */
bool AreAllACLFloatTypesRegistered();

/**
 * @brief Check whether all custom int dtypes have been registered.
 */
bool AreAllACLIntTypesRegistered();

}  // namespace dtypes
}  // namespace asnumpy


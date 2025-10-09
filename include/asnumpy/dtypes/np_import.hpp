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

// Prevent multiple inclusion
#pragma once

#ifdef PyArray_Type
#error "NumPy headers must not be included before np_import.hpp"
#endif

// Disallow NumPy 1.7 deprecated symbols
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// Unique NumPy C-API symbol for this extension
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL _asnumpy_numpy_api
#endif

// Only the one translation unit defining ASNUMPY_IMPORT_NUMPY should import the API
#ifndef ASNUMPY_IMPORT_NUMPY
#ifndef NO_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#endif

// Place <locale> before <Python.h> for macOS compatibility (harmless elsewhere)
#include <locale>
#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

// NumPy 2.x compatibility shim for descriptor prototype type name
#ifndef NPY_ABI_VERSION
#error "NumPy headers did not define NPY_ABI_VERSION"
#endif

#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

namespace asnumpy{
    namespace dtypes{
        void ImportNumpy();
    }
}
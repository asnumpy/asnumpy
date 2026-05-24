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

#include <algorithm>
#include <vector>

#include <asnumpy/dtypes/acl_float_reg.hpp>
#include <asnumpy/dtypes/acl_int_reg.hpp>
#include <asnumpy/dtypes/desc.hpp>
#include <asnumpy/dtypes/float_types.hpp>
#include <asnumpy/dtypes/int_types.hpp>

namespace asnumpy {
namespace dtypes {

// Explicit specialization of static members for all ACL float types
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float8_e5m2)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float8_e4m3fn)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float8_e8m0)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(bfloat16)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float6_e2m3fn)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float6_e3m2fn)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float4_e2m1fn)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float4_e1m2fn)

// Explicit specialization of static members for all ACL integer types
EXPLICIT_INSTANTIATE_ACL_INT_MANAGER(int4)
EXPLICIT_INSTANTIATE_ACL_INT_MANAGER(uint1)

// TypeDescriptor specializations for concrete types
// TypeDescriptor specialization for float8_e5m2
template <> struct TypeDescriptor<float8_e5m2> : ACLFloatManager<float8_e5m2> {
    using T = float8_e5m2;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float8_e5m2";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float8_e5m2";
    static constexpr const char* kTpDoc = "Float8 E5M2 floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '5';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for float8_e4m3fn
template <> struct TypeDescriptor<float8_e4m3fn> : ACLFloatManager<float8_e4m3fn> {
    using T = float8_e4m3fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float8_e4m3fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float8_e4m3fn";
    static constexpr const char* kTpDoc = "Float8 E4M3FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '6';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for float8_e8m0
template <> struct TypeDescriptor<float8_e8m0> : ACLFloatManager<float8_e8m0> {
    using T = float8_e8m0;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float8_e8m0";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float8_e8m0";
    static constexpr const char* kTpDoc = "Float8 E8M0 floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '7';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for bfloat16
template <> struct TypeDescriptor<bfloat16> : ACLFloatManager<bfloat16> {
    using T = bfloat16;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "bfloat16";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.bfloat16";
    static constexpr const char* kTpDoc = "BFloat16 floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '8';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for float6_e2m3fn
template <> struct TypeDescriptor<float6_e2m3fn> : ACLFloatManager<float6_e2m3fn> {
    using T = float6_e2m3fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float6_e2m3fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float6_e2m3fn";
    static constexpr const char* kTpDoc = "Float6 E2M3FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '9';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for float6_e3m2fn
template <> struct TypeDescriptor<float6_e3m2fn> : ACLFloatManager<float6_e3m2fn> {
    using T = float6_e3m2fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float6_e3m2fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float6_e3m2fn";
    static constexpr const char* kTpDoc = "Float6 E3M2FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = 'A';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for float4_e2m1fn
template <> struct TypeDescriptor<float4_e2m1fn> : ACLFloatManager<float4_e2m1fn> {
    using T = float4_e2m1fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float4_e2m1fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float4_e2m1fn";
    static constexpr const char* kTpDoc = "Float4 E2M1FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = 'B';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for float4_e1m2fn
template <> struct TypeDescriptor<float4_e1m2fn> : ACLFloatManager<float4_e1m2fn> {
    using T = float4_e1m2fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float4_e1m2fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float4_e1m2fn";
    static constexpr const char* kTpDoc = "Float4 E1M2FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = 'C';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for int4
template <> struct TypeDescriptor<int4> : ACLIntManager<int4> {
    using T = int4;

    static constexpr bool is_floating = false;
    static constexpr bool is_integral = true;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "int4";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.int4";
    static constexpr const char* kTpDoc = "4-bit signed integer values";

    static constexpr char kNpyDescrKind = 'i';
    static constexpr char kNpyDescrType = 'D';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor specialization for uint1
template <> struct TypeDescriptor<uint1> : ACLIntManager<uint1> {
    using T = uint1;

    static constexpr bool is_floating = false;
    static constexpr bool is_integral = true;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "uint1";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.uint1";
    static constexpr const char* kTpDoc = "1-bit unsigned integer values";

    static constexpr char kNpyDescrKind = 'u';
    static constexpr char kNpyDescrType = 'E';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// Public initialization and registration entry point
void InitAndRegisterDtypes() {
    // 1) ensure NumPy C API is imported only once
    ImportNumpy();

    // 2) register all ACL float types
    FloatTypeRegistrar<float8_e5m2>::RegisterDtype();
    FloatTypeRegistrar<float8_e4m3fn>::RegisterDtype();
    FloatTypeRegistrar<float8_e8m0>::RegisterDtype();
    FloatTypeRegistrar<bfloat16>::RegisterDtype();
    FloatTypeRegistrar<float6_e2m3fn>::RegisterDtype();
    FloatTypeRegistrar<float6_e3m2fn>::RegisterDtype();
    FloatTypeRegistrar<float4_e2m1fn>::RegisterDtype();
    FloatTypeRegistrar<float4_e1m2fn>::RegisterDtype();

    // 3) register all ACL integer types
    IntTypeRegistrar<int4>::RegisterDtype();
    IntTypeRegistrar<uint1>::RegisterDtype();
}

// Check whether all types are registered
bool AreAllACLFloatTypesRegistered() {
    return ACLFloatManager<float8_e5m2>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float8_e4m3fn>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float8_e8m0>::npy_type != NPY_NOTYPE && ACLFloatManager<bfloat16>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float6_e2m3fn>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float6_e3m2fn>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float4_e2m1fn>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float4_e1m2fn>::npy_type != NPY_NOTYPE;
}

// Check whether all ACL integer types are registered
bool AreAllACLIntTypesRegistered() {
    return ACLIntManager<int4>::npy_type != NPY_NOTYPE && ACLIntManager<uint1>::npy_type != NPY_NOTYPE;
}

PyArray_Descr* RegisteredArrayDescrForAclType(aclDataType acl_type) {
    switch (acl_type) {
    case ACL_BF16:
        return ACLFloatManager<bfloat16>::npy_descr;
    case ACL_FLOAT8_E5M2:
        return ACLFloatManager<float8_e5m2>::npy_descr;
    case ACL_FLOAT8_E4M3FN:
        return ACLFloatManager<float8_e4m3fn>::npy_descr;
    case ACL_FLOAT8_E8M0:
        return ACLFloatManager<float8_e8m0>::npy_descr;
    case ACL_FLOAT6_E3M2:
        return ACLFloatManager<float6_e3m2fn>::npy_descr;
    case ACL_FLOAT6_E2M3:
        return ACLFloatManager<float6_e2m3fn>::npy_descr;
    case ACL_FLOAT4_E2M1:
        return ACLFloatManager<float4_e2m1fn>::npy_descr;
    case ACL_FLOAT4_E1M2:
        return ACLFloatManager<float4_e1m2fn>::npy_descr;
    case ACL_INT4:
        return ACLIntManager<int4>::npy_descr;
    case ACL_UINT1:
        return ACLIntManager<uint1>::npy_descr;
    default:
        return nullptr;
    }
}

// Get dtype type number for a specific type
template <typename T> int GetACLFloatTypeNum() { return ACLFloatManager<T>::npy_type; }

// Get dtype descriptor for a specific type
template <typename T> PyArray_Descr* GetACLFloatDescr() {
    if (ACLFloatManager<T>::npy_descr == nullptr) {
        return nullptr;
    }
    Py_INCREF(ACLFloatManager<T>::npy_descr);
    return ACLFloatManager<T>::npy_descr;
}

// Create array of a specific type
template <typename T>
PyObject* CreateACLFloatArray(const std::vector<npy_intp>& shape, const std::vector<float>& data = {}) {
    int type_num = GetACLFloatTypeNum<T>();
    if (type_num == NPY_NOTYPE) {
        PyErr_SetString(PyExc_RuntimeError, "dtype not registered");
        return nullptr;
    }

    // create array
    PyObject* array = PyArray_EMPTY(static_cast<int>(shape.size()), const_cast<npy_intp*>(shape.data()), type_num, 0);
    if (array == nullptr) {
        return nullptr;
    }

    // fill data
    if (!data.empty()) {
        T* array_data = static_cast<T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
        for (size_t i = 0;
             i < std::min(data.size(), static_cast<size_t>(PyArray_SIZE(reinterpret_cast<PyArrayObject*>(array))));
             ++i) {
            array_data[i] = T(data[i]);
        }
    }

    return array;
}

// Get float data from array
template <typename T> std::vector<float> GetACLFloatArrayData(PyObject* array) {
    if (!PyArray_Check(array)) {
        return {};
    }

    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(array);
    if (PyArray_TYPE(arr) != GetACLFloatTypeNum<T>()) {
        return {};
    }

    npy_intp size = PyArray_SIZE(arr);
    T* data = static_cast<T*>(PyArray_DATA(arr));

    std::vector<float> result;
    result.reserve(size);

    for (npy_intp i = 0; i < size; ++i) {
        result.push_back(static_cast<float>(data[i]));
    }

    return result;
}

// Get type object pointer (consistent with acl_float_reg.hpp declaration)
template <typename T> PyObject* GetACLFloatTypeObject() { return ACLFloatManager<T>::type_ptr; }

template <typename T> PyObject* GetACLIntTypeObject() { return ACLIntManager<T>::type_ptr; }

// Explicit instantiations for linking GetACL*TypeObject from other translation units
template PyObject* GetACLFloatTypeObject<float8_e5m2>();
template PyObject* GetACLFloatTypeObject<float8_e4m3fn>();
template PyObject* GetACLFloatTypeObject<float8_e8m0>();
template PyObject* GetACLFloatTypeObject<bfloat16>();
template PyObject* GetACLFloatTypeObject<float6_e2m3fn>();
template PyObject* GetACLFloatTypeObject<float6_e3m2fn>();
template PyObject* GetACLFloatTypeObject<float4_e2m1fn>();
template PyObject* GetACLFloatTypeObject<float4_e1m2fn>();

template PyObject* GetACLIntTypeObject<int4>();
template PyObject* GetACLIntTypeObject<uint1>();

} // namespace dtypes
} // namespace asnumpy

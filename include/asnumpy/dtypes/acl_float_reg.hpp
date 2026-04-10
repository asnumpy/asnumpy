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

#include <asnumpy/dtypes/acl_manager_members.hpp>
#include <asnumpy/dtypes/acl_reg_common.hpp>
#include <cstdint>

namespace asnumpy {
namespace dtypes {

template <typename T>
struct ACLFloatManager {
    ASNUMPY_ACL_MANAGER_COMMON_MEMBERS
    static int Dtype() { return npy_type; }
};

template <typename T>
struct PyACLScalar {
    PyObject_HEAD;
    T value;
};

struct ACLFloatPolicy {
    template <typename T>
    static void InitZero(T& v) {
        v = T(0.0f);
    }

    template <typename T>
    static PyObject* GetItem(void* data) {
        T val = *static_cast<T*>(data);
        float float_val = static_cast<float>(val);
        return PyFloat_FromDouble(static_cast<double>(float_val));
    }

    template <typename T>
    static int SetItem(PyObject* obj, void* data) {
        float val;
        if (PyFloat_Check(obj)) {
            val = static_cast<float>(PyFloat_AsDouble(obj));
        } else if (PyLong_Check(obj)) {
            val = static_cast<float>(PyLong_AsDouble(obj));
        } else {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to float type");
            return -1;
        }
        *static_cast<T*>(data) = T(val);
        return 0;
    }

    template <typename T>
    static int ScalarInit(PyObject* obj, T& out) {
        if (PyFloat_Check(obj)) {
            out = T(static_cast<float>(PyFloat_AsDouble(obj)));
            return 0;
        }
        if (PyLong_Check(obj)) {
            out = T(static_cast<float>(PyLong_AsDouble(obj)));
            return 0;
        }
        PyErr_SetString(PyExc_TypeError, "Cannot convert to ACL float type");
        return -1;
    }

    template <typename T, typename Scalar>
    static PyObject* ScalarRepr(Scalar* self) {
        float val = static_cast<float>(self->value);
        return PyUnicode_FromFormat("%s(%g)", TypeDescriptor<T>::kTypeName, val);
    }

    template <typename T, typename Scalar>
    static PyObject* ScalarStr(Scalar* self) {
        float val = static_cast<float>(self->value);
        return PyUnicode_FromFormat("%g", val);
    }

    template <typename T>
    static PyObject* ScalarRepr(PyACLScalar<T>* self) {
        return ScalarRepr<T, PyACLScalar<T>>(self);
    }

    template <typename T>
    static PyObject* ScalarStr(PyACLScalar<T>* self) {
        return ScalarStr<T, PyACLScalar<T>>(self);
    }

    template <typename U, typename T>
    static U ConvertToPeer(const T& v) {
        return U(static_cast<float>(v));
    }

    template <typename U, typename T>
    static T ConvertFromPeer(const U& v) {
        return T(static_cast<float>(v));
    }

    template <typename T, template <typename, template <typename> class, template <typename> class, class> class RegistrarT>
    static void RegisterBuiltins(int type_num) {
        using Registrar = RegistrarT<T, ACLFloatManager, PyACLScalar, ACLFloatPolicy>;
        PyArray_RegisterCastFunc(PyArray_DescrFromType(type_num), NPY_FLOAT,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(
                                     Registrar::template cast_to_builtin<float>));
        PyArray_RegisterCastFunc(PyArray_DescrFromType(NPY_FLOAT), type_num,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(
                                     Registrar::template cast_from_builtin<float>));

        PyArray_RegisterCastFunc(PyArray_DescrFromType(type_num), NPY_DOUBLE,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(
                                     Registrar::template cast_to_builtin<double>));
        PyArray_RegisterCastFunc(PyArray_DescrFromType(NPY_DOUBLE), type_num,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(
                                     Registrar::template cast_from_builtin<double>));
    }

    template <typename T, template <typename, template <typename> class, template <typename> class, class> class RegistrarT>
    static void RegisterPeers(int type_num) {
        using Registrar = RegistrarT<T, ACLFloatManager, PyACLScalar, ACLFloatPolicy>;
        Registrar::template RegisterConversionIfRegistered<float8_e5m2>(type_num);
        Registrar::template RegisterConversionIfRegistered<float8_e4m3fn>(type_num);
        Registrar::template RegisterConversionIfRegistered<float8_e8m0>(type_num);
        Registrar::template RegisterConversionIfRegistered<bfloat16>(type_num);
        Registrar::template RegisterConversionIfRegistered<float6_e2m3fn>(type_num);
        Registrar::template RegisterConversionIfRegistered<float6_e3m2fn>(type_num);
        Registrar::template RegisterConversionIfRegistered<float4_e2m1fn>(type_num);
        Registrar::template RegisterConversionIfRegistered<float4_e1m2fn>(type_num);
    }
};

template <typename T>
using FloatTypeRegistrar = ACLTypeRegistrar<T, ACLFloatManager, PyACLScalar, ACLFloatPolicy>;

template <typename T>
PyObject* ACLFloatManager<T>::type_ptr = nullptr;

template <typename T>
int ACLFloatManager<T>::npy_type = NPY_NOTYPE;

template <typename T>
PyArray_ArrFuncs ACLFloatManager<T>::arr_funcs = {};

template <typename T>
PyArray_DescrProto ACLFloatManager<T>::npy_descr_proto = {};

template <typename T>
PyArray_Descr* ACLFloatManager<T>::npy_descr = nullptr;

#define EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(T) \
    template <>                                   \
    PyObject* ACLFloatManager<T>::type_ptr = nullptr; \
    template <>                                       \
    int ACLFloatManager<T>::npy_type = NPY_NOTYPE;     \
    template <>                                       \
    PyArray_ArrFuncs ACLFloatManager<T>::arr_funcs = {}; \
    template <>                                       \
    PyArray_Descr* ACLFloatManager<T>::npy_descr = nullptr;

template <typename T>
PyObject* GetACLFloatTypeObject();

}  // namespace dtypes
}  // namespace asnumpy

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
struct ACLIntManager {
    ASNUMPY_ACL_MANAGER_COMMON_MEMBERS
    static int Dtype() { return npy_type; }
};

template <typename T>
struct PyACLIntScalar {
    PyObject_HEAD;
    T value;
};

struct ACLIntPolicy {
    template <typename T>
    static void InitZero(T& v) {
        v = T(0);
    }

    template <typename T>
    static PyObject* GetItem(void* data) {
        T val = *static_cast<T*>(data);
        int64_t int_val = static_cast<int64_t>(val);
        return PyLong_FromLongLong(int_val);
    }

    template <typename T>
    static int SetItem(PyObject* obj, void* data) {
        int64_t val;
        if (PyLong_Check(obj)) {
            val = PyLong_AsLongLong(obj);
        } else if (PyFloat_Check(obj)) {
            val = static_cast<int64_t>(PyFloat_AsDouble(obj));
        } else {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int type");
            return -1;
        }
        *static_cast<T*>(data) = T(val);
        return 0;
    }

    template <typename T>
    static int ScalarInit(PyObject* obj, T& out) {
        if (PyLong_Check(obj)) {
            out = T(static_cast<int64_t>(PyLong_AsLongLong(obj)));
            return 0;
        }
        if (PyFloat_Check(obj)) {
            out = T(static_cast<int64_t>(PyFloat_AsDouble(obj)));
            return 0;
        }
        PyErr_SetString(PyExc_TypeError, "Cannot convert to ACL int type");
        return -1;
    }

    template <typename T, typename Scalar>
    static PyObject* ScalarRepr(Scalar* self) {
        int64_t val = static_cast<int64_t>(self->value);
        return PyUnicode_FromFormat("%s(%lld)", TypeDescriptor<T>::kTypeName, val);
    }

    template <typename T, typename Scalar>
    static PyObject* ScalarStr(Scalar* self) {
        int64_t val = static_cast<int64_t>(self->value);
        return PyUnicode_FromFormat("%lld", val);
    }

    template <typename T>
    static PyObject* ScalarRepr(PyACLIntScalar<T>* self) {
        return ScalarRepr<T, PyACLIntScalar<T>>(self);
    }

    template <typename T>
    static PyObject* ScalarStr(PyACLIntScalar<T>* self) {
        return ScalarStr<T, PyACLIntScalar<T>>(self);
    }

    template <typename U, typename T>
    static U ConvertToPeer(const T& v) {
        return U(static_cast<int64_t>(v));
    }

    template <typename U, typename T>
    static T ConvertFromPeer(const U& v) {
        return T(static_cast<int64_t>(v));
    }

    template <typename T, template <typename, template <typename> class, template <typename> class, class> class RegistrarT>
    static void RegisterBuiltins(int type_num) {
        using Registrar = RegistrarT<T, ACLIntManager, PyACLIntScalar, ACLIntPolicy>;
        PyArray_RegisterCastFunc(PyArray_DescrFromType(type_num), NPY_INT32,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(
                                     Registrar::template cast_to_builtin<int32_t>));
        PyArray_RegisterCastFunc(PyArray_DescrFromType(NPY_INT32), type_num,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(
                                     Registrar::template cast_from_builtin<int32_t>));

        PyArray_RegisterCastFunc(PyArray_DescrFromType(type_num), NPY_INT64,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(
                                     Registrar::template cast_to_builtin<int64_t>));
        PyArray_RegisterCastFunc(PyArray_DescrFromType(NPY_INT64), type_num,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(
                                     Registrar::template cast_from_builtin<int64_t>));
    }

    template <typename T, template <typename, template <typename> class, template <typename> class, class> class RegistrarT>
    static void RegisterPeers(int type_num) {
        using Registrar = RegistrarT<T, ACLIntManager, PyACLIntScalar, ACLIntPolicy>;
        Registrar::template RegisterConversionIfRegistered<int4>(type_num);
        Registrar::template RegisterConversionIfRegistered<uint1>(type_num);
    }
};

template <typename T>
using IntTypeRegistrar = ACLTypeRegistrar<T, ACLIntManager, PyACLIntScalar, ACLIntPolicy>;

template <typename T>
PyObject* ACLIntManager<T>::type_ptr = nullptr;

template <typename T>
int ACLIntManager<T>::npy_type = NPY_NOTYPE;

template <typename T>
PyArray_ArrFuncs ACLIntManager<T>::arr_funcs = {};

template <typename T>
PyArray_DescrProto ACLIntManager<T>::npy_descr_proto = {};

template <typename T>
PyArray_Descr* ACLIntManager<T>::npy_descr = nullptr;

#define EXPLICIT_INSTANTIATE_ACL_INT_MANAGER(T) \
    template <>                                 \
    PyObject* ACLIntManager<T>::type_ptr = nullptr; \
    template <>                                     \
    int ACLIntManager<T>::npy_type = NPY_NOTYPE;     \
    template <>                                     \
    PyArray_ArrFuncs ACLIntManager<T>::arr_funcs = {}; \
    template <>                                     \
    PyArray_Descr* ACLIntManager<T>::npy_descr = nullptr;

template <typename T>
PyObject* GetACLIntTypeObject();

}  // namespace dtypes
}  // namespace asnumpy


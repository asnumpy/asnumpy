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

#define ASNUMPY_ACL_MANAGER_COMMON_MEMBERS                \
    static PyObject* type_ptr;      /* Python类型对象指针 */  \
    static int npy_type;            /* NumPy类型ID */          \
    static PyType_Spec type_spec;                               \
    static PyType_Slot type_slots[];                             \
    static PyArray_ArrFuncs arr_funcs;  /* NumPy数组操作函数 */   \
    static PyArray_DescrProto npy_descr_proto; /* NumPy 2.x */    \
    static PyArray_Descr* npy_descr;    /* 注册后有效 */


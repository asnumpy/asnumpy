# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import numpy
import pytest
from asnumpy import testing


# ========== 辅助函数 ==========


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. 基础三角函数测试 (Sin, Cos, Tan) ==========


# 对于 sin/cos/tan，float16 目前在 C++ 绑定层可能存在映射问题
@pytest.mark.xfail(condition=True, reason="Bug: aclDataType mapping for float16 is missing in C++ core")
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_trig_float16_xfail(xp, dtype):
    data = [0.0, numpy.pi / 4]
    a = _create_array(xp, data, dtype)
    return xp.sin(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sin_basic(xp, dtype):
    data = [0.0, numpy.pi / 2, numpy.pi]
    a = _create_array(xp, data, dtype)
    return xp.sin(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_cos_basic(xp, dtype):
    data = [0.0, numpy.pi / 2, numpy.pi]
    a = _create_array(xp, data, dtype)
    return xp.cos(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_tan_basic(xp, dtype):
    """测试 tan (使用 allclose 处理 NPU 硬件计算位差)"""
    data = [0.0, numpy.pi / 4, numpy.pi / 3]
    a = _create_array(xp, data, dtype)
    return xp.tan(a)


# --- 针对非浮点类型的 xfail 标注 (遵循 Logic 风格) ---


@pytest.mark.xfail(reason="Bug: aclnnSin does not support Int32, asnumpy missing auto-cast")
@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_sin_int(xp, dtype):
    data = [1, 2, 3]
    a = _create_array(xp, data, dtype)
    return xp.sin(a)


@pytest.mark.xfail(reason="Bug: aclnnCos does not support Bool, asnumpy missing auto-cast")
@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_array_equal()
def test_cos_bool(xp, dtype):
    data = [True, False]
    a = _create_array(xp, data, dtype)
    return xp.cos(a)


# ========== 2. 反三角函数测试 (Arcsin, Arccos, Arctan) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arcsin_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.arcsin(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arccos_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.arccos(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arctan_basic(xp, dtype):
    data = [-10.0, 0.0, 10.0]
    a = _create_array(xp, data, dtype)
    return xp.arctan(a)


@pytest.mark.xfail(reason="Bug: aclnnArccos throws RuntimeError on x > 1")
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_arccos_out_of_domain(xp, dtype):
    """测试反余弦越界行为"""
    data = [2.0]
    a = _create_array(xp, data, dtype)
    return xp.arccos(a)


# ========== 3. 双输入反正切 (Arctan2) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arctan2_basic(xp, dtype):
    y = [1.0, 0.0, -1.0]
    x = [1.0, 1.0, 1.0]
    t_y = _create_array(xp, y, dtype)
    t_x = _create_array(xp, x, dtype)
    return xp.arctan2(t_y, t_x)


@pytest.mark.xfail(reason="Bug: aclnnArctan2 does not support Int32 input")
@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_arctan2_int(xp, dtype):
    y = [1, 0]
    x = [1, 1]
    t_y = _create_array(xp, y, dtype)
    t_x = _create_array(xp, x, dtype)
    return xp.arctan2(t_y, t_x)

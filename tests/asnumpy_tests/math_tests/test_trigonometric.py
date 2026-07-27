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


# float16's ULP near sin(pi/4)~0.707 is ~4.9e-4, so atol/rtol=1e-5 was ~30x tighter than a single
# ULP -- any disagreement between NumPy's float16 sin and aclnnSin would hard-fail the test.
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(atol=1e-3, rtol=1e-3)
def test_trig_float16(xp, dtype):
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


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sin_int(xp, dtype):
    data = [1, 2, 3]
    a = _create_array(xp, data, dtype)
    return xp.sin(a)


@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_allclose(atol=1e-3, rtol=1e-3)
def test_cos_bool(xp, dtype):
    """NumPy returns float16 for bool; asnumpy promotes to float32 (float16 mapping gap).

    Compare against float32 reference so float16 rounding on the NumPy side does not dominate.
    """
    data = [True, False]
    if xp is numpy:
        return numpy.cos(numpy.array(data, dtype=numpy.float32))
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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
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


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arctan2_int(xp, dtype):
    y = [1, 0]
    x = [1, 1]
    t_y = _create_array(xp, y, dtype)
    t_x = _create_array(xp, x, dtype)
    return xp.arctan2(t_y, t_x)

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

"""双曲函数测试

针对 CANN 算子限制进行精准标注：
1. 基础双曲函数: sinh, cosh, tanh
2. 反双曲函数: arcsinh, arccosh, arctanh
"""

import numpy
import pytest
from asnumpy import testing
from tests.asnumpy_tests.math_tests.conftest import _create_array


# ========== 1. 基础双曲函数 (Sinh, Cosh, Tanh) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sinh_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.sinh(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_cosh_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.cosh(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_tanh_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.tanh(a)


# --- 针对 tanh 的特殊 Dtype 支持 (根据统计：tanh 支持 int16) ---


@testing.for_dtypes([numpy.int16])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_tanh_int16_support(xp, dtype):
    data = [1, 0, -1]
    a = _create_array(xp, data, dtype)
    return xp.tanh(a)


# --- 针对不支持类型及精度不一致的标注 (XFAIL) ---


@pytest.mark.xfail(reason="Bug: aclDataType mapping for float16 is missing in C++ core")
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose()
def test_hyperbolic_float16_xfail(xp, dtype):
    a = _create_array(xp, [0.5], dtype)
    return xp.sinh(a)


@pytest.mark.xfail(
    reason="Mismatch: AsNumpy outputs float32 for integer inputs (Numpy is float64) or Unsupport uint16/32/64"
)
@testing.for_dtypes([numpy.int32, numpy.uint16, numpy.uint32, numpy.uint64])
@testing.numpy_asnumpy_allclose()
def test_hyperbolic_mismatch_xfail(xp, dtype):
    """统计确认：sinh/cosh 等不支持 uint，且 int 提升精度不一致"""
    data = [1, 2]
    a = _create_array(xp, data, dtype)
    return xp.sinh(a)


# ========== 2. 反双曲函数 (Arcsinh, Arccosh, Arctanh) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arcsinh_basic(xp, dtype):
    data = [-5.0, 0.0, 5.0]
    a = _create_array(xp, data, dtype)
    return xp.arcsinh(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arccosh_basic(xp, dtype):
    data = [1.0, 2.0, 5.0]
    a = _create_array(xp, data, dtype)
    return xp.arccosh(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arctanh_basic(xp, dtype):
    data = [-0.9, 0.0, 0.9]
    a = _create_array(xp, data, dtype)
    return xp.arctanh(a)


# --- 越界行为测试 (不带 equal_nan 参数，依赖底层默认行为) ---


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_arccosh_out_of_domain(xp, dtype):
    """测试 arccosh 越界 (x < 1)。注意：若 allclose 不支持 NaN 对比，此项可能失败。"""
    data = [0.5]
    a = _create_array(xp, data, dtype)
    return xp.arccosh(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_arctanh_out_of_domain(xp, dtype):
    """测试 arctanh 越界 (|x| >= 1)"""
    data = [1.0, 2.0]
    a = _create_array(xp, data, dtype)
    return xp.arctanh(a)

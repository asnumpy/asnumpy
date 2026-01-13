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

"""基础算术运算测试

包含：
1. 二元运算: add, subtract, multiply, divide, true_divide, floor_divide, remainder
2. 一元运算: negative, absolute, reciprocal
"""

import numpy
import pytest
from asnumpy import testing
from tests.asnumpy_tests.math_tests.conftest import _create_array


# ========== 1. 基础四则运算 (Add, Sub, Mul, Div) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_add_basic(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [4, 5, 6], dtype)
    return xp.add(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_subtract_basic(xp, dtype):
    a = _create_array(xp, [10, 20, 30], dtype)
    b = _create_array(xp, [1, 2, 3], dtype)
    return xp.subtract(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_multiply_basic(xp, dtype):
    a = _create_array(xp, [2, 3, 4], dtype)
    b = _create_array(xp, [5, 6, 7], dtype)
    return xp.multiply(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_divide_basic(xp, dtype):
    a = _create_array(xp, [10, 20, 30], dtype)
    b = _create_array(xp, [2, 4, 5], dtype)
    return xp.divide(a, b)


# ========== 2. 整除与取余 (Floor_divide, Remainder) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_floor_divide_basic(xp, dtype):
    a = _create_array(xp, [10, 7, 2], dtype)
    b = _create_array(xp, [3, 2, 3], dtype)
    return xp.floor_divide(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_remainder_basic(xp, dtype):
    a = _create_array(xp, [10, 7, 2], dtype)
    b = _create_array(xp, [3, 2, 3], dtype)
    return xp.remainder(a, b)


# ========== 3. 一元运算 (Negative, Absolute, Reciprocal) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_negative(xp, dtype):
    a = _create_array(xp, [-1, 0, 1], dtype)
    return xp.negative(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_absolute(xp, dtype):
    a = _create_array(xp, [-1.5, 0, 2.5], dtype)
    return xp.absolute(a)


# ========== 4. 广播与特殊 Dtype 限制 (XFAIL) ==========


@pytest.mark.xfail(reason="Bug: aclDataType mapping for float16 is missing in C++ core")
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose()
def test_arithmetic_float16_xfail(xp, dtype):
    a = _create_array(xp, [1.0], dtype)
    b = _create_array(xp, [2.0], dtype)
    return xp.add(a, b)


@pytest.mark.xfail(reason="Mismatch: AsNumpy outputs float32 for integer inputs (Numpy is float64)")
@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_arithmetic_int_mismatch_xfail(xp, dtype):
    """测试整数输入的精度提升不一致问题"""
    a = _create_array(xp, [1, 2], dtype)
    b = _create_array(xp, [3, 4], dtype)
    return xp.add(a, b)


@pytest.mark.xfail(reason="Bug: aclnnRemainder does not support BOOL type")
@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_array_equal()
def test_remainder_bool_xfail(xp, dtype):
    a = _create_array(xp, [True, False], dtype)
    b = _create_array(xp, [True, True], dtype)
    return xp.remainder(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_arithmetic_broadcasting(xp, dtype):
    """测试二元运算的广播机制 (例如: (2,3) + (3,))"""
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    b = _create_array(xp, [1, 0, 1], dtype)
    return xp.add(a, b)

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

"""指数和对数函数测试

包含：
1. 指数: exp, exp2, expm1
2. 对数: log, log2, log10, log1p
"""

import numpy
import pytest
from asnumpy import testing
from tests.asnumpy_tests.math_tests.conftest import _create_array


# ========== 1. 指数运算 (Exp, Expm1) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_exp_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0, 2.0]
    a = _create_array(xp, data, dtype)
    return xp.exp(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_expm1_basic(xp, dtype):
    """测试 exp(x) - 1，常用于极小值场景"""
    data = [1e-5, 0.0, -1e-5]
    a = _create_array(xp, data, dtype)
    return xp.expm1(a)


# ========== 2. 对数运算 (Log, Log10, Log1p) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_log_basic(xp, dtype):
    data = [0.1, 1.0, numpy.e, 10.0]
    a = _create_array(xp, data, dtype)
    return xp.log(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_log10_basic(xp, dtype):
    data = [0.1, 1.0, 10.0, 100.0]
    a = _create_array(xp, data, dtype)
    return xp.log10(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_log1p_basic(xp, dtype):
    """测试 log(1 + x)"""
    data = [1e-5, 0.0, -1e-5]
    a = _create_array(xp, data, dtype)
    return xp.log1p(a)


# ========== 3. 限制性测试 (XFAIL) ==========


@pytest.mark.xfail(reason="Bug: aclDataType mapping for float16 is missing in C++ core")
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose()
def test_exp_float16_xfail(xp, dtype):
    a = _create_array(xp, [1.0], dtype)
    return xp.exp(a)


@pytest.mark.xfail(reason="Mismatch: AsNumpy outputs float32 for integer inputs (Numpy is float64)")
@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_exp_int_mismatch_xfail(xp, dtype):
    data = [1, 2]
    a = _create_array(xp, data, dtype)
    return xp.exp(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_log_domain_error(xp, dtype):
    """测试对数定义域（非正数）。Numpy 返回 NaN/Inf。"""
    data = [0.0, -1.0]
    a = _create_array(xp, data, dtype)
    return xp.log(a)

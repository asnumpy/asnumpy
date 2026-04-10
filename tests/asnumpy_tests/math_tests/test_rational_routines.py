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

"""有理数算子测试

针对已记录的 Gcd/Lcm 算子限制进行精准标注：
1. GCD/LCM: int32, int64 基础功能。
2. 异常记录: int8/int16 触发 RuntimeError 161002。
"""

import numpy
import pytest
from asnumpy import testing


# ========== 辅助函数 ==========


def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. 基础公约数/公倍数 (Int32, Int64) ==========


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_gcd_basic(xp, dtype):
    """测试 int32/int64 下的正常 GCD 运算"""
    x1 = _create_array(xp, [12, 20, 32], dtype)
    x2 = _create_array(xp, [18, 24, 16], dtype)
    return xp.gcd(x1, x2)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_lcm_basic(xp, dtype):
    """测试 int32/int64 下的正常 LCM 运算"""
    x1 = _create_array(xp, [12, 20, 32], dtype)
    x2 = _create_array(xp, [18, 24, 16], dtype)
    return xp.lcm(x1, x2)


# ========== 2. 针对低精度整数的 Bug 记录 (XFAIL) ==========


@pytest.mark.xfail(reason="Bug: aclnnGcd/Lcm throws RuntimeError 161002 (get workspace size failed) for Int8/Int16")
@testing.for_dtypes([numpy.int8, numpy.int16])
@testing.numpy_asnumpy_array_equal()
def test_rational_low_precision_int_xfail(xp, dtype):
    """
    记录：当测试用例使用 int8 或 int16 等低精度整数类型时，
    底层报错 RuntimeError 161002。
    """
    x1 = _create_array(xp, [4, 8], dtype)
    x2 = _create_array(xp, [6, 12], dtype)
    return xp.gcd(x1, x2)


# ========== 3. 其他硬件与行为限制 (XFAIL) ==========


@pytest.mark.xfail(reason="Bug: aclnnGcd/Lcm does not support Float types")
@testing.for_dtypes([numpy.float32, numpy.float16])
def test_rational_float_xfail(xp, dtype):
    x1 = _create_array(xp, [12.0], dtype)
    x2 = _create_array(xp, [18.0], dtype)
    return xp.gcd(x1, x2)


@pytest.mark.xfail(reason="Behavior Mismatch: Handling of negative inputs in GCD between NPU and NumPy")
@testing.for_dtypes([numpy.int32])
def test_gcd_negative_behavior_xfail(xp, dtype):
    """
    记录：验证 NPU 是否遵循 NumPy 规范（GCD 结果始终为正）。
    NumPy: gcd(-12, 18) -> 6
    """
    x1 = _create_array(xp, [-12], dtype)
    x2 = _create_array(xp, [18], dtype)
    return xp.gcd(x1, x2)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_rational_broadcasting(xp, dtype):
    """测试 GCD 的广播支持情况"""
    x1 = _create_array(xp, [[10, 20], [30, 40]], dtype)
    x2 = _create_array(xp, [5, 10], dtype)
    return xp.gcd(x1, x2)

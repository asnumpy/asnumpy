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

"""复数运算算子测试

当前阶段仅测试：real
记录问题：代码强制将所有复数（包括 complex128）的输出类型硬编码为 ACL_FLOAT (float32)，
导致 complex128 -> float64 的精度下降。
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


# ========== 1. 实部提取 (Real) ==========


@testing.for_dtypes([numpy.complex64])
@testing.numpy_asnumpy_allclose()
def test_real_complex64_basic(xp, dtype):
    """测试 complex64 -> float32 链路"""
    data = [1.0 + 2.0j, -3.5 + 4.5j, 0.0 + 0.0j]
    a = _create_array(xp, data, dtype)
    return xp.real(a)


@pytest.mark.xfail(reason="Mismatch: complex128 output is hardcoded to float32 (ACL_FLOAT) instead of float64.")
@testing.for_dtypes([numpy.complex128])
@testing.numpy_asnumpy_allclose()
def test_real_complex128_precision_xfail(xp, dtype):
    """
    记录：complex128 提取实部时，后端硬编码输出为 float32。
    预期：返回 float64 (保持高精度)
    实际：返回 float32 (精度受损)
    """
    # 构造一个需要 float64 精度才能精确表达的数值
    data = [1.23456789012345 + 0.5j]
    a = _create_array(xp, data, dtype)

    # 当 xp 为 asnumpy 时，由于后端硬编码，返回的将是截断后的 float32
    return xp.real(a)


@pytest.mark.xfail(reason="Bug: real() on non-complex dtypes might be unsupported in current asnumpy implementation")
@testing.for_dtypes([numpy.float32, numpy.int32])
def test_real_non_complex_xfail(xp, dtype):
    """NumPy 允许对实数调 real (返回自身)，验证 asnumpy 是否支持"""
    a = _create_array(xp, [1, 2], dtype)
    return xp.real(a)

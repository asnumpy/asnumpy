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

"""舍入算子测试

针对已记录的 CANN 算子限制进行精准标注：
1. around: 支持 float32/64, int32/64 (需显式传 decimals)
2. rint: 支持 float32/64, 整数类型存在 dtype 不一致
3. fix: 仅支持 float32
4. floor: 不支持 float16 和复数
5. ceil: 仅支持 float32/64
6. trunc: 仅支持 float32
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


# ========== 1. 基础兼容性测试 (Around) ==========


@testing.for_dtypes([numpy.float32, numpy.float64, numpy.int32, numpy.int64])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_around_basic(xp, dtype):
    """记录：around 支持 float32/64, int32/64
    修正：显式传入 decimals=0 以适配 C++ 绑定
    """
    data = [0.4, 0.5, 0.6, 1.5, 2.5]
    a = _create_array(xp, data, dtype)

    if xp is numpy:
        return xp.around(a, decimals=0)
    # 适配当前 AsNumpy 强制要求 decimals 的接口
    return xp.around(a, 0)


# ========== 2. 存在 Dtype 行为差异的测试 (Rint, Floor) ==========


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose()
def test_rint_float_basic(xp, dtype):
    """记录：rint 支持 float32/64"""
    data = [-1.7, -1.5, 0.2, 1.5, 1.7]
    a = _create_array(xp, data, dtype)
    return xp.rint(a)


@pytest.mark.xfail(reason="Mismatch: rint on int32/64 results in inconsistent output dtype compared to Numpy")
@testing.for_dtypes([numpy.int32, numpy.int64])
def test_rint_int_mismatch_xfail(xp, dtype):
    """记录：rint 对整数类型的输出 dtype 与 numpy 不一致"""
    a = _create_array(xp, [1, 2], dtype)
    return xp.rint(a)


@testing.for_dtypes([numpy.float32, numpy.float64, numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_floor_basic(xp, dtype):
    """记录：floor 支持常见类型，但不支持 float16 和复数"""
    data = [-1.7, 0.2, 1.5]
    a = _create_array(xp, data, dtype)
    return xp.floor(a)


@pytest.mark.xfail(reason="Bug: aclnnFloor does not support float16 and Complex types")
@testing.for_dtypes([numpy.float16, numpy.complex64])
def test_floor_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1.5], dtype)
    return xp.floor(a)


# ========== 3. 严格限制 Dtype 的算子 (Fix, Ceil, Trunc) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_fix_basic(xp, dtype):
    """记录：fix 仅支持 float32"""
    data = [-1.7, 0.2, 1.5]
    a = _create_array(xp, data, dtype)
    return xp.fix(a)


@pytest.mark.xfail(reason="Bug: aclnnFix only supports float32")
@testing.for_dtypes([numpy.float64, numpy.int32])
def test_fix_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1.5], dtype)
    return xp.fix(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose()
def test_ceil_basic(xp, dtype):
    """记录：ceil 仅支持 float32, float64"""
    data = [-1.7, 0.2, 1.5]
    a = _create_array(xp, data, dtype)
    return xp.ceil(a)


@pytest.mark.xfail(reason="Bug: aclnnCeil only supports float32/64")
@testing.for_dtypes([numpy.int32, numpy.int64])
def test_ceil_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1], dtype)
    return xp.ceil(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_trunc_basic(xp, dtype):
    """记录：trunc 仅支持 float32"""
    data = [-1.7, 0.2, 1.5]
    a = _create_array(xp, data, dtype)
    return xp.trunc(a)


@pytest.mark.xfail(reason="Bug: aclnnTrunc only supports float32")
@testing.for_dtypes([numpy.float64, numpy.int32])
def test_trunc_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1.5], dtype)
    return xp.trunc(a)

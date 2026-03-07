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

"""杂项数学算子测试

针对已记录的 CANN 算子限制和 AsNumpy 行为不一致进行精准标注。
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


# ========== 1. 绝对值与符号 (Absolute, Fabs, Sign) ==========


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_absolute_basic(xp, dtype):
    data = [-1.5, 0, 2.5]
    a = _create_array(xp, data, dtype)
    return xp.absolute(a)


@pytest.mark.xfail(reason="Bug: aclnnAbs/Fabs does not support float16/uint/complex")
@testing.for_dtypes([numpy.float16, numpy.uint16, numpy.uint32, numpy.complex64])
def test_absolute_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1], dtype)
    return xp.absolute(a)


@pytest.mark.xfail(reason="Mismatch: AsNumpy returns original type for int, but Numpy returns float for fabs")
@testing.for_dtypes([numpy.int8, numpy.int32, numpy.uint8])
def test_fabs_dtype_mismatch_xfail(xp, dtype):
    a = _create_array(xp, [-1, 2], dtype)
    return xp.fabs(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_sign_basic(xp, dtype):
    data = [-5, 0, 5]
    a = _create_array(xp, data, dtype)
    return xp.sign(a)


@pytest.mark.xfail(reason="Bug: aclnnSign unsupported types (float16, int8/16, uints)")
@testing.for_dtypes([numpy.float16, numpy.int8, numpy.int16, numpy.uint8, numpy.uint16])
def test_sign_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1], dtype)
    return xp.sign(a)


# ========== 2. 平方与阶梯函数 (Square, Heaviside) ==========


@testing.for_dtypes([numpy.float64])  # 仅 float64 表现一致
@testing.numpy_asnumpy_allclose()
def test_square_float64(xp, dtype):
    a = _create_array(xp, [1.0, 2.0], dtype)
    return xp.square(a)


@pytest.mark.xfail(reason="Mismatch: AsNumpy square outputs float32 for integers, Numpy preserves dtype or promotes")
@testing.for_dtypes([numpy.int32, numpy.int64])
def test_square_int_mismatch_xfail(xp, dtype):
    a = _create_array(xp, [1, 2], dtype)
    return xp.square(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_heaviside_basic(xp, dtype):
    x1 = _create_array(xp, [-1.5, 0, 1.5], dtype)
    x2 = _create_array(xp, [0.5, 0.5, 0.5], dtype)
    return xp.heaviside(x1, x2)


# ========== 3. 最大/最小比较 (Maximum, Minimum) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_maximum_basic(xp, dtype):
    x1 = _create_array(xp, [1, 5, 2], dtype)
    x2 = _create_array(xp, [3, 4, 2], dtype)
    return xp.maximum(x1, x2)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_minimum_basic(xp, dtype):
    x1 = _create_array(xp, [1, 5, 2], dtype)
    x2 = _create_array(xp, [3, 4, 2], dtype)
    return xp.minimum(x1, x2)


@pytest.mark.xfail(reason="Bug: aclnnMaximum/Minimum does not support float16/uints/complex")
@testing.for_dtypes([numpy.float16, numpy.uint16, numpy.uint32, numpy.complex64])
def test_max_min_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1], dtype)
    return xp.maximum(a, a)


# ========== 4. 裁剪与填补 (Clip, Nan_to_num) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_clip_basic(xp, dtype):
    a = _create_array(xp, [1, 5, 10], dtype)
    return xp.clip(a, 3, 8)


@pytest.mark.xfail(reason="Mismatch: Clip forces non-float32 types to float32")
@testing.for_dtypes([numpy.int32, numpy.float64])
def test_clip_dtype_mismatch_xfail(xp, dtype):
    a = _create_array(xp, [1, 5, 10], dtype)
    return xp.clip(a, 3, 8)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_nan_to_num_basic(xp, dtype):
    """
    修正：显式传入参数以适配 C++ 绑定
    nan=0.0, posinf=max_float, neginf=min_float
    """
    data = [float('nan'), float('inf'), float('-inf'), 1.0]
    a = _create_array(xp, data, dtype)

    # 假设 AsNumpy 的 C++ 接口需要这几个参数，而没有默认值
    if xp is numpy:
        return xp.nan_to_num(a)

    # 获取 float32 的极大极小值
    finfo = numpy.finfo(numpy.float32)
    return xp.nan_to_num(a, nan=0.0, posinf=finfo.max, neginf=finfo.min)

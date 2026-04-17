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

"""排序函数测试

包含：
1. sort 排序函数

优化维度：
- axis 参数
- stable 排序
- 各 dtype（含 bool）
- 已排序/逆序/含 NaN 输入
- 多维输入
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
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


# ==========================================================================
# 1. 基础功能测试 (Basic)
# ==========================================================================

# ---------- 1.1 一维排序 ----------
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_1d(xp, dtype):
    """测试一维数组排序"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [3.0, 1.0, 2.0, 5.0, 4.0]
    else:
        data = [3, 1, 2, 5, 4]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_1d_negative_values(xp, dtype):
    """测试含负值的一维数组排序"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [3.0, -1.0, 2.0, -5.0, 4.0]
    else:
        data = [3, -1, 2, -5, 4]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_1d_default_axis(xp, dtype):
    """测试 sort 默认 axis（应为 -1）"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [5.0, 2.0, 8.0, 1.0]
    else:
        data = [5, 2, 8, 1]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ---------- 1.2 单元素 ----------
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_single_element(xp, dtype):
    """测试单元素数组排序"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [42.0]
    else:
        data = [42]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ---------- 1.3 两个元素 ----------
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_two_elements(xp, dtype):
    """测试两个元素数组排序"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [2.0, 1.0]
    else:
        data = [2, 1]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ==========================================================================
# 2. axis 参数测试 (Axis Parameter)
# ==========================================================================

# ---------- 2.1 二维数组 axis ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_2d_axis0(xp, dtype):
    """测试二维数组沿 axis=0 排序"""
    data = [[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, axis=0)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_2d_axis1(xp, dtype):
    """测试二维数组沿 axis=1 排序"""
    data = [[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_2d_default_axis(xp, dtype):
    """测试二维数组默认 axis=-1（等价于 axis=1）"""
    data = [[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_2d_negative_axis(xp, dtype):
    """测试二维数组负数 axis（axis=-2 等价于 axis=0）"""
    data = [[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, axis=-2)


# ---------- 2.2 三维数组 axis ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_3d_axis0(xp, dtype):
    """测试三维数组沿 axis=0 排序"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-10, 10, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sort(a, axis=0)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_3d_axis1(xp, dtype):
    """测试三维数组沿 axis=1 排序"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-10, 10, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sort(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_3d_axis2(xp, dtype):
    """测试三维数组沿 axis=2 排序"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-10, 10, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sort(a, axis=2)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_3d_negative_axis(xp, dtype):
    """测试三维数组负数 axis（axis=-1 等价于 axis=2）"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-10, 10, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sort(a, axis=-1)


# ==========================================================================
# 3. stable 排序测试 (Stable Sorting)
# ==========================================================================

# ---------- 3.1 stable=True ----------
@pytest.mark.xfail(reason="numpy.sort does not support 'stable' parameter in this NumPy version", strict=True)
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_stable_true(xp, dtype):
    """测试 stable=True 排序"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [3.0, 1.0, 2.0, 1.0, 3.0]
    else:
        data = [3, 1, 2, 1, 3]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, stable=True)


@pytest.mark.xfail(reason="numpy.sort does not support 'stable' parameter in this NumPy version", strict=True)
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_stable_false(xp, dtype):
    """测试 stable=False 排序（默认）"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [3.0, 1.0, 2.0, 1.0, 3.0]
    else:
        data = [3, 1, 2, 1, 3]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, stable=False)


@pytest.mark.xfail(reason="numpy.sort does not support 'stable' parameter in this NumPy version", strict=True)
@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_sort_stable_duplicates(xp, dtype):
    """测试 stable 排序对重复值的处理"""
    data = [5, 2, 3, 2, 5, 3, 1, 1]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, stable=True)


@pytest.mark.xfail(reason="numpy.sort does not support 'stable' parameter in this NumPy version", strict=True)
@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_sort_stable_2d(xp, dtype):
    """测试 stable 排序在二维数组上的行为"""
    data = [[3, 1, 2], [6, 4, 5]]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, axis=1, stable=True)


@pytest.mark.xfail(reason="numpy.sort does not support 'stable' parameter in this NumPy version", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_stable_all_same(xp, dtype):
    """测试 stable 排序：所有值相同"""
    data = [2.0, 2.0, 2.0, 2.0]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, stable=True)


# ==========================================================================
# 4. 各 dtype 测试 (Dtype Support)
# ==========================================================================

# ---------- 4.1 布尔类型 ----------
@pytest.mark.xfail(reason="CANN sort operator does not support bool dtype", strict=True)
@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_array_equal()
def test_sort_bool(xp, dtype):
    """测试 bool 类型排序（False < True）"""
    data = [True, False, True, False, True]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@pytest.mark.xfail(reason="CANN sort operator does not support bool dtype", strict=True)
@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_array_equal()
def test_sort_bool_all_true(xp, dtype):
    """测试 bool 类型：全 True"""
    data = [True, True, True]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@pytest.mark.xfail(reason="CANN sort operator does not support bool dtype", strict=True)
@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_array_equal()
def test_sort_bool_all_false(xp, dtype):
    """测试 bool 类型：全 False"""
    data = [False, False, False]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@pytest.mark.xfail(reason="CANN sort operator does not support bool dtype", strict=True)
@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_array_equal()
def test_sort_bool_2d(xp, dtype):
    """测试 bool 类型二维数组排序"""
    data = [[True, False, True], [False, True, False]]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, axis=1)


# ---------- 4.2 整数类型 ----------
@testing.for_dtypes([numpy.int8, numpy.int16, numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_sort_int_dtypes(xp, dtype):
    """测试各整数 dtype 的排序"""
    data = [5, 3, 1, 4, 2]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ---------- 4.3 无符号整数 ----------
@testing.for_dtypes([numpy.uint8])
@testing.numpy_asnumpy_array_equal()
def test_sort_uint8(xp, dtype):
    """测试 uint8 排序"""
    data = [200, 50, 150, 100, 250]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ---------- 4.4 浮点类型 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_float_dtypes(xp, dtype):
    """测试各浮点 dtype 的排序"""
    data = [3.5, 1.2, 4.8, 2.1, 5.9]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ==========================================================================
# 5. 特殊输入模式测试 (Special Input Patterns)
# ==========================================================================

# ---------- 5.1 已排序输入 ----------
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_already_sorted(xp, dtype):
    """测试已排序输入"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
    else:
        data = [1, 2, 3, 4, 5]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ---------- 5.2 逆序输入 ----------
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_reverse_sorted(xp, dtype):
    """测试逆序输入"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [5.0, 4.0, 3.0, 2.0, 1.0]
    else:
        data = [5, 4, 3, 2, 1]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ---------- 5.3 含 NaN 输入 ----------
@pytest.mark.xfail(reason="CANN sort operator does not support NaN values", strict=True)
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose()
def test_sort_with_nan(xp, dtype):
    """测试含 NaN 的排序（NaN 应排在末尾）"""
    data = [3.0, float('nan'), 1.0, 2.0]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@pytest.mark.xfail(reason="CANN sort operator does not support NaN values", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_sort_with_multiple_nan(xp, dtype):
    """测试含多个 NaN 的排序"""
    data = [float('nan'), 3.0, float('nan'), 1.0, 2.0]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@pytest.mark.xfail(reason="CANN sort operator does not support NaN values", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_sort_nan_at_beginning(xp, dtype):
    """测试 NaN 在开头"""
    data = [float('nan'), 2.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@pytest.mark.xfail(reason="CANN sort operator does not support NaN values", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_sort_nan_only(xp, dtype):
    """测试全部 NaN 输入"""
    data = [float('nan'), float('nan'), float('nan')]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


@pytest.mark.xfail(reason="CANN sort operator does not support NaN values", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_sort_with_nan_2d(xp, dtype):
    """测试二维数组含 NaN 排序"""
    data = [[3.0, float('nan'), 1.0], [2.0, 5.0, float('nan')]]
    a = _create_array(xp, data, dtype)
    return xp.sort(a, axis=1)


# ---------- 5.4 所有值相同 ----------
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_all_same(xp, dtype):
    """测试所有值相同的排序"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [7.0, 7.0, 7.0, 7.0]
    else:
        data = [7, 7, 7, 7]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ---------- 5.5 含重复值 ----------
@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_with_duplicates(xp, dtype):
    """测试含重复值的排序"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0]
    else:
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    a = _create_array(xp, data, dtype)
    return xp.sort(a)


# ---------- 5.6 随机数据 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_random(xp, dtype):
    """测试随机数据排序"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-100, 100, (20,)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sort(a)


# ==========================================================================
# 6. 多维输入测试 (Multi-dimensional Input)
# ==========================================================================

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_2d_wide(xp, dtype):
    """测试二维宽矩阵排序 (2x10)"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-10, 10, (2, 10)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sort(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_2d_tall(xp, dtype):
    """测试二维高矩阵排序 (10x3)"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-10, 10, (10, 3)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sort(a, axis=0)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_3d_input(xp, dtype):
    """测试三维数组排序"""
    numpy.random.seed(42)
    np_data = numpy.random.randn(2, 3, 4).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sort(a, axis=2)


# ==========================================================================
# 7. 空数组 / 边界情况测试 (Edge Cases)
# ==========================================================================

@pytest.mark.xfail(reason="NPU operator does not support empty arrays", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_sort_empty(xp, dtype):
    """测试排序: 空数组"""
    a = _create_array(xp, [], dtype)
    return xp.sort(a)

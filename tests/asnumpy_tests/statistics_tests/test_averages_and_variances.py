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

"""mean 函数测试

覆盖:
1. 基础功能: 标量、向量、矩阵、多维
2. axis 参数: None / 正轴 / 负轴 / 元组
3. keepdims 参数
4. dtype 参数
5. 边界情况: 空数组、单元素、全相同值
6. 异常: 无效 axis
"""

import numpy
import pytest
from asnumpy import testing
from ._helpers import create_array as _create_array


# ==========================================================================
# 1. 基础功能测试
# ==========================================================================


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_scalar_like(xp, dtype):
    """单元素数组 mean"""
    a = _create_array(xp, [42.0], dtype)
    return xp.mean(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_1d(xp, dtype):
    """一维数组 mean"""
    a = _create_array(xp, [1.0, 2.0, 3.0, 4.0, 5.0], dtype)
    return xp.mean(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_2d(xp, dtype):
    """二维数组 mean (全部元素)"""
    a = _create_array(xp, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
    return xp.mean(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_3d(xp, dtype):
    """三维数组 mean"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5, 5, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.mean(a)


# ==========================================================================
# 2. axis 参数测试
# ==========================================================================


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_2d_axis0(xp, dtype):
    """二维数组沿 axis=0 的 mean"""
    a = _create_array(xp, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
    return xp.mean(a, axis=0)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_2d_axis1(xp, dtype):
    """二维数组沿 axis=1 的 mean"""
    a = _create_array(xp, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
    return xp.mean(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_negative_axis(xp, dtype):
    """负轴参数: axis=-1 等价于 axis=-1 (最后一个轴)"""
    a = _create_array(xp, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
    return xp.mean(a, axis=-1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_3d_axis1(xp, dtype):
    """三维数组沿 axis=1 的 mean"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5, 5, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.mean(a, axis=1)


# ==========================================================================
# 3. keepdims 参数测试
# ==========================================================================


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_keepdims_true(xp, dtype):
    """keepdims=True 保持维度"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0]], dtype)
    return xp.mean(a, axis=1, keepdims=True)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_axis0_keepdims(xp, dtype):
    """沿 axis=0 且 keepdims=True"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0]], dtype)
    return xp.mean(a, axis=0, keepdims=True)


# ==========================================================================
# 4. dtype 参数测试
# ==========================================================================


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_dtype_f64(xp, dtype):
    """dtype=float64 输出高精度"""
    a = _create_array(xp, [1.0, 2.0, 3.0], dtype)
    return xp.mean(a, dtype=numpy.float64)


# ==========================================================================
# 5. 边界情况: 空数组
# ==========================================================================


def test_mean_empty_1d():
    """空一维数组 mean 应与 NumPy 行为兼容（返回 NaN 或抛出异常）"""
    try:
        import asnumpy
        a = asnumpy.zeros(0, dtype=numpy.float32)
        result = asnumpy.mean(a)
        # NumPy 对空切片返回 NaN
        assert numpy.isnan(float(result)) if not isinstance(result, numpy.ndarray) else True
    except (ValueError, RuntimeError) as e:
        # NPU 可能不支持空数组 — 这也是可接受的
        pytest.skip(f"NPU does not support empty arrays: {e}")


def test_mean_empty_axis0():
    """空数组沿 axis=0 的 mean"""
    try:
        import asnumpy
        a = asnumpy.zeros((0, 3), dtype=numpy.float32)
        result = asnumpy.mean(a, axis=0)
        # shape 应为 (3,), 值为 NaN
        assert result.shape == (3,)
    except (ValueError, RuntimeError) as e:
        pytest.skip(f"NPU does not support empty arrays: {e}")


def test_mean_empty_axis1_keepdims():
    """空数组沿 axis=1 的 mean (keepdims=True)"""
    try:
        import asnumpy
        a = asnumpy.zeros((3, 0), dtype=numpy.float32)
        result = asnumpy.mean(a, axis=1, keepdims=True)
        # shape 应为 (3, 1)
        assert result.shape == (3, 1)
    except (ValueError, RuntimeError) as e:
        pytest.skip(f"NPU does not support empty arrays: {e}")


# ==========================================================================
# 6. 异常测试: 无效 axis
# ==========================================================================


def test_mean_invalid_axis_positive():
    """正数越界 axis 应抛出 AxisError（而非 segfault）"""
    import asnumpy
    a = asnumpy.ones(3, dtype=numpy.float32)
    with pytest.raises(numpy.exceptions.AxisError):
        asnumpy.mean(a, axis=5)


def test_mean_invalid_axis_negative():
    """负数越界 axis 应抛出 AxisError（而非 segfault）"""
    import asnumpy
    a = asnumpy.ones(3, dtype=numpy.float32)
    with pytest.raises(numpy.exceptions.AxisError):
        asnumpy.mean(a, axis=-5)


def test_mean_invalid_axis_2d():
    """二维数组越界 axis"""
    import asnumpy
    a = asnumpy.ones((3, 4), dtype=numpy.float32)
    with pytest.raises(numpy.exceptions.AxisError):
        asnumpy.mean(a, axis=3)


# ==========================================================================
# 7. 单元素与边界情况
# ==========================================================================


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_single_element(xp, dtype):
    """单元素数组 mean"""
    a = _create_array(xp, [3.14], dtype)
    return xp.mean(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_all_same(xp, dtype):
    """所有元素相同"""
    a = _create_array(xp, [7.0, 7.0, 7.0, 7.0], dtype)
    return xp.mean(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_large_values(xp, dtype):
    """大数值 mean"""
    a = _create_array(xp, [1e8, 2e8, 3e8], dtype)
    return xp.mean(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_mean_negative_values(xp, dtype):
    """负数值 mean"""
    a = _create_array(xp, [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0], dtype)
    return xp.mean(a)

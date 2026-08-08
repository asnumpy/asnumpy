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

"""平均值与方差相关统计函数测试

包含：
1. 均值函数: mean

优化维度：
- axis / keepdims 参数
- dtype 参数行为
- 空数组输入
- NaN 传播行为
- 非法 axis
"""

import warnings

import numpy
import pytest


# ========== 辅助函数 ==========
def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


def _to_numpy(value):
    """辅助函数：将 asnumpy 结果转换为 NumPy 对象"""
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return value


def _assert_mean_allclose(
    data,
    axis=None,
    keepdims=False,
    dtype=None,
    rtol=1e-5,
    atol=1e-5,
    equal_nan=False,
    expect_empty_warning=False,
):
    """辅助函数：比较 numpy 和 asnumpy 的 mean 结果"""
    import asnumpy as ap

    np_data = numpy.array(data)
    ap_data = ap.ndarray.from_numpy(np_data)

    if expect_empty_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np_result = numpy.mean(np_data, axis=axis, keepdims=keepdims, dtype=dtype)
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            ap_result = ap.mean(ap_data, axis=axis, keepdims=keepdims, dtype=dtype)
    else:
        np_result = numpy.mean(np_data, axis=axis, keepdims=keepdims, dtype=dtype)
        ap_result = ap.mean(ap_data, axis=axis, keepdims=keepdims, dtype=dtype)

    assert isinstance(ap_result, ap.ndarray)
    actual = numpy.asarray(_to_numpy(ap_result))
    expected = numpy.asarray(np_result)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    numpy.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)


def _assert_mean_dtype(data, axis=None, keepdims=False, dtype=None):
    """辅助函数：比较 mean 的返回 dtype"""
    import asnumpy as ap

    np_data = numpy.array(data)
    ap_data = ap.ndarray.from_numpy(np_data)

    np_result = numpy.mean(np_data, axis=axis, keepdims=keepdims, dtype=dtype)
    ap_result = ap.mean(ap_data, axis=axis, keepdims=keepdims, dtype=dtype)

    assert numpy.asarray(_to_numpy(ap_result)).dtype == numpy.asarray(np_result).dtype


# ============================================================================
# 1. 均值函数测试 (Mean)
# ============================================================================


# ---------- 1.1 基础功能: axis ----------
def test_mean_basic_global_float32():
    """测试 mean: 全局均值"""
    data = numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32)
    _assert_mean_allclose(data)


def test_mean_basic_global_float32_dtype():
    """测试 mean: 全局均值返回 dtype 与 NumPy 一致"""
    data = numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32)
    _assert_mean_dtype(data)


def test_mean_axis0():
    """测试 mean: axis=0"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0)


def test_mean_axis0_dtype():
    """测试 mean: axis=0 返回 dtype 与 NumPy 一致"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_dtype(data, axis=0)


def test_mean_axis1():
    """测试 mean: axis=1"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=1)


def test_mean_negative_axis():
    """测试 mean: 负 axis"""
    data = numpy.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=-1)


@pytest.mark.parametrize(("axis", "keepdims"), [((0, 2), False), ((0, -1), True), ((), False)])
def test_mean_tuple_axes(axis, keepdims):
    """测试 mean: 多轴、负多轴及空轴元组均与 NumPy 一致"""
    data = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    _assert_mean_allclose(data, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize("input_dtype", [numpy.bool_, numpy.int32, numpy.uint64])
def test_mean_default_integer_dtype(input_dtype):
    """测试 mean: bool/整数输入默认提升并以 float64 累加"""
    data = numpy.array([0, 1, 3, 8], dtype=input_dtype)
    _assert_mean_allclose(data)


def test_mean_default_float16_uses_float32_accumulator():
    """测试 mean: float16 默认累加不会在可表示均值上溢出"""
    data = numpy.array([65504, 65504], dtype=numpy.float16)
    _assert_mean_allclose(data, rtol=0, atol=0)


def test_mean_default_integer_retains_float64_precision():
    """测试 mean: 默认整数归约不会偷偷降为 float32"""
    data = numpy.array([16777217, 16777217], dtype=numpy.int32)
    _assert_mean_allclose(data, rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype", [numpy.complex64, numpy.complex128])
def test_mean_complex_value_and_dtype(input_dtype):
    """测试 mean: complex 的实部、虚部与 dtype 均保留"""
    data = numpy.array([1 + 2j, 3 + 4j], dtype=input_dtype)
    _assert_mean_allclose(data, rtol=0, atol=0)


@pytest.mark.parametrize(
    "result_dtype",
    [
        numpy.bool_,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
    ],
)
def test_mean_explicit_integral_dtype_uses_float64_accumulator(result_dtype):
    """测试 mean: 显式整数结果遵循 CuPy，先以 float64 计算再转换"""
    import asnumpy as ap

    source = ap.ndarray.from_numpy(numpy.array([127, 127, 127], dtype=numpy.int8))
    actual = ap.mean(source, dtype=result_dtype).to_numpy()
    expected = numpy.array(127, dtype=result_dtype)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axis", [None, ()])
def test_mean_zero_dim(axis):
    """测试 mean: 标量全归约与空轴归约都返回零维设备数组"""
    data = numpy.array(7, dtype=numpy.int32)
    _assert_mean_allclose(data, axis=axis)


@pytest.mark.parametrize(
    ("data", "axis"),
    [(numpy.array(-0.0, dtype=numpy.float32), None), (numpy.array([-0.0]), ())],
)
def test_mean_canonicalizes_negative_zero(data, axis):
    """测试 mean: 真正执行算术，而非把 axis=() 降级成逐字节拷贝"""
    import asnumpy as ap

    result = ap.mean(ap.ndarray.from_numpy(data), axis=axis).to_numpy()
    assert not numpy.signbit(result).any()


@pytest.mark.parametrize("input_dtype", [numpy.complex64, numpy.complex128])
def test_mean_axis_empty_canonicalizes_complex_negative_zero(input_dtype):
    """测试 mean: axis=() 同时规范化复数实部和虚部的负零"""
    import asnumpy as ap

    data = numpy.array([complex(-0.0, -0.0)], dtype=input_dtype)
    result = ap.mean(ap.ndarray.from_numpy(data), axis=()).to_numpy()
    assert not numpy.signbit(result.real).any()
    assert not numpy.signbit(result.imag).any()


def test_mean_method_out_identity_value_and_dtype():
    """测试 ndarray.mean: out 保持身份并按 out dtype 写入"""
    import asnumpy as ap

    data = numpy.arange(24, dtype=numpy.int32).reshape(2, 3, 4)
    source = ap.ndarray.from_numpy(data)
    out = ap.ndarray.from_numpy(numpy.empty((3,), dtype=numpy.float32))
    result = source.mean(axis=(0, 2), out=out)
    assert result is out
    actual = result.to_numpy()
    expected = numpy.mean(data, axis=(0, 2), dtype=numpy.float64).astype(numpy.float32)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    numpy.testing.assert_allclose(actual, expected)


def test_mean_out_direct_alias_scalar_and_dtype_override():
    """测试 out: 直接写、输入别名、零维及 out dtype 覆盖结果 dtype"""
    from asnumpy._core import testing as core_testing

    import asnumpy as ap

    direct = ap.ndarray.from_numpy(numpy.arange(6, dtype=numpy.float32).reshape(2, 3))
    direct_out = ap.ndarray.from_numpy(numpy.empty((3,), dtype=numpy.float32))
    direct_address = core_testing._device_address(direct_out)
    assert direct.mean(axis=0, out=direct_out) is direct_out
    assert core_testing._device_address(direct_out) == direct_address
    numpy.testing.assert_array_equal(
        direct_out.to_numpy(), numpy.array([1.5, 2.5, 3.5], dtype=numpy.float32)
    )

    alias = ap.ndarray.from_numpy(numpy.array([-0.0, 2.0], dtype=numpy.float32))
    alias_address = core_testing._device_address(alias)
    assert alias.mean(axis=(), out=alias) is alias
    assert core_testing._device_address(alias) == alias_address
    numpy.testing.assert_array_equal(alias.to_numpy(), numpy.array([0.0, 2.0], dtype=numpy.float32))

    scalar = ap.ndarray.from_numpy(numpy.array(5, dtype=numpy.int32))
    scalar_out = ap.ndarray.from_numpy(numpy.empty((), dtype=numpy.float64))
    scalar_address = core_testing._device_address(scalar_out)
    assert scalar.mean(out=scalar_out) is scalar_out
    assert core_testing._device_address(scalar_out) == scalar_address
    numpy.testing.assert_array_equal(scalar_out.to_numpy(), numpy.array(5.0))

    narrow = ap.ndarray.from_numpy(numpy.array([1, 2, 4], dtype=numpy.int8))
    float_out = ap.ndarray.from_numpy(numpy.empty((), dtype=numpy.float64))
    float_address = core_testing._device_address(float_out)
    assert narrow.mean(dtype=numpy.int8, out=float_out) is float_out
    assert core_testing._device_address(float_out) == float_address
    numpy.testing.assert_allclose(float_out.to_numpy(), numpy.array(7.0 / 3.0), rtol=0, atol=0)


# ---------- 1.2 keepdims 参数 ----------
def test_mean_keepdims_axis0():
    """测试 mean: axis=0 且 keepdims=True"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0, keepdims=True)


def test_mean_keepdims_axis1():
    """测试 mean: axis=1 且 keepdims=True"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=1, keepdims=True)


def test_mean_keepdims_axis_none():
    """测试 mean: axis=None 且 keepdims=True"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, keepdims=True)


# ---------- 1.3 dtype 参数 ----------
def test_mean_dtype_axis0_float64():
    """测试 mean: axis=0 且 dtype=float64"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0, dtype=numpy.float64)


def test_mean_dtype_axis_none_from_int32_value():
    """测试 mean: int32 输入 axis=None 且 dtype=float64 的数值行为"""
    data = numpy.array([1, 2, 4], dtype=numpy.int32)
    _assert_mean_allclose(data, dtype=numpy.float64)


def test_mean_dtype_axis_none_from_int32_dtype():
    """测试 mean: int32 输入 axis=None 且 dtype=float64 的 dtype 一致性"""
    data = numpy.array([1, 2, 4], dtype=numpy.int32)
    _assert_mean_dtype(data, dtype=numpy.float64)


# ---------- 1.4 空数组输入 ----------
def test_mean_empty_1d():
    """测试 mean: 一维空数组"""
    data = numpy.array([], dtype=numpy.float32)
    _assert_mean_allclose(data, equal_nan=True, expect_empty_warning=True)


def test_mean_empty_axis0():
    """测试 mean: 空维度数组 axis=0"""
    data = numpy.zeros((0, 3), dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0, equal_nan=True, expect_empty_warning=True)


def test_mean_empty_axis1_keepdims():
    """测试 mean: 空维度数组 axis=1 且 keepdims=True"""
    data = numpy.zeros((2, 0), dtype=numpy.float32)
    _assert_mean_allclose(data, axis=1, keepdims=True, equal_nan=True, expect_empty_warning=True)


@pytest.mark.parametrize("input_dtype", [numpy.complex64, numpy.complex128])
def test_mean_empty_complex_returns_nan(input_dtype):
    """测试 mean: CANN 不支持的空 complex 归约由兼容层返回 complex NaN"""
    import asnumpy as ap

    source = ap.ndarray.from_numpy(numpy.empty((0,), dtype=input_dtype))
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        actual = ap.mean(source).to_numpy()
    assert actual.shape == ()
    assert actual.dtype == numpy.dtype(input_dtype)
    assert numpy.isnan(actual.real) and numpy.isnan(actual.imag)


# ---------- 1.5 NaN 传播行为 ----------
def test_mean_nan_global():
    """测试 mean: 全局归约应传播 NaN"""
    data = numpy.array([1.0, numpy.nan, 3.0], dtype=numpy.float32)
    _assert_mean_allclose(data, equal_nan=True)


def test_mean_nan_axis0():
    """测试 mean: axis=0 按切片传播 NaN"""
    data = numpy.array([[1.0, numpy.nan, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0, equal_nan=True)


def test_mean_nan_axis1_keepdims():
    """测试 mean: axis=1 且 keepdims=True 时传播 NaN"""
    data = numpy.array([[1.0, numpy.nan, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=1, keepdims=True, equal_nan=True)


# ---------- 1.6 非法 axis ----------
def test_mean_invalid_axis_positive():
    """测试 mean: 正向越界 axis 应抛出异常"""
    import asnumpy as ap

    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    a = ap.ndarray.from_numpy(data)
    with pytest.raises(numpy.exceptions.AxisError):
        ap.mean(a, axis=2)


def test_mean_invalid_axis_negative():
    """测试 mean: 负向越界 axis 应抛出异常"""
    import asnumpy as ap

    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    a = ap.ndarray.from_numpy(data)
    with pytest.raises(numpy.exceptions.AxisError):
        ap.mean(a, axis=-3)


@pytest.mark.parametrize("axis", [0, -1])
def test_mean_zero_dim_invalid_axis(axis):
    """测试 mean: 零维数组不接受显式整数 axis"""
    import asnumpy as ap

    scalar = ap.ndarray.from_numpy(numpy.array(1, dtype=numpy.float32))
    with pytest.raises(numpy.exceptions.AxisError):
        ap.mean(scalar, axis=axis)

# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
# *****************************************************************************

"""逻辑运算函数测试

包含：
1. 归约运算: all, any
2. 无穷/有限检查: isfinite, isinf, isneginf, isposinf
3. 逻辑运算: logical_and, logical_or, logical_not, logical_xor
4. 比较运算: greater, less, equal, not_equal 等
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

# ========== 1. 归约运算测试 (Reduction) ==========

@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_all_basic(xp, dtype):
    """测试 all (逻辑与归约)"""
    data = [True, True, False, True]
    a = _create_array(xp, data, dtype)
    return xp.all(a)

@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_any_basic(xp, dtype):
    """测试 any (逻辑或归约)"""
    data = [False, False, True, False]
    a = _create_array(xp, data, dtype)
    return xp.any(a)

@pytest.mark.xfail(reason="Bug: aclnnAll throws RuntimeError 161002 with axis argument")
@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_all_axis(xp, dtype):
    """测试带 axis 的 all"""
    data = [[True, False], [True, True]]
    a = _create_array(xp, data, dtype)
    return xp.all(a, axis=(0,))

@pytest.mark.xfail(reason="Bug: aclnnAny throws RuntimeError 161002 with axis argument")
@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_any_axis(xp, dtype):
    """测试带 axis 的 any"""
    data = [[False, False], [True, False]]
    a = _create_array(xp, data, dtype)
    return xp.any(a, axis=(1,))

# ========== 2. 无穷/有限检查测试 (Finite Checks) ==========

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isfinite(xp, dtype):
    data = [0.0, 1.0, float('inf'), float('-inf'), float('nan')]
    a = _create_array(xp, data, dtype)
    return xp.isfinite(a)

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isinf(xp, dtype):
    data = [0.0, 1.0, float('inf'), float('-inf'), float('nan')]
    a = _create_array(xp, data, dtype)
    return xp.isinf(a)

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isposinf(xp, dtype):
    data = [0.0, float('inf'), float('-inf')]
    a = _create_array(xp, data, dtype)
    return xp.isposinf(a)

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isneginf(xp, dtype):
    data = [0.0, float('inf'), float('-inf')]
    a = _create_array(xp, data, dtype)
    return xp.isneginf(a)

# ========== 3. 逻辑运算测试 (Logical Operators) ==========

@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_logical_and(xp, dtype):
    data1 = [True, False, True, False]
    data2 = [True, True, False, False]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_and(x1, x2)

@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_logical_or(xp, dtype):
    data1 = [True, False, True, False]
    data2 = [True, True, False, False]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_or(x1, x2)

@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_logical_xor(xp, dtype):
    data1 = [True, False, True, False]
    data2 = [True, True, False, False]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_xor(x1, x2)

@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_logical_not(xp, dtype):
    data = [True, False]
    x = _create_array(xp, data, dtype)
    return xp.logical_not(x)

# ========== 4. 比较运算测试 (Comparisons) ==========

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater(xp, dtype):
    numpy.random.seed(42)
    np_a = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    np_b = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.greater(a, b)

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater_equal(xp, dtype):
    numpy.random.seed(42)
    np_a = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    np_b = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.greater_equal(a, b)

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_less(xp, dtype):
    numpy.random.seed(42)
    np_a = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    np_b = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.less(a, b)

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_less_equal(xp, dtype):
    numpy.random.seed(42)
    np_a = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    np_b = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.less_equal(a, b)

@pytest.mark.xfail(reason="Bug: aclnnEqual throws RuntimeError 161002")
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_equal(xp, dtype):
    """测试 equal (已知失败)"""
    data_a = [1.0, 2.0, 3.0]
    data_b = [1.0, 2.0, 4.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.equal(a, b)

@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_not_equal(xp, dtype):
    data_a = [1.0, 2.0, 3.0]
    data_b = [1.0, 2.0, 4.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.not_equal(a, b)

@pytest.mark.xfail(reason="Bug: aclnnGtScalar returns wrong results (all True)")
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater_scalar(xp, dtype):
    """测试 greater_scalar (已知数值错误)"""
    data = [1.0, 2.0, 3.0]
    scalar = 2.0
    a = _create_array(xp, data, dtype)
    return xp.greater(a, scalar)

# ========== 5. 问题测试 (保持 xfail) ==========

@testing.suppress_warnings
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
@pytest.mark.xfail(reason="Bug: aclnnEqual throws RuntimeError 161002 with NaN")
def test_equal_with_nan(xp, dtype):
    """测试 NaN 的比较行为 (已知会崩溃)"""
    data_a = [float('nan'), 1.0]
    data_b = [float('nan'), 1.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.equal(a, b)
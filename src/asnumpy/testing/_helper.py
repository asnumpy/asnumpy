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

"""Test helper functions.

Provides convenient utilities for generating and handling test data.
"""

__all__ = [
    "shaped_arange",
    "shaped_random",
    "shaped_reverse_arange",
    "assert_array_list_equal",
    "suppress_warnings",
    "with_seed",
    "generate_test_data",
    "TEST_SHAPES",
    "TEST_DTYPES",
    "TEST_ORDERS",
]

import functools

import numpy


def shaped_arange(shape, dtype=numpy.float64, order="C", xp=None, start=0):
<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    """Generate a sequential array with the given shape.

    Produces a contiguous integer sequence starting from ``start``,
    then reshapes it to ``shape``. Useful in tests because the values
    make it easy to verify array operations.
=======
    """生成指定形状的序列数组

    生成从start开始的连续整数序列，然后reshape成指定形状。
    这在测试中非常有用，因为可以轻松验证数组操作的正确性。
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py

    Args:
        start: Starting value (default 0).
    """
    if xp is None:
        xp = numpy

<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    if isinstance(shape, int):
        shape = (shape,)

=======
    # 处理shape参数
    if isinstance(shape, int):
        shape = (shape,)

    # 计算总元素数
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
    size = 1
    for dim in shape:
        size *= dim

<<<<<<< HEAD:src/asnumpy/testing/_helper.py
=======
    # 生成序列 (从 start 到 start+size)
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
    if xp is numpy:
        arr = numpy.arange(start, start + size, dtype=dtype)
        return arr.reshape(shape, order=order)
    else:
        # For asnumpy, generate with numpy then convert
        arr = numpy.arange(start, start + size, dtype=dtype)
        arr = arr.reshape(shape, order=order)
        return xp.ndarray.from_numpy(arr)


def shaped_random(shape, dtype=numpy.float64, scale=1.0, seed=None, xp=None):
<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    """Generate a random array with the given shape.

    Produces a uniform-distribution random array.
=======
    """生成指定形状的随机数组

    生成服从均匀分布的随机数数组。
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
    """
    if xp is None:
        xp = numpy

<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    if isinstance(shape, int):
        shape = (shape,)

    if seed is not None:
        numpy.random.seed(seed)

=======
    # 处理shape参数
    if isinstance(shape, int):
        shape = (shape,)

    # 设置随机种子
    if seed is not None:
        numpy.random.seed(seed)

    # 生成随机数
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
    if xp is numpy:
        arr = numpy.random.random(shape).astype(dtype)
        return arr * scale
    else:
        # For asnumpy, generate with numpy then convert
        arr = numpy.random.random(shape).astype(dtype)
        arr = arr * scale
        return xp.ndarray.from_numpy(arr)


def shaped_reverse_arange(shape, dtype=numpy.float64, order="C", xp=None):
<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    """Generate a descending-order sequential array with the given shape.

    Produces a descending sequence then reshapes it to ``shape``.
    Useful for testing operations on reverse-sorted data.
=======
    """生成指定形状的反向序列数组

    生成从大到小的序列，然后reshape成指定形状。
    用于测试降序数据的处理。
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
    """
    if xp is None:
        xp = numpy

<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    if isinstance(shape, int):
        shape = (shape,)

=======
    # 处理shape参数
    if isinstance(shape, int):
        shape = (shape,)

    # 计算总元素数
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
    size = 1
    for dim in shape:
        size *= dim

<<<<<<< HEAD:src/asnumpy/testing/_helper.py
=======
    # 生成反向序列
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
    if xp is numpy:
        arr = numpy.arange(size - 1, -1, -1, dtype=dtype)
        return arr.reshape(shape, order=order)
    else:
        # For asnumpy, generate with numpy then convert
        arr = numpy.arange(size - 1, -1, -1, dtype=dtype)
        arr = arr.reshape(shape, order=order)
        return xp.ndarray.from_numpy(arr)


def assert_array_list_equal(x_list, y_list, err_msg="", verbose=True):
<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    """Assert that two lists of arrays are element-wise equal.

    Used for testing functions that return multiple arrays.
=======
    """比较两个数组列表是否相等

    用于测试返回多个数组的函数。
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
    """
    from . import _array

    if len(x_list) != len(y_list):
        raise AssertionError(f"List lengths differ: {len(x_list)} vs {len(y_list)}")

<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    for i, (x, y) in enumerate(zip(x_list, y_list, strict=False)):
=======
    for i, (x, y) in enumerate(zip(x_list, y_list)):
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
        try:
            _array.assert_array_equal(x, y, err_msg, verbose)
        except AssertionError as e:
            raise AssertionError(f"Arrays at index {i} differ: {e}") from e


def suppress_warnings(func):
<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    """Decorator: suppress warnings during function execution.

    Use in tests to temporarily silence known warnings.
=======
    """装饰器：抑制函数执行时的警告

    用于测试中临时忽略已知的警告。
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py

    Examples:
        @suppress_warnings
        def test_something():
            # warnings here will be suppressed
            pass
    """
    import warnings

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


def with_seed(seed):
<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    """Decorator: run a test with a fixed random seed for reproducibility.

    Args:
        seed: Random seed.
=======
    """装饰器：使用固定随机种子运行测试

    确保测试的可重现性。

    Args:
        seed: 随机种子
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py

    Examples:
        @with_seed(42)
        def test_random_function():
            arr = numpy.random.random((3, 3))
            # same random values on every run
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Save current random state
            old_state = numpy.random.get_state()
            try:
                numpy.random.seed(seed)
                return func(*args, **kwargs)
            finally:
                # Restore original random state
                numpy.random.set_state(old_state)

        return wrapper

    return decorator


def generate_test_data(func):
<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    """Decorator: automatically generate test data for a test function.

    Generates common test cases based on the function signature.
    This is a simplified implementation that can be extended as needed.
=======
    """装饰器：为测试函数自动生成测试数据

    根据函数签名自动生成常见的测试用例。
    这是一个简化的实现，可以根据需要扩展。
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py

    Examples:
        @generate_test_data
        def test_add(a, b):
            return a + b
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Simplified: delegate directly to the original function
        return func(*args, **kwargs)

    return wrapper


# Common test constants
TEST_SHAPES = [
<<<<<<< HEAD:src/asnumpy/testing/_helper.py
    (),          # scalar
    (0,),        # empty array
    (1,),        # single element
    (5,),        # 1-D
    (2, 3),      # 2-D
    (2, 3, 4),   # 3-D
    (1, 2, 3, 4),# 4-D
=======
    (),  # 标量
    (0,),  # 空数组
    (1,),  # 单元素
    (5,),  # 一维
    (2, 3),  # 二维
    (2, 3, 4),  # 三维
    (1, 2, 3, 4),  # 四维
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_helper.py
]

TEST_DTYPES = [
    numpy.float32,
    numpy.float64,
    numpy.int32,
    numpy.int64,
]

TEST_ORDERS = ["C", "F"]

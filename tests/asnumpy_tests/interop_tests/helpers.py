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

"""互操作性协议测试的共享辅助函数。

这些工具封装了 NumPy 与 AsNumpy 之间的类型转换和断言逻辑，
使测试用例可以聚焦于行为验证，而非重复的转换样板代码。
"""

from __future__ import annotations

import numpy as np


def to_numpy(value):
    """将 AsNumpy ndarray 或其他类数组对象转换为 NumPy ndarray。

    如果 value 具有 ``to_numpy()`` 方法（如 AsNumpy ndarray），
    则调用该方法；否则委托给 ``np.asarray``。

    Args:
        value: 任意类数组对象或 AsNumpy ndarray。

    Returns:
        numpy.ndarray: 转换后的 NumPy 数组。
    """
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def assert_numpy_equal(actual, expected):
    """断言 actual 与 expected 的值完全相等。

    自动将 AsNumpy ndarray 转换为 NumPy 数组后再比较，
    同时校验 shape 和 dtype 的一致性。

    Args:
        actual: 实际值（AsNumpy ndarray 或类数组对象）。
        expected: 期望值（NumPy ndarray 或类数组对象）。

    Raises:
        AssertionError: shape、dtype 或数值不匹配时抛出。
    """
    actual_np = to_numpy(actual)
    expected_np = np.asarray(expected)
    assert actual_np.shape == expected_np.shape, (
        f"shape mismatch: {actual_np.shape} != {expected_np.shape}"
    )
    assert actual_np.dtype == expected_np.dtype, (
        f"dtype mismatch: {actual_np.dtype} != {expected_np.dtype}"
    )
    np.testing.assert_array_equal(actual_np, expected_np)


def assert_numpy_allclose(actual, expected, *, rtol=1e-6, atol=1e-6):
    """断言 actual 与 expected 的值在容差范围内近似相等。

    浮点运算可能存在精度损失，使用此函数进行近似比较。
    自动将 AsNumpy ndarray 转换为 NumPy 数组后再比较，
    同时校验 shape 和 dtype 的一致性。

    Args:
        actual: 实际值（AsNumpy ndarray 或类数组对象）。
        expected: 期望值（NumPy ndarray 或类数组对象）。
        rtol: 相对容差（默认 1e-6）。
        atol: 绝对容差（默认 1e-6）。

    Raises:
        AssertionError: shape、dtype 不匹配或数值超出容差范围时抛出。
    """
    actual_np = to_numpy(actual)
    expected_np = np.asarray(expected)
    assert actual_np.shape == expected_np.shape, (
        f"shape mismatch: {actual_np.shape} != {expected_np.shape}"
    )
    assert actual_np.dtype == expected_np.dtype, (
        f"dtype mismatch: {actual_np.dtype} != {expected_np.dtype}"
    )
    np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol, equal_nan=True)

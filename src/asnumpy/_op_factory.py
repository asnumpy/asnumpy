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

"""算子工厂 — 自动生成 Python wrapper，消除样板代码。

设计原则 (KISS):
- 工厂函数仅负责: dtype转换 + ndarray包装
- 复杂逻辑 (signbit修正、floor特殊处理) 仍手写
- 不引入算子图/延迟执行

使用:
    from ._op_factory import wrap_unary, wrap_binary, wrap_reduction

    sin = wrap_unary(_sin, auto_cast=True)
    add = wrap_binary(_add)
    sum = wrap_reduction(_sum)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import _convert_dtype, ndarray


def _cast_to_float32(arr: Any) -> Any:
    """Cast integer/bool inputs to float32 for CANN operator compatibility."""
    if isinstance(arr, ndarray):
        if arr.dtype.kind in ("i", "u", "b"):
            return ndarray.from_numpy(arr.to_numpy().astype(np.float32))
        return arr
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind in ("i", "u", "b"):
            return arr.astype(np.float32)
        return arr
    return arr


# ── Unary operators ──────────────────────────────────────────────


def wrap_unary(cpp_func, *, auto_cast: bool = False):
    """Wrap a C++ unary function: fn(x) -> ndarray."""
    if auto_cast:

        def wrapper(x):
            return ndarray(cpp_func(_cast_to_float32(x)))
    else:

        def wrapper(x):
            return ndarray(cpp_func(x))

    wrapper.__name__ = cpp_func.__name__
    wrapper.__qualname__ = cpp_func.__name__
    return wrapper


def wrap_unary_dtype(cpp_func, *, auto_cast: bool = False):
    """Wrap a C++ unary function with dtype: fn(x, dtype=None) -> ndarray."""
    if auto_cast:

        def wrapper(x, dtype=None):
            return ndarray(cpp_func(_cast_to_float32(x), _convert_dtype(dtype)))
    else:

        def wrapper(x, dtype=None):
            return ndarray(cpp_func(x, _convert_dtype(dtype)))

    wrapper.__name__ = cpp_func.__name__
    wrapper.__qualname__ = cpp_func.__name__
    return wrapper


# ── Binary operators ─────────────────────────────────────────────


def wrap_binary(cpp_func):
    """Wrap a C++ binary function: fn(x1, x2, dtype=None) -> ndarray."""

    def wrapper(x1, x2, dtype=None):
        return ndarray(cpp_func(x1, x2, _convert_dtype(dtype)))

    wrapper.__name__ = cpp_func.__name__
    wrapper.__qualname__ = cpp_func.__name__
    return wrapper


def wrap_binary_no_dtype(cpp_func):
    """Wrap a C++ binary function WITHOUT dtype: fn(x1, x2) -> ndarray."""

    def wrapper(x1, x2):
        return ndarray(cpp_func(x1, x2))

    wrapper.__name__ = cpp_func.__name__
    wrapper.__qualname__ = cpp_func.__name__
    return wrapper


def wrap_binary_auto_cast(cpp_func):
    """Wrap a C++ binary function with auto-cast of both args (with dtype)."""

    def wrapper(x1, x2, dtype=None):
        return ndarray(cpp_func(_cast_to_float32(x1), _cast_to_float32(x2), _convert_dtype(dtype)))

    wrapper.__name__ = cpp_func.__name__
    wrapper.__qualname__ = cpp_func.__name__
    return wrapper


def wrap_binary_auto_cast_no_dtype(cpp_func):
    """Wrap a C++ binary function with auto-cast (no dtype param)."""

    def wrapper(x1, x2):
        return ndarray(cpp_func(_cast_to_float32(x1), _cast_to_float32(x2)))

    wrapper.__name__ = cpp_func.__name__
    wrapper.__qualname__ = cpp_func.__name__
    return wrapper


# ── Reduction operators ──────────────────────────────────────────


def wrap_reduction(cpp_func_scalar, cpp_func_axis, *, with_dtype: bool = False):
    """Wrap a C++ reduction: fn(a, axis=None, keepdims=False, [dtype=None]) -> ndarray|float.

    Args:
        cpp_func_scalar: C++ overload for axis=None (returns scalar).
        cpp_func_axis: C++ overload with axis/keepdims/[dtype] (returns NPUArray).
        with_dtype: Whether the C++ function accepts a dtype parameter.
    """
    if with_dtype:

        def wrapper(a, axis=None, keepdims=False, dtype=None):
            if axis is None:
                return cpp_func_scalar(a)
            return ndarray(cpp_func_axis(a, axis, keepdims, _convert_dtype(dtype)))
    else:

        def wrapper(a, axis=None, keepdims=False):
            if axis is None:
                return cpp_func_scalar(a)
            return ndarray(cpp_func_axis(a, axis, keepdims))

    wrapper.__name__ = cpp_func_axis.__name__
    wrapper.__qualname__ = cpp_func_axis.__name__
    return wrapper


def wrap_cumulative(cpp_func, *, with_dtype: bool = True):
    """Wrap a C++ cumulative function: fn(a, axis=None, [dtype=None]) -> ndarray."""
    if with_dtype:

        def wrapper(a, axis=None, dtype=None):
            return ndarray(cpp_func(a, axis, _convert_dtype(dtype)))
    else:

        def wrapper(a, axis=None):
            return ndarray(cpp_func(a, axis))

    wrapper.__name__ = cpp_func.__name__
    wrapper.__qualname__ = cpp_func.__name__
    return wrapper

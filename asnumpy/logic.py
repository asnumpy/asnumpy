# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

from ._types import ArrayLike, DTypeLike, AxisLike
from .lib.asnumpy_core.logic import (
    all as _all,
    any as _any,
    equal as _equal,
    greater as _greater,
    greater_equal as _greater_equal,
    isfinite as _isfinite,
    isinf as _isinf,
    isneginf as _isneginf,
    isposinf as _isposinf,
    less as _less,
    less_equal as _less_equal,
    logical_and as _logical_and,
    logical_not as _logical_not,
    logical_or as _logical_or,
    logical_xor as _logical_xor,
    not_equal as _not_equal,
)
from .utils import ndarray, _convert_dtype


def all(x: ArrayLike, axis: AxisLike = None, keepdims: bool = False) -> ndarray:
    if axis is None:
        return ndarray(_all(x))
    return ndarray(_all(x, axis, keepdims))


def any(x: ArrayLike, axis: AxisLike = None, keepdims: bool = False) -> ndarray:
    if axis is None:
        return ndarray(_any(x))
    return ndarray(_any(x, axis, keepdims))


def isfinite(x: ArrayLike) -> ndarray:
    return ndarray(_isfinite(x))


def isinf(x: ArrayLike) -> ndarray:
    return ndarray(_isinf(x))


def isneginf(x: ArrayLike) -> ndarray:
    return ndarray(_isneginf(x))


def isposinf(x: ArrayLike) -> ndarray:
    return ndarray(_isposinf(x))


def logical_and(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logical_and(x1, x2))


def logical_or(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logical_or(x1, x2))


def logical_not(x: ArrayLike) -> ndarray:
    return ndarray(_logical_not(x))


def logical_xor(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logical_xor(x1, x2))


def greater(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_greater(x1, x2, _convert_dtype(dtype)))


def greater_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_greater_equal(x1, x2, _convert_dtype(dtype)))


def less(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_less(x1, x2, _convert_dtype(dtype)))


def less_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_less_equal(x1, x2, _convert_dtype(dtype)))


def equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_equal(x1, x2, _convert_dtype(dtype)))


def not_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_not_equal(x1, x2, _convert_dtype(dtype)))

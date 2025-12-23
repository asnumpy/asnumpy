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

from typing import Optional, Union, Sequence, Any
import numpy as np
from .lib.asnumpy_core.logic import (
    all as _ap_all,
    any as _ap_any,
    equal as _ap_equal,
    greater as _ap_greater,
    greater_equal as _ap_greater_equal,
    isfinite as _ap_isfinite,
    isinf as _ap_isinf,
    isneginf as _ap_isneginf,
    isposinf as _ap_isposinf,
    less as _ap_less,
    less_equal as _ap_less_equal,
    logical_and as _ap_logical_and,
    logical_not as _ap_logical_not,
    logical_or as _ap_logical_or,
    logical_xor as _ap_logical_xor,
    not_equal as _ap_not_equal,
)
from .utils import ndarray, _convert_dtype


def all(
    x: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> ndarray:
    if axis is None:
        return ndarray(_ap_all(x))
    return ndarray(_ap_all(x, axis, keepdims))


def any(
    x: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> ndarray:
    if axis is None:
        return ndarray(_ap_any(x))
    return ndarray(_ap_any(x, axis, keepdims))


def isfinite(x: ndarray) -> ndarray:
    return ndarray(_ap_isfinite(x))


def isinf(x: ndarray) -> ndarray:
    return ndarray(_ap_isinf(x))


def isneginf(x: ndarray) -> ndarray:
    return ndarray(_ap_isneginf(x))


def isposinf(x: ndarray) -> ndarray:
    return ndarray(_ap_isposinf(x))


def logical_and(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_logical_and(x1, x2))


def logical_or(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_logical_or(x1, x2))


def logical_not(x: ndarray) -> ndarray:
    return ndarray(_ap_logical_not(x))


def logical_xor(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_logical_xor(x1, x2))


def greater(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_greater(x1, x2, _convert_dtype(dtype)))


def greater_equal(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_greater_equal(x1, x2, _convert_dtype(dtype)))


def less(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_less(x1, x2, _convert_dtype(dtype)))


def less_equal(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_less_equal(x1, x2, _convert_dtype(dtype)))


def equal(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_equal(x1, x2, _convert_dtype(dtype)))


def not_equal(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_not_equal(x1, x2, _convert_dtype(dtype)))

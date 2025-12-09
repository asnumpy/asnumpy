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
from .asnumpy_core.logic import (
    all as ap_all,
    any as ap_any,
    equal as ap_equal,
    greater as ap_greater,
    greater_equal as ap_greater_equal,
    isfinite as ap_isfinite,
    isinf as ap_isinf,
    isneginf as ap_isneginf,
    isposinf as ap_isposinf,
    less as ap_less,
    less_equal as ap_less_equal,
    logical_and as ap_logical_and,
    logical_not as ap_logical_not,
    logical_or as ap_logical_or,
    logical_xor as ap_logical_xor,
    not_equal as ap_not_equal
)
from .utils import ndarray

def _convert_dtype(dtype: Optional[np.dtype]) -> Optional[np.dtype]:
    if dtype is None:
        return None
    if not isinstance(dtype, np.dtype):
        return np.dtype(dtype)
    return dtype

def all(x: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> ndarray:
    if axis is None:
        return ndarray(ap_all(x._impl))
    return ndarray(ap_all(x._impl, axis, keepdims))

def any(x: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> ndarray:
    if axis is None:
        return ndarray(ap_any(x._impl))
    return ndarray(ap_any(x._impl, axis, keepdims))

def isfinite(x: ndarray) -> ndarray:
    return ndarray(ap_isfinite(x._impl))

def isinf(x: ndarray) -> ndarray:
    return ndarray(ap_isinf(x._impl))

def isneginf(x: ndarray) -> ndarray:
    return ndarray(ap_isneginf(x._impl))

def isposinf(x: ndarray) -> ndarray:
    return ndarray(ap_isposinf(x._impl))

def logical_and(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_logical_and(x1._impl, x2._impl))

def logical_or(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_logical_or(x1._impl, x2._impl))

def logical_not(x: ndarray) -> ndarray:
    return ndarray(ap_logical_not(x._impl))

def logical_xor(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_logical_xor(x1._impl, x2._impl))

def greater(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_greater(x1_impl, x2_impl, _convert_dtype(dtype)))

def greater_equal(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_greater_equal(x1_impl, x2_impl, _convert_dtype(dtype)))

def less(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_less(x1_impl, x2_impl, _convert_dtype(dtype)))

def less_equal(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_less_equal(x1_impl, x2_impl, _convert_dtype(dtype)))

def equal(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_equal(x1_impl, x2_impl, _convert_dtype(dtype)))

def not_equal(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_not_equal(x1_impl, x2_impl, _convert_dtype(dtype)))
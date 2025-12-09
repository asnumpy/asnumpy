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

from typing import Union, Optional, Sequence, Any
import numpy as np
from .asnumpy_core.array import (
    empty as ap_empty,
    empty_like as ap_empty_like,
    eye as ap_eye,
    full as ap_full,
    full_like as ap_full_like,
    identity as ap_identity,
    linspace as ap_linspace,
    ones as ap_ones,
    ones_like as ap_ones_like,
    zeros as ap_zeros,
    zeros_like as ap_zeros_like
)
from .utils import ndarray, _convert_dtype


def zeros(shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_zeros(shape, _convert_dtype(dtype)))


def zeros_like(other: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    other_impl = other.impl if isinstance(other, ndarray) else other
    return ndarray(ap_zeros_like(other_impl, _convert_dtype(dtype)))


def full(shape: Union[int, Sequence[int]], value: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_full(shape, value, _convert_dtype(dtype)))


def full_like(other: Any, value: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    other_impl = other.impl if isinstance(other, ndarray) else other
    return ndarray(ap_full_like(other_impl, value, _convert_dtype(dtype)))


def empty(shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_empty(shape, _convert_dtype(dtype)))


def empty_like(prototype: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    prototypeimpl = prototype.impl if isinstance(prototype, ndarray) else prototype
    return ndarray(ap_empty_like(prototypeimpl, _convert_dtype(dtype)))


def eye(n: int, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_eye(n, _convert_dtype(dtype)))


def ones(shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_ones(shape, _convert_dtype(dtype)))


def ones_like(other: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    other_impl = other.impl if isinstance(other, ndarray) else other
    return ndarray(ap_ones_like(other_impl, _convert_dtype(dtype)))


def identity(n: int, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_identity(n, _convert_dtype(dtype)))


def linspace(start: Union[int, float], end: Union[int, float], 
             steps: int = 50, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_linspace(start, end, steps, _convert_dtype(dtype)))
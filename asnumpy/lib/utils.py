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

from typing import Sequence
import numpy as np
from .asnumpy_core import ndarray as _ndarray
from .asnumpy_core import broadcast_shape as _broadcast_shape


class ndarray:
    def __init__(self, shape: Sequence[int], dtype: np.dtype):
        self._impl = _ndarray(shape, dtype)
    
    def __init__(self, other: _ndarray):
        self._impl = other
    
    def to_numpy(self) -> np.ndarray:
        return self._impl.to_numpy()
    
    @classmethod
    def from_numpy(cls, host_data: np.ndarray) -> 'ndarray':
        result = cls.__new__(cls)
        result._impl = _ndarray.from_numpy(host_data)
        return result
    
    def __repr__(self) -> str:
        return f"ndarray(shape={self.shape}, dtype={self.dtype})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    @property
    def shape(self) -> tuple:
        return tuple(self._impl.shape)
    
    @property
    def dtype(self) -> np.dtype:
        return self._impl.dtype
    
    @property
    def aclDtype(self) -> int:
        return self._impl.aclDtype
    
    @property
    def impl(self):
        return self._impl


def broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple:
    return _broadcast_shape(shape_a, shape_b)


def _convert_dtype(dtype):
    """Convert dtype parameter to appropriate format if needed"""
    if dtype is None:
        return None
    if not isinstance(dtype, np.dtype):
        return np.dtype(dtype)
    return dtype


__all__ = ['ndarray', 'broadcast_shape']
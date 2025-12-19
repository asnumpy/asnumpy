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

from typing import Sequence, Union, overload
import numpy as np
from .lib.asnumpy_core import ndarray as _ndarray
from .lib.asnumpy_core import broadcast_shape as _broadcast_shape


class ndarray(_ndarray):
    @overload
    def __init__(self, shape: Sequence[int], dtype: np.dtype) -> None: 
        ...

    @overload
    def __init__(self, other: _ndarray) -> None: 
        ...

    def __init__(self, shape_or_array, dtype: np.dtype = None):
        if isinstance(shape_or_array, _ndarray):
            super().__init__(shape_or_array)
        elif isinstance(shape_or_array, (Sequence, int)):
            if dtype is None:
                raise ValueError("dtype must be specified when initializing with shape")
            shape = (
                shape_or_array
                if isinstance(shape_or_array, Sequence)
                else (shape_or_array,)
            )
            super().__init__(shape, np.dtype(dtype))
        else:
            raise TypeError(
                f"Unsupported type for initialization: {type(shape_or_array)}"
            )

    def __repr__(self) -> str:
        return f"ndarray(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def shape(self) -> tuple:
        return super().shape

    @property
    def dtype(self) -> np.dtype:
        return super().dtype

    @property
    def acl_dtype(self) -> int:
        return super().aclDtype

    @classmethod
    def from_numpy(cls, host_data: np.ndarray) -> "ndarray":
        base_obj = _ndarray.from_numpy(host_data)
        return cls(base_obj)

    def to_numpy(self) -> np.ndarray:
        return super().to_numpy()


def broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple:
    return _broadcast_shape(shape_a, shape_b)


def _convert_dtype(dtype):
    """Convert dtype parameter to appropriate format if needed"""
    if dtype is None:
        return None
    if not isinstance(dtype, np.dtype):
        return np.dtype(dtype)
    return dtype


def _convert_size(size: Union[int, Sequence[int]]) -> Sequence[int]:
    """Convert size from int to tuple"""
    if isinstance(size, int):
        return (size,)
    return size

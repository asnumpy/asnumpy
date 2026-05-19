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

import operator
from collections.abc import Sequence
from typing import overload

import numpy as np
from loguru import logger

from ._core import broadcast_shape as _broadcast_shape
from ._core import ndarray as _ndarray


class ndarray(_ndarray):
    @overload
    def __init__(self, shape: Sequence[int], dtype: np.dtype) -> None: ...

    @overload
    def __init__(self, other: _ndarray) -> None: ...

    def __init__(self, shape_or_array, dtype=None):
        if isinstance(shape_or_array, _ndarray):
            super().__init__(shape_or_array)
        elif isinstance(shape_or_array, (Sequence, int)):
            if dtype is None:
                raise ValueError("dtype must be specified when initializing with shape")
            shape = shape_or_array if isinstance(shape_or_array, Sequence) else (shape_or_array,)
            super().__init__(shape, np.dtype(dtype))
        else:
            raise TypeError(f"Unsupported type for initialization: {type(shape_or_array)}")

    def __repr__(self) -> str:
        return f"ndarray(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def shape(self) -> tuple:
        return super().shape  # type: ignore[no-any-return]

    @property
    def dtype(self) -> np.dtype:
        return super().dtype  # type: ignore[no-any-return]

    @property
    def acl_dtype(self) -> int:
        return super().aclDtype  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:
        return super().ndim  # type: ignore[no-any-return]

    @property
    def itemsize(self) -> int:
        return super().itemsize  # type: ignore[no-any-return]

    @property
    def nbytes(self) -> int:
        return super().nbytes  # type: ignore[no-any-return]

    @property
    def strides(self) -> tuple:
        return super().strides  # type: ignore[no-any-return]

    @classmethod
    def from_numpy(cls, host_data: np.ndarray) -> "ndarray":
        base_obj = _ndarray.from_numpy(host_data)
        return cls(base_obj)

    def to_numpy(self) -> np.ndarray:
        return super().to_numpy()


@logger.catch(reraise=True)
def broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple:
    logger.debug(f"Broadcasting shapes {shape_a}, {shape_b}")
    return _broadcast_shape(shape_a, shape_b)  # type: ignore[no-any-return]


@logger.catch(reraise=True)
def _convert_dtype(dtype):
    """Convert dtype parameter to appropriate format if needed"""
    logger.debug(f"Converting dtype {dtype}")
    if dtype is None:
        return None
    if not isinstance(dtype, np.dtype):
        return np.dtype(dtype)
    return dtype


@logger.catch(reraise=True)
def _convert_size(size: int | Sequence[int]) -> Sequence[int]:
    """Convert size from int to tuple"""
    logger.debug(f"Converting size {size}")
    if isinstance(size, int):
        return (size,)
    return size


@logger.catch(reraise=True)
def _normalize_shape(shape: int | Sequence[int]) -> list[int]:
    """Normalize a shape argument to a list and reject negative dimensions."""
    logger.debug(f"Normalizing shape {shape}")

    if isinstance(shape, (int, np.integer)):
        normalized = [operator.index(shape)]
    else:
        normalized = [operator.index(dim) for dim in shape]

    if any(dim < 0 for dim in normalized):
        raise ValueError("negative dimensions are not allowed")

    return normalized

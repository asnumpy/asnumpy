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
    """NumPy-compatible array backed by Ascend NPU memory.

    Wraps C++ NPUArray with Python-side NumPy-compatible attributes.
    """

    @overload
    def __init__(self, shape: Sequence[int], dtype: np.dtype) -> None: ...

    @overload
    def __init__(self, other: _ndarray) -> None: ...

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

    # -- NumPy-compatible properties --

    @property
    def shape(self) -> tuple:
        return super().shape

    @property
    def dtype(self) -> np.dtype:
        return super().dtype

    @property
    def acl_dtype(self) -> int:
        return super().aclDtype

    @property
    def ndim(self) -> int:
        return super().ndim

    @property
    def itemsize(self) -> int:
        return super().itemsize

    @property
    def nbytes(self) -> int:
        return super().nbytes

    @property
    def strides(self) -> tuple:
        return super().strides

    @property
    def size(self) -> int:
        """Total number of elements in the array."""
        s = 1
        for d in self.shape:
            s *= d
        return s

    @property
    def T(self) -> "ndarray":
        """Transpose by reversing axes.

        .. warning::
            Performs NPU→CPU→NPU round-trip via NumPy. This is expensive for large arrays.
            A zero-copy NPU transpose kernel is planned.
        """
        if self.ndim < 2:
            return ndarray(self)
        cpu = self.to_numpy().T.copy()
        return ndarray.from_numpy(cpu)

    @property
    def real(self) -> "ndarray":
        """Real part of the array.

        .. warning::
            Performs NPU→CPU→NPU round-trip for complex arrays. Non-complex arrays
            return a shallow copy (no data transfer).
        """
        if np.issubdtype(self.dtype, np.complexfloating):
            cpu = np.real(self.to_numpy())
            return ndarray.from_numpy(cpu)
        return ndarray(self)

    @property
    def imag(self) -> "ndarray":
        """Imaginary part of the array.

        .. warning::
            Performs NPU→CPU→NPU round-trip for complex arrays.
        """
        if np.issubdtype(self.dtype, np.complexfloating):
            cpu = np.imag(self.to_numpy())
            return ndarray.from_numpy(cpu)
        return ndarray(self.shape, self.dtype)  # zero-filled on device

    def __len__(self) -> int:
        """Length of the first dimension."""
        if self.ndim == 0:
            raise TypeError("len() of a 0-d array")
        return self.shape[0]

    def __repr__(self) -> str:
        return f"ndarray(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        try:
            arr = self.to_numpy()
            return np.array2string(arr, separator=", ")
        except Exception:
            return self.__repr__()

    # -- Arithmetic operator overloading (NumPy-compatible) --

    @staticmethod
    def _scalar_to_array(x):
        """Convert Python scalars to ndarray for C++ operator compatibility.

        Integer scalars are cast to float32 to avoid CANN dtype mismatch
        when operating with floating-point arrays.
        """
        if isinstance(x, (int, bool, np.integer)):
            return ndarray.from_numpy(np.array(x, dtype=np.float32))
        if isinstance(x, (float, np.floating)):
            return ndarray.from_numpy(np.array(x, dtype=np.float32))
        return x

    def __add__(self, other):
        from .math import add
        return add(self, self._scalar_to_array(other))

    def __radd__(self, other):
        from .math import add
        return add(self._scalar_to_array(other), self)

    def __sub__(self, other):
        from .math import subtract
        return subtract(self, self._scalar_to_array(other))

    def __rsub__(self, other):
        from .math import subtract
        return subtract(self._scalar_to_array(other), self)

    def __mul__(self, other):
        from .math import multiply
        return multiply(self, self._scalar_to_array(other))

    def __rmul__(self, other):
        from .math import multiply
        return multiply(self._scalar_to_array(other), self)

    def __truediv__(self, other):
        from .math import true_divide
        return true_divide(self, self._scalar_to_array(other))

    def __rtruediv__(self, other):
        from .math import true_divide
        return true_divide(self._scalar_to_array(other), self)

    def __floordiv__(self, other):
        from .math import floor_divide
        return floor_divide(self, self._scalar_to_array(other))

    def __rfloordiv__(self, other):
        from .math import floor_divide
        return floor_divide(self._scalar_to_array(other), self)

    def __mod__(self, other):
        from .math import remainder
        return remainder(self, self._scalar_to_array(other))

    def __rmod__(self, other):
        from .math import remainder
        return remainder(self._scalar_to_array(other), self)

    def __pow__(self, other):
        from .math import power
        return power(self, self._scalar_to_array(other))

    def __rpow__(self, other):
        from .math import power
        return power(self._scalar_to_array(other), self)

    def __matmul__(self, other):
        from .linalg.direct import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .linalg.direct import matmul
        return matmul(other, self)

    # -- Unary operators --

    def __neg__(self):
        from .math import negative
        return negative(self)

    def __pos__(self):
        from .math import positive
        return positive(self)

    def __abs__(self):
        from .math import absolute
        return absolute(self)

    # -- Comparison operators (element-wise, return boolean ndarray) --

    def __eq__(self, other):
        from .logic import equal
        return equal(self, self._scalar_to_array(other))

    def __ne__(self, other):
        from .logic import not_equal
        return not_equal(self, self._scalar_to_array(other))

    def __lt__(self, other):
        from .logic import less
        return less(self, self._scalar_to_array(other))

    def __le__(self, other):
        from .logic import less_equal
        return less_equal(self, self._scalar_to_array(other))

    def __gt__(self, other):
        from .logic import greater
        return greater(self, self._scalar_to_array(other))

    def __ge__(self, other):
        from .logic import greater_equal
        return greater_equal(self, self._scalar_to_array(other))

    # -- Convenience methods --

    @classmethod
    def from_numpy(cls, host_data: np.ndarray) -> "ndarray":
        base_obj = _ndarray.from_numpy(host_data)
        return cls(base_obj)

    def to_numpy(self) -> np.ndarray:
        return super().to_numpy()

    def astype(self, dtype) -> "ndarray":
        """Cast to a specified dtype.

        .. warning::
            Performs NPU→CPU→NPU round-trip via NumPy.
        """
        cpu = self.to_numpy().astype(dtype)
        return ndarray.from_numpy(cpu)

    def flatten(self) -> "ndarray":
        """Return a flattened copy of the array.

        .. warning::
            Performs NPU→CPU→NPU round-trip via NumPy.
        """
        cpu = self.to_numpy().flatten()
        return ndarray.from_numpy(cpu.copy())

    def ravel(self) -> "ndarray":
        """Return a flattened copy (always copies on NPU).

        .. warning::
            Performs NPU→CPU→NPU round-trip via NumPy. Unlike NumPy, never returns a view.
        """
        return self.flatten()

    def copy(self) -> "ndarray":
        """Deep copy of the array."""
        return ndarray(self)


@logger.catch(reraise=True)
def broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple:
    logger.debug(f"Broadcasting shapes {shape_a}, {shape_b}")
    return _broadcast_shape(shape_a, shape_b)


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


# NumPy AxisError: np.exceptions.AxisError (NumPy >= 1.22), fallback np.AxisError
_AxisError = getattr(np.exceptions, "AxisError", getattr(np, "AxisError", IndexError))


@logger.catch(reraise=True)
def validate_axis(axis: int | Sequence[int] | None, ndim: int, *,
                  name: str = "axis") -> int | tuple[int, ...] | None:
    """Validate that an axis parameter is within bounds.

    Args:
        axis: None, int, or tuple of int
        ndim: Number of array dimensions
        name: Parameter name for error messages

    Returns:
        Validated axis value (normalized)

    Raises:
        np.exceptions.AxisError: axis out of bounds
        TypeError: axis has wrong type
    """
    if axis is None:
        return None

    if isinstance(axis, int):
        _check_single_axis(axis, ndim, name)
        return axis

    if isinstance(axis, tuple):
        for ax in axis:
            if not isinstance(ax, (int, np.integer)):
                raise TypeError(
                    f"{name} must be an integer or tuple of integers, got {type(ax).__name__}"
                )
            _check_single_axis(int(ax), ndim, name)
        return axis

    if isinstance(axis, list):
        return validate_axis(tuple(axis), ndim, name=name)

    raise TypeError(
        f"{name} must be an integer, tuple of integers, or None, "
        f"got {type(axis).__name__}"
    )


def _check_single_axis(axis: int, ndim: int, name: str = "axis") -> None:
    """Validate a single axis value is in range."""
    if ndim == 0:
        raise _AxisError(
            f"{name} {axis} is out of bounds for array of dimension {ndim}"
        )
    if axis < -ndim or axis >= ndim:
        raise _AxisError(
            f"{name} {axis} is out of bounds for array of dimension {ndim}"
        )

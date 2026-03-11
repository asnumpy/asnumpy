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
from loguru import logger
import numpy as np
from .lib.asnumpy_core import ndarray as _ndarray
from .lib.asnumpy_core import broadcast_shape as _broadcast_shape


class ndarray(_ndarray):
    """
    Represent a multi-dimensional array on the device.

    This class encapsulates a C++ backend array and provides a Python interface
    for performing array operations directly on the accelerator (NPU).
    Users interact with device-resident arrays through this class, enabling
    efficient computations while maintaining a NumPy-like interface.

    .. note::
        This class is typically not created directly. Use array creation
        routines such as :func:`asnumpy.zeros`, :func:`asnumpy.ones`, or
        :meth:`asnumpy.ndarray.from_numpy` to construct arrays on the device.
    """
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
        """
        Tuple of array dimensions.

        Returns
        -------
        tuple
            The shape of the array.
        """
        return super().shape

    @property
    def dtype(self) -> np.dtype:
        """
        Data-type of the array's elements.

        Returns
        -------
        numpy.dtype
            The data type of the array elements.
        """
        return super().dtype

    @property
    def acl_dtype(self) -> int:
        """
        Internal ACL data type identifier.

        Returns
        -------
        int
            The ACL data type enum value.
        """
        return super().aclDtype

    @classmethod
    def from_numpy(cls, host_data: np.ndarray) -> "ndarray":
        """
        Create an asnumpy.ndarray from a numpy.ndarray.

        This function copies the data from the host (CPU) to the device (NPU).

        Arguments
        ----------
        host_data : numpy.ndarray
            The input NumPy array.

        Returns
        -------
        asnumpy.ndarray
            A new array on the device containing the same data as `host_data`.

        Examples
        --------
        >>> import numpy as np
        >>> import asnumpy as ap
        >>> x_cpu = np.array([1, 2, 3])
        >>> x_npu = ap.ndarray.from_numpy(x_cpu)
        """
        base_obj = _ndarray.from_numpy(host_data)
        return cls(base_obj)

    def to_numpy(self) -> np.ndarray:
        """
        Return a copy of the array data as a numpy.ndarray on the host.

        This function copies the data from the device (NPU) to the host (CPU).

        Returns
        -------
        numpy.ndarray
            A NumPy array containing the data from the device array.

        Examples
        --------
        >>> import asnumpy as ap
        >>> x_npu = ap.zeros(3)
        >>> x_cpu = x_npu.to_numpy()
        >>> type(x_cpu)
        <class 'numpy.ndarray'>
        """
        return super().to_numpy()


@logger.catch
def broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple:
    logger.debug(f"Broadcasting shapes {shape_a}, {shape_b}")
    return _broadcast_shape(shape_a, shape_b)


@logger.catch
def _convert_dtype(dtype):
    """Convert dtype parameter to appropriate format if needed"""
    logger.debug(f"Converting dtype {dtype}")
    if dtype is None:
        return None
    if not isinstance(dtype, np.dtype):
        return np.dtype(dtype)
    return dtype


@logger.catch
def _convert_size(size: Union[int, Sequence[int]]) -> Sequence[int]:
    """Convert size from int to tuple"""
    logger.debug(f"Converting size {size}")
    if isinstance(size, int):
        return (size,)
    return size

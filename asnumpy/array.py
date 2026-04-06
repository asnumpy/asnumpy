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

from loguru import logger
import numpy as np
from ._types import ArrayLike, DTypeLike, ShapeLike, ScalarLike
from .lib.asnumpy_core.array import (
    empty as _empty,
    empty_like as _empty_like,
    eye as _eye,
    full as _full,
    full_like as _full_like,
    identity as _identity,
    linspace as _linspace,
    ones as _ones,
    ones_like as _ones_like,
    zeros as _zeros,
    zeros_like as _zeros_like,
)
from .utils import ndarray, _convert_dtype


@logger.catch
def zeros(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating zeros array shape={shape}, dtype={dtype}")
    return ndarray(_zeros(shape, _convert_dtype(dtype)))


@logger.catch
def zeros_like(other: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating zeros_like array other={other}, dtype={dtype}")
    return ndarray(_zeros_like(other, _convert_dtype(dtype)))


@logger.catch
def full(shape: ShapeLike, value: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating full array shape={shape}, value={value}, dtype={dtype}")
    return ndarray(_full(shape, value, _convert_dtype(dtype)))


@logger.catch
def full_like(other: ArrayLike, value: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating full_like array other={other}, value={value}, dtype={dtype}")
    return ndarray(_full_like(other, value, _convert_dtype(dtype)))


@logger.catch
def empty(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating empty array shape={shape}, dtype={dtype}")
    return ndarray(_empty(shape, _convert_dtype(dtype)))


@logger.catch
def empty_like(prototype: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating empty_like array prototype={prototype}, dtype={dtype}")
    return ndarray(_empty_like(prototype, _convert_dtype(dtype)))


@logger.catch
def eye(n: int, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating eye array n={n}, dtype={dtype}")
    return ndarray(_eye(n, _convert_dtype(dtype)))


@logger.catch
def ones(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating ones array shape={shape}, dtype={dtype}")
    return ndarray(_ones(shape, _convert_dtype(dtype)))


@logger.catch
def ones_like(other: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating ones_like array other={other}, dtype={dtype}")
    return ndarray(_ones_like(other, _convert_dtype(dtype)))


@logger.catch
def identity(n: int, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating identity array n={n}, dtype={dtype}")
    return ndarray(_identity(n, _convert_dtype(dtype)))


@logger.catch
def linspace(
    start: ScalarLike,
    end: ScalarLike,
    steps: int = 50,
    dtype: DTypeLike = None,
) -> ndarray:
    logger.debug(f"Creating linspace array start={start}, end={end}, steps={steps}, dtype={dtype}")
    return ndarray(_linspace(start, end, steps, _convert_dtype(dtype)))


@logger.catch
def array(obj, dtype=None, copy=True, order='K', subok=False, ndmin=0) -> ndarray:
    """Create an NPU array from a Python object.

    Parameters
    ----------
    obj : array_like
        Input data: list, tuple, scalar, ndarray, or np.ndarray.
    dtype : dtype, optional
        Desired data type for the array.
    copy : bool, default True
        If True, always copy the data. If False, reuse when possible.
    order : {'K', 'A', 'C', 'F'}, default 'K'
        Memory layout order. Only 'C' (row-major) is fully supported on NPU.
    subok : bool, default False
        If True, subclasses are preserved.
    ndmin : int, default 0
        Minimum number of dimensions.

    Returns
    -------
    ndarray
        NPU array with the given data.
    """
    logger.debug(
        f"Creating array from {type(obj).__name__}, "
        f"dtype={dtype}, copy={copy}, ndmin={ndmin}"
    )

    # Already an asnumpy ndarray
    if isinstance(obj, ndarray):
        if dtype is not None and obj.dtype != np.dtype(dtype):
            np_arr = obj.to_numpy().astype(dtype)
            return ndarray.from_numpy(np_arr)
        if copy:
            np_arr = obj.to_numpy().copy()
            return ndarray.from_numpy(np_arr)
        return obj

    # numpy ndarray: transfer to NPU
    if isinstance(obj, np.ndarray):
        if dtype is not None:
            obj = obj.astype(dtype)
        result = ndarray.from_numpy(obj)
    else:
        # Python list/tuple/scalar -> numpy -> NPU
        np_arr = np.array(obj, dtype=dtype, order=order, ndmin=ndmin)
        result = ndarray.from_numpy(np_arr)

    # Handle ndmin: prepend dimensions if needed
    if ndmin > 0 and hasattr(result, 'ndim') and result.ndim < ndmin:
        np_arr = result.to_numpy()
        shape = (1,) * (ndmin - result.ndim) + np_arr.shape
        np_arr = np_arr.reshape(shape)
        result = ndarray.from_numpy(np_arr)

    return result


@logger.catch
def asarray(obj, dtype=None, order=None) -> ndarray:
    """Convert input to an NPU array without unnecessary copying.

    Parameters
    ----------
    obj : array_like
        Input data to convert.
    dtype : dtype, optional
        Desired data type. If None, preserve the existing dtype.
    order : {'K', 'A', 'C', 'F'}, default None
        Memory layout order.

    Returns
    -------
    ndarray
        NPU array. If obj is already an ndarray with matching dtype,
        returns obj directly without copying.
    """
    logger.debug(f"asarray from {type(obj).__name__}, dtype={dtype}")
    if isinstance(obj, ndarray):
        if dtype is not None and obj.dtype != np.dtype(dtype):
            np_arr = obj.to_numpy().astype(dtype)
            return ndarray.from_numpy(np_arr)
        return obj
    return array(obj, dtype=dtype, copy=False, order=order)


@logger.catch
def asanyarray(obj, dtype=None, order=None) -> ndarray:
    """Convert input to an NPU array, preserving subclasses.

    Parameters
    ----------
    obj : array_like
        Input data to convert.
    dtype : dtype, optional
        Desired data type.
    order : {'K', 'A', 'C', 'F'}, default None
        Memory layout order.

    Returns
    -------
    ndarray
        NPU array.
    """
    logger.debug(f"asanyarray from {type(obj).__name__}, dtype={dtype}")
    return asarray(obj, dtype=dtype, order=order)


@logger.catch
def copy(obj, order='K') -> ndarray:
    """Return an array copy of the given object.

    Parameters
    ----------
    obj : array_like
        Input data.
    order : {'K', 'A', 'C', 'F'}, default 'K'
        Memory layout order for the copy.

    Returns
    -------
    ndarray
        A new NPU array that is a copy of the input.
    """
    logger.debug(f"Copying {type(obj).__name__}, order={order}")
    if isinstance(obj, ndarray):
        np_arr = obj.to_numpy().copy(order=order)
        return ndarray.from_numpy(np_arr)
    elif isinstance(obj, np.ndarray):
        return ndarray.from_numpy(obj.copy(order=order))
    else:
        return array(obj, order=order)

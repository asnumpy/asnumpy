# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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


def zeros(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    """
    Create an array initialized with zero values.

    This function allocates a new :class:`asnumpy.ndarray` with the specified
    shape and fills all elements with zeros. The array is created on the
    current execution device used by asnumpy.

    Arguments
    ---------
    shape : int or sequence of int
        Specifies the dimensions of the output array. A single integer
        creates a one-dimensional array, while a sequence defines a
        multi-dimensional shape.
    dtype : data-type, optional
        Data type of the returned array. If not provided, the default
        numeric type is used.

    Returns
    -------
    ndarray
        An array whose elements are all set to zero and whose shape matches
        the given ``shape`` argument.

    See Also
    --------
    numpy.zeros : NumPy equivalent for creating an array of zeros.

    Notes
    -----
    - The returned array is allocated by asnumpy and may reside on an
      accelerator device depending on the runtime configuration.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.zeros(3)
    array([0., 0., 0.])
    >>> ap.zeros((2, 2), dtype=int)
    array([[0, 0],
           [0, 0]])
    """
    return ndarray(_zeros(shape, _convert_dtype(dtype)))


def zeros_like(other: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Create an array of zeros with the same shape as an existing array.

    This function returns a new :class:`asnumpy.ndarray` whose shape matches
    that of the input object. All elements of the returned array are
    initialized to zero. By default, the data type is inferred from the
    input unless explicitly overridden.

    Arguments
    ---------
    other : array_like
        Reference object that provides the shape of the output array.
    dtype : data-type, optional
        Data type of the returned array. If specified, it overrides the
        data type inferred from ``other``.

    Returns
    -------
    ndarray
        An array filled with zeros and having the same shape as ``other``.

    See Also
    --------
    zeros
    numpy.zeros_like

    Notes
    -----
    - The returned array is allocated by asnumpy and may be placed on an
      accelerator device depending on the current runtime configuration.

    Examples
    --------
    >>> import asnumpy as ap
    >>> x = ap.arange(4).reshape(2, 2)
    >>> ap.zeros_like(x)
    array([[0, 0],
           [0, 0]])
    """
    return ndarray(_zeros_like(other, _convert_dtype(dtype)))


def full(shape: ShapeLike, value: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    """
    Create an array filled with a specific value.

    Returns a new asnumpy.ndarray where all elements are set to `value`.
    Shape and dtype are controlled by the `shape` and `dtype` parameters.

    Arguments
    ---------
    shape : int or sequence of ints
        Shape of the output array.
    value : scalar
        Value to fill the array with.
    dtype : data-type, optional
        Desired data type of the array.

    Returns
    -------
    asnumpy.ndarray
        Array filled with the specified value.

    See Also
    --------
    full_like : Create an array filled with value matching another array.
    zeros : Create an array of zeros.
    ones : Create an array of ones.
    empty : Create an uninitialized array.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.full((2, 2), 7)
    array([[7, 7],
           [7, 7]])
    >>> ap.full((2, 3), 3.5)
    array([[3.5, 3.5, 3.5],
           [3.5, 3.5, 3.5]])
    """
    return ndarray(_full(shape, value, _convert_dtype(dtype)))


def full_like(other: ArrayLike, value: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    """
    Create an array filled with a specific value, matching another array's shape.

    Arguments
    ---------
    other : array_like
        Array whose shape is used for the output.
    value : scalar
        Value to fill the array.
    dtype : data-type, optional
        Desired data type of the output array.

    Returns
    -------
    asnumpy.ndarray
        Array filled with `value` matching the shape of `other`.

    See Also
    --------
    full : Create a full array from shape.
    zeros_like : Create a zeros array matching another array.
    ones_like : Create an array of ones matching another array.
    empty_like : Create an uninitialized array matching another array.

    Examples
    --------
    >>> import asnumpy as ap
    >>> x = ap.arange(4)
    >>> ap.full_like(x, 9)
    array([9, 9, 9, 9])
    """
    return ndarray(_full_like(other, value, _convert_dtype(dtype)))


def empty(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    """
    Create an uninitialized array.

    Returns a new asnumpy.ndarray with arbitrary content.
    The array shape and dtype are determined by the parameters.

    Arguments
    ---------
    shape : int or sequence of ints
        Shape of the output array.
    dtype : data-type, optional
        Desired data type of the array.

    Returns
    -------
    asnumpy.ndarray
        Array with uninitialized values (may contain random memory data).

    See Also
    --------
    empty_like : Create an uninitialized array matching another array.
    zeros : Create a zeros array.
    ones : Create a ones array.
    full : Create a full array with specified value.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.empty((2, 2))
    array([[... , ...],
           [... , ...]])  # values arbitrary
    """
    return ndarray(_empty(shape, _convert_dtype(dtype)))


def empty_like(prototype: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Create an uninitialized array matching another array's shape.

    Arguments
    ---------
    prototype : array_like
        Array whose shape is used for the output.
    dtype : data-type, optional
        Desired data type of the output array.

    Returns
    -------
    asnumpy.ndarray
        Array with uninitialized values and same shape as `prototype`.

    See Also
    --------
    zeros_like : Create a zeros array matching another array.
    ones_like : Create an array of ones matching another array.
    full_like : Create a full array with specified value.
    empty : Create an uninitialized array from shape.

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([[1, 2], [3, 4]])
    >>> ap.empty_like(a)
    array([[... , ...],
           [... , ...]])  # values arbitrary
    """
    return ndarray(_empty_like(prototype, _convert_dtype(dtype)))


def eye(n: int, dtype: DTypeLike = None) -> ndarray:
    """
    Create a 2-D identity matrix.

    Returns a square asnumpy.ndarray with ones on the main diagonal
    and zeros elsewhere.

    Arguments
    ---------
    n : int
        Number of rows (and columns) of the matrix.
    dtype : data-type, optional
        Desired data type of the matrix.

    Returns
    -------
    asnumpy.ndarray
        Identity matrix of shape (n, n).

    See Also
    --------
    identity : Equivalent function to create identity array.
    diag : Extract or create diagonal arrays.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.eye(3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    return ndarray(_eye(n, _convert_dtype(dtype)))


def ones(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    """
    Create an array filled with ones.

    Arguments
    ---------
    shape : int or sequence of ints
        Shape of the output array.
    dtype : data-type, optional
        Desired data type of the array.

    Returns
    -------
    asnumpy.ndarray
        Array of ones with specified shape and dtype.

    See Also
    --------
    ones_like : Create an array of ones matching another array.
    zeros : Create a zeros array.
    full : Create a full array with specified value.
    empty : Create an uninitialized array.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.ones(4)
    array([1., 1., 1., 1.])
    """
    return ndarray(_ones(shape, _convert_dtype(dtype)))


def ones_like(other: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Create an array of ones matching another array's shape.

    Arguments
    ---------
    other : array_like
        Array whose shape is used for output.
    dtype : data-type, optional
        Desired data type of the output array.

    Returns
    -------
    asnumpy.ndarray
        Array of ones with same shape (and optionally dtype) as `other`.

    See Also
    --------
    zeros_like : Create a zeros array matching another array.
    full_like : Create a full array with specified value.
    empty_like : Create an uninitialized array.
    ones : Create an array of ones from shape.

    Examples
    --------
    >>> import asnumpy as ap
    >>> x = ap.arange(6).reshape((2, 3))
    >>> ap.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])
    """
    return ndarray(_ones_like(other, _convert_dtype(dtype)))


def identity(n: int, dtype: DTypeLike = None) -> ndarray:
    """
    Create a square identity matrix.

    Arguments
    ---------
    n : int
        Number of rows and columns.
    dtype : data-type, optional
        Desired data type of the matrix.

    Returns
    -------
    asnumpy.ndarray
        Identity matrix of shape (n, n).

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.identity(3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    return ndarray(_identity(n, _convert_dtype(dtype)))


def linspace(
    start: ScalarLike,
    end: ScalarLike,
    steps: int = 50,
    dtype: DTypeLike = None,
) -> ndarray:
    """
    Generate evenly spaced samples over an interval.

    Arguments
    ---------
    start : int or float
        Starting value of the sequence.
    end : int or float
        End value of the sequence.
    steps : int, optional
        Number of samples to generate. Default is 50.
    dtype : data-type, optional
        Desired data type of the output array.

    Returns
    -------
    asnumpy.ndarray
        Array of evenly spaced samples between start and end.

    See Also
    --------
    arange : Generate values with a fixed step size.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.linspace(0, 1, 5)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    return ndarray(_linspace(start, end, steps, _convert_dtype(dtype)))

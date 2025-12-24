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
from .lib.asnumpy_core.array import (
    empty as _ap_empty,
    empty_like as _ap_empty_like,
    eye as _ap_eye,
    full as _ap_full,
    full_like as _ap_full_like,
    identity as _ap_identity,
    linspace as _ap_linspace,
    ones as _ap_ones,
    ones_like as _ap_ones_like,
    zeros as _ap_zeros,
    zeros_like as _ap_zeros_like,
)
from .utils import ndarray, _convert_dtype


def zeros(
    shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array.

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.zeros(5)
    array([ 0.,  0.,  0.,  0.,  0.])
    >>> ap.zeros((5,), dtype=int)
    array([0, 0, 0, 0, 0])
    >>> ap.zeros((2, 1))
    array([[ 0.],
           [ 0.]])
    """
    return ndarray(_ap_zeros(shape, _convert_dtype(dtype)))


def zeros_like(other: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    other : array_like
        The shape and data-type of `other` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data-type of the result.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `other`.

    See Also
    --------
    zeros : Return a new array setting values to zero.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.

    Examples
    --------
    >>> import asnumpy as ap
    >>> x = ap.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> ap.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])
    """
    return ndarray(_ap_zeros_like(other, _convert_dtype(dtype)))


def full(
    shape: Union[int, Sequence[int]], value: Any, dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array. 

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    full_like : Return a new array with shape of input filled with value.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.full((2, 2), 10)
    array([[10, 10],
           [10, 10]])
    >>> ap.full((2, 2), [1, 2])
    array([[1, 2],
           [1, 2]])
    """
    return ndarray(_ap_full(shape, value, _convert_dtype(dtype)))


def full_like(other: Any, value: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    other : array_like
        The shape and data-type of `other` define these same attributes of
        the returned array.
    value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data-type of the result.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `other`.

    See Also
    --------
    full : Return a new array of given shape filled with value.
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.

    Examples
    --------
    >>> import asnumpy as ap
    >>> x = ap.arange(6, dtype=int)
    >>> ap.full_like(x, 1)
    array([1, 1, 1, 1, 1, 1])
    """
    return ndarray(_ap_full_like(other, value, _convert_dtype(dtype)))


def empty(
    shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape, dtype, and
        order.  Object arrays will be initialized to None.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.empty([2, 2])
    array([[ -9.74499359e+001,   6.69583040e-309],
           [  2.13182611e-314,   3.06959433e-309]])         #random
    >>> ap.empty([2, 2], dtype=int)
    array([[-1073741821, -1067949133],
           [  496041986,    19249760]])                     #random
    """
    return ndarray(_ap_empty(shape, _convert_dtype(dtype)))


def empty_like(prototype: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : array_like
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data-type of the result.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same shape and type
        as `prototype`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> ap.empty_like(a)
    array([[-1073741821, -1067949133,   496041986],
           [   19249760, -1073741821, -1067949133]])    #random
    >>> a = ap.array([[1., 2., 3.],[4., 5., 6.]])
    >>> ap.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000],
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]]) #random
    """
    return ndarray(_ap_empty_like(prototype, _convert_dtype(dtype)))


def eye(n: int, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
      Number of rows in the output.
    dtype : data-type, optional
      Data-type of the returned array.

    Returns
    -------
    I : ndarray of shape (N,M)
      An array where all elements are equal to zero, except for the $k$-th
      diagonal, whose values are equal to one.

    See Also
    --------
    identity : (almost) equivalent function
    diag : diagonal 2-D array from a 1-D array specified by the user.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])
    >>> ap.eye(3)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])
    """
    return ndarray(_ap_eye(n, _convert_dtype(dtype)))


def ones(shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array.

    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and order.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.ones(5)
    array([1., 1., 1., 1., 1.])
    >>> ap.ones((5,), dtype=int)
    array([1, 1, 1, 1, 1])
    >>> ap.ones((2, 1))
    array([[1.],
           [1.]])
    """
    return ndarray(_ap_ones(shape, _convert_dtype(dtype)))


def ones_like(other: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    other : array_like
        The shape and data-type of `other` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data-type of the result.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `other`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    ones : Return a new array setting values to one.

    Examples
    --------
    >>> import asnumpy as ap
    >>> x = ap.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> ap.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])
    >>> y = ap.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> ap.ones_like(y)
    array([1.,  1.,  1.])
    """
    return ndarray(_ap_ones_like(other, _convert_dtype(dtype)))


def identity(n: int, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return the identity array.

    The identity array is a square array with ones on the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.identity(3)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])
    """
    return ndarray(_ap_identity(n, _convert_dtype(dtype)))


def linspace(
    start: Union[int, float],
    end: Union[int, float],
    steps: int = 50,
    dtype: Optional[np.dtype] = None,
) -> ndarray:
    """
    Return evenly spaced numbers over a specified interval.

    Generate `steps` evenly spaced samples over the interval
    [`start`, `end`].

    Parameters
    ----------
    start : float or int
        The starting value of the sequence.
    end : float or int
        The end value of the sequence.
    steps : int, optional
        Number of samples to generate. Default is 50.
    dtype : dtype, optional
        The type of the output array. If not specified, the dtype is
        inferred from the inputs.

    Returns
    -------
    samples : ndarray
        An array of evenly spaced samples.

    See Also
    --------
    arange
    """
    return ndarray(_ap_linspace(start, end, steps, _convert_dtype(dtype)))

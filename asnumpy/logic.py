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

from ._types import ArrayLike, DTypeLike, AxisLike
from .lib.asnumpy_core.logic import (
    all as _all,
    any as _any,
    equal as _equal,
    greater as _greater,
    greater_equal as _greater_equal,
    isfinite as _isfinite,
    isinf as _isinf,
    isneginf as _isneginf,
    isposinf as _isposinf,
    less as _less,
    less_equal as _less_equal,
    logical_and as _logical_and,
    logical_not as _logical_not,
    logical_or as _logical_or,
    logical_xor as _logical_xor,
    not_equal as _not_equal,
)
from .utils import ndarray, _convert_dtype


def all(x: ArrayLike, axis: AxisLike = None, keepdims: bool = False) -> ndarray:
    """
    Test if all elements evaluate to True.

    Checks whether all elements in the input array `x` evaluate to True.
    This check can be performed over the entire array or along a specified axis.

    Arguments
    ---------
    x : asnumpy.ndarray
        The input array to be checked.
    axis : int or sequence of ints, optional
        The axis or axes along which to perform the logical AND reduction.
        If None, the reduction is performed over all dimensions.
        Negative values count from the last axis.
    keepdims : bool, optional
        If True, the reduced axes are retained in the result as dimensions with size one.
        This allows the result to broadcast correctly against the input array.

    Returns
    -------
    asnumpy.ndarray
        A boolean array or scalar indicating whether all elements evaluate to True.
        If `axis` is None, a scalar boolean is returned.
        Otherwise, an array of booleans is returned.

    See Also
    --------
    numpy.all
    asnumpy.any

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.all(ap.array([True, True, True]))
    array(True)
    >>> ap.all(ap.array([True, False, True]))
    array(False)
    >>> mat = ap.array([[True, False], [True, True]])
    >>> ap.all(mat, axis=0)
    array([ True, False])
    """
    if axis is None:
        return ndarray(_all(x))
    return ndarray(_all(x, axis, keepdims))


def any(x: ArrayLike, axis: AxisLike = None, keepdims: bool = False) -> ndarray:
    """
    Test if any element evaluates to True.

    Checks whether any element in the input array `x` evaluates to True.
    This check can be performed over the entire array or along a specified axis.

    Arguments
    ---------
    x : asnumpy.ndarray
        The input array to be checked.
    axis : int or sequence of ints, optional
        The axis or axes along which to perform the logical OR reduction.
        If None, the reduction is performed over all dimensions.
        Negative values count from the last axis.
    keepdims : bool, optional
        If True, the reduced axes are retained in the result as dimensions with size one.
        This allows the result to broadcast correctly against the input array.

    Returns
    -------
    asnumpy.ndarray
        A boolean array or scalar indicating whether any element evaluates to True.
        If `axis` is None, a scalar boolean is returned.
        Otherwise, an array of booleans is returned.

    See Also
    --------
    numpy.any
    asnumpy.all

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.any(ap.array([False, False, True]))
    array(True)
    >>> ap.any(ap.array([False, False, False]))
    array(False)
    >>> mat = ap.array([[True, False], [False, False]])
    >>> ap.any(mat, axis=1)
    array([ True, False])
    """
    if axis is None:
        return ndarray(_any(x))
    return ndarray(_any(x, axis, keepdims))


def isfinite(x: ArrayLike) -> ndarray:
    """
    Test for finiteness element-wise.

    Determines whether each element of the input array `x` is finite.
    An element is considered finite if it is not positive infinity, negative infinity, or NaN (Not a Number).

    Arguments
    ---------
    x : asnumpy.ndarray
        The input array to be tested.

    Returns
    -------
    asnumpy.ndarray
        A boolean array with the same shape as `x`.
        True where the element is finite, False otherwise.

    See Also
    --------
    numpy.isfinite
    asnumpy.isinf
    asnumpy.isneginf
    asnumpy.isposinf
    asnumpy.isnan

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.isfinite(ap.array([1, np.inf, -np.inf, np.nan]))
    array([ True, False, False, False])
    """
    return ndarray(_isfinite(x))


def isinf(x: ArrayLike) -> ndarray:
    """
    Test for infinity element-wise.

    Determines whether each element of the input array `x` is positive or negative infinity.

    Arguments
    ---------
    x : asnumpy.ndarray
        The input array to be tested.

    Returns
    -------
    asnumpy.ndarray
        A boolean array with the same shape as `x`.
        True where the element is positive or negative infinity, False otherwise.

    See Also
    --------
    numpy.isinf
    asnumpy.isneginf
    asnumpy.isposinf
    asnumpy.isnan
    asnumpy.isfinite

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.isinf(ap.array([1, np.inf, -np.inf, np.nan]))
    array([False,  True,  True, False])
    """
    return ndarray(_isinf(x))


def isneginf(x: ArrayLike) -> ndarray:
    """
    Test for negative infinity element-wise.

    Determines whether each element of the input array `x` is negative infinity.

    Arguments
    ---------
    x : asnumpy.ndarray
        The input array to be tested.

    Returns
    -------
    asnumpy.ndarray
        A boolean array with the same shape as `x`.
        True where the element is negative infinity, False otherwise.

    See Also
    --------
    numpy.isneginf
    asnumpy.isinf
    asnumpy.isposinf
    asnumpy.isnan
    asnumpy.isfinite

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.isneginf(ap.array([1, np.inf, -np.inf, np.nan]))
    array([False, False,  True, False])
    """
    return ndarray(_isneginf(x))


def isposinf(x: ArrayLike) -> ndarray:
    """
    Test for positive infinity element-wise.

    Determines whether each element of the input array `x` is positive infinity.

    Arguments
    ---------
    x : asnumpy.ndarray
        The input array to be tested.

    Returns
    -------
    asnumpy.ndarray
        A boolean array with the same shape as `x`.
        True where the element is positive infinity, False otherwise.

    See Also
    --------
    numpy.isposinf
    asnumpy.isinf
    asnumpy.isneginf
    asnumpy.isnan
    asnumpy.isfinite

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.isposinf(ap.array([1, np.inf, -np.inf, np.nan]))
    array([False,  True, False, False])
    """
    return ndarray(_isposinf(x))


def logical_and(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Compute the logical AND of two arrays element-wise.

    Performs the logical AND operation on the elements of `x1` and `x2`.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        The first input array.
    x2 : asnumpy.ndarray
        The second input array.

    Returns
    -------
    asnumpy.ndarray
        A boolean array containing the result of the logical AND operation.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.logical_and
    asnumpy.logical_or
    asnumpy.logical_not
    asnumpy.logical_xor
    asnumpy.bitwise_and

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.logical_and(True, False)
    array(False)
    >>> ap.logical_and([True, False], [True, True])
    array([ True, False])
    """
    return ndarray(_logical_and(x1, x2))


def logical_or(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Compute the logical OR of two arrays element-wise.

    Performs the logical OR operation on the elements of `x1` and `x2`.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        The first input array.
    x2 : asnumpy.ndarray
        The second input array.

    Returns
    -------
    asnumpy.ndarray
        A boolean array containing the result of the logical OR operation.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.logical_or
    asnumpy.logical_and
    asnumpy.logical_not
    asnumpy.logical_xor
    asnumpy.bitwise_or

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.logical_or(True, False)
    array(True)
    >>> ap.logical_or([True, False], [False, False])
    array([ True, False])
    """
    return ndarray(_logical_or(x1, x2))


def logical_not(x: ArrayLike) -> ndarray:
    """
    Compute the logical NOT of an array element-wise.

    Performs the logical NOT operation on the elements of `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        The input array.

    Returns
    -------
    asnumpy.ndarray
        A boolean array containing the result of the logical NOT operation.
        The shape is the same as `x`.

    See Also
    --------
    numpy.logical_not
    asnumpy.logical_and
    asnumpy.logical_or
    asnumpy.logical_xor
    asnumpy.bitwise_not

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.logical_not(True)
    array(False)
    >>> ap.logical_not([True, False])
    array([False,  True])
    """
    return ndarray(_logical_not(x))


def logical_xor(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Compute the logical XOR of two arrays element-wise.

    Performs the logical XOR (exclusive OR) operation on the elements of `x1` and `x2`.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        The first input array.
    x2 : asnumpy.ndarray
        The second input array.

    Returns
    -------
    asnumpy.ndarray
        A boolean array containing the result of the logical XOR operation.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.logical_xor
    asnumpy.logical_and
    asnumpy.logical_or
    asnumpy.logical_not
    asnumpy.bitwise_xor

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.logical_xor(True, False)
    array(True)
    >>> ap.logical_xor([True, True, False, False], [True, False, True, False])
    array([False,  True,  True, False])
    """
    return ndarray(_logical_xor(x1, x2))


def greater(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Return the truth value of (x1 > x2) element-wise.

    Compares two arrays element-wise and returns True where `x1` is greater than `x2`, and False otherwise.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : array_like
        First input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x2.
    x2 : array_like
        Second input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x1.
    dtype : data-type, optional
        The desired data type for the output array.
        If not specified, the output is a boolean array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the result of the element-wise comparison.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.greater
    asnumpy.greater_equal
    asnumpy.less
    asnumpy.less_equal
    asnumpy.equal
    asnumpy.not_equal

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.greater([4, 2], [2, 2])
    array([ True, False])
    >>> ap.greater([1, 2], [3, 1])
    array([False,  True])
    """
    return ndarray(_greater(x1, x2, _convert_dtype(dtype)))


def greater_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Return the truth value of (x1 >= x2) element-wise.

    Compares two arrays element-wise and returns True where `x1` is greater than or equal to `x2`, and False otherwise.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : array_like
        First input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x2.
    x2 : array_like
        Second input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x1.
    dtype : data-type, optional
        The desired data type for the output array.
        If not specified, the output is a boolean array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the result of the element-wise comparison.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.greater_equal
    asnumpy.greater
    asnumpy.less
    asnumpy.less_equal
    asnumpy.equal
    asnumpy.not_equal

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.greater_equal([4, 2, 1], [2, 2, 2])
    array([ True,  True, False])
    """
    return ndarray(_greater_equal(x1, x2, _convert_dtype(dtype)))


def less(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Return the truth value of (x1 < x2) element-wise.

    Compares two arrays element-wise and returns True where `x1` is less than `x2`, and False otherwise.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : array_like
        First input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x2.
    x2 : array_like
        Second input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x1.
    dtype : data-type, optional
        The desired data type for the output array.
        If not specified, the output is a boolean array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the result of the element-wise comparison.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.less
    asnumpy.greater
    asnumpy.greater_equal
    asnumpy.less_equal
    asnumpy.equal
    asnumpy.not_equal

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.less([1, 2], [2, 2])
    array([ True, False])
    """
    return ndarray(_less(x1, x2, _convert_dtype(dtype)))


def less_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Return the truth value of (x1 <= x2) element-wise.

    Compares two arrays element-wise and returns True where `x1` is less than or equal to `x2`, and False otherwise.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : array_like
        First input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x2.
    x2 : array_like
        Second input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x1.
    dtype : data-type, optional
        The desired data type for the output array.
        If not specified, the output is a boolean array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the result of the element-wise comparison.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.less_equal
    asnumpy.greater
    asnumpy.greater_equal
    asnumpy.less
    asnumpy.equal
    asnumpy.not_equal

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.less_equal([4, 2, 1], [2, 2, 2])
    array([False,  True,  True])
    """
    return ndarray(_less_equal(x1, x2, _convert_dtype(dtype)))


def equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Return (x1 == x2) element-wise.

    Compares two arrays element-wise and returns True where `x1` is equal to `x2`, and False otherwise.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : array_like
        First input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x2.
    x2 : array_like
        Second input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x1.
    dtype : data-type, optional
        The desired data type for the output array.
        If not specified, the output is a boolean array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the result of the element-wise comparison.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.equal
    asnumpy.not_equal
    asnumpy.greater_equal
    asnumpy.less_equal
    asnumpy.greater
    asnumpy.less

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.equal([1, 2], [1, 3])
    array([ True, False])
    """
    return ndarray(_equal(x1, x2, _convert_dtype(dtype)))


def not_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Return (x1 != x2) element-wise.

    Compares two arrays element-wise and returns True where `x1` is not equal to `x2`, and False otherwise.
    The input arrays must be broadcastable to a common shape.

    Arguments
    ---------
    x1 : array_like
        First input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x2.
    x2 : array_like
        Second input. Can be an asnumpy.ndarray, Python scalar, or any object broadcastable to the shape of x1.
    dtype : data-type, optional
        The desired data type for the output array.
        If not specified, the output is a boolean array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the result of the element-wise comparison.
        The shape is determined by broadcasting `x1` and `x2`.

    See Also
    --------
    numpy.not_equal
    asnumpy.equal
    asnumpy.greater_equal
    asnumpy.less_equal
    asnumpy.greater
    asnumpy.less

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.not_equal([1, 2], [1, 3])
    array([False,  True])
    """
    return ndarray(_not_equal(x1, x2, _convert_dtype(dtype)))

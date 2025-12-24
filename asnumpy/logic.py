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

from typing import Optional, Union, Sequence, Any
import numpy as np
from .lib.asnumpy_core.logic import (
    all as _ap_all,
    any as _ap_any,
    equal as _ap_equal,
    greater as _ap_greater,
    greater_equal as _ap_greater_equal,
    isfinite as _ap_isfinite,
    isinf as _ap_isinf,
    isneginf as _ap_isneginf,
    isposinf as _ap_isposinf,
    less as _ap_less,
    less_equal as _ap_less_equal,
    logical_and as _ap_logical_and,
    logical_not as _ap_logical_not,
    logical_or as _ap_logical_or,
    logical_xor as _ap_logical_xor,
    not_equal as _ap_not_equal,
)
from .utils import ndarray, _convert_dtype


def all(
    x: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> ndarray:
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    x : ndarray
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (axis=None) is to perform a logical AND over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    out : ndarray
        A new boolean or array is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    numpy.all
    """
    if axis is None:
        return ndarray(_ap_all(x))
    return ndarray(_ap_all(x, axis, keepdims))


def any(
    x: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> ndarray:
    """
    Test whether any array elements along a given axis evaluate to True.

    Parameters
    ----------
    x : ndarray
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (axis=None) is to perform a logical OR over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    out : ndarray
        A new boolean or array is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    numpy.any
    """
    if axis is None:
        return ndarray(_ap_any(x))
    return ndarray(_ap_any(x, axis, keepdims))


def isfinite(x: ndarray) -> ndarray:
    """
    Test element-wise for finiteness (not infinity and not Not a Number).

    The result is returned as a boolean array.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    y : ndarray, bool
        True where ``x`` is not positive infinity, negative infinity,
        or NaN; false otherwise.

    See Also
    --------
    isinf, isneginf, isposinf, isnan
    numpy.isfinite
    """
    return ndarray(_ap_isfinite(x))


def isinf(x: ndarray) -> ndarray:
    """
    Test element-wise for positive or negative infinity.

    Returns a boolean array of the same shape as `x`, true where ``x == +/-inf``,
    otherwise false.

    Parameters
    ----------
    x : ndarray
        Input values

    Returns
    -------
    y : ndarray, bool
        For scalar input, the result is a new boolean with value True if
        the input is positive or negative infinity; otherwise the value is
        False.

        For array input, the result is a boolean array with the same shape
        as the input and the values are True where the corresponding
        element of the input is positive or negative infinity; otherwise
        the values are False.

    See Also
    --------
    isneginf, isposinf, isnan, isfinite
    numpy.isinf
    """
    return ndarray(_ap_isinf(x))


def isneginf(x: ndarray) -> ndarray:
    """
    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : ndarray
        The input array.

    Returns
    -------
    y : ndarray, bool
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a numpy boolean array is
        returned with values True where the corresponding element of the
        input is negative infinity and values False where the element of
        the input is not negative infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as
        zeros and ones, if the type is boolean then as False and True.
        The return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isposinf, isnan, isfinite
    numpy.isneginf
    """
    return ndarray(_ap_isneginf(x))


def isposinf(x: ndarray) -> ndarray:
    """
    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : ndarray
        The input array.

    Returns
    -------
    y : ndarray, bool
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a boolean array is returned
        with values True where the corresponding element of the input is
        positive infinity and values False where the element of the input is
        not positive infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as
        zeros and ones, if the type is boolean then as False and True.
        The return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isneginf, isnan, isfinite
    numpy.isposinf
    """
    return ndarray(_ap_isposinf(x))


def logical_and(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Compute the truth value of x1 AND x2 element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.

    Returns
    -------
    y : ndarray, bool
        Boolean result of the logical AND operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    See Also
    --------
    logical_or, logical_not, logical_xor
    bitwise_and
    numpy.logical_and
    """
    return ndarray(_ap_logical_and(x1, x2))


def logical_or(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Compute the truth value of x1 OR x2 element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.

    Returns
    -------
    y : ndarray, bool
        Boolean result of the logical OR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    See Also
    --------
    logical_and, logical_not, logical_xor
    bitwise_or
    numpy.logical_or
    """
    return ndarray(_ap_logical_or(x1, x2))


def logical_not(x: ndarray) -> ndarray:
    """
    Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    y : ndarray, bool
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.

    See Also
    --------
    logical_and, logical_or, logical_xor
    bitwise_not
    numpy.logical_not
    """
    return ndarray(_ap_logical_not(x))


def logical_xor(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Compute the truth value of x1 XOR x2, element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.

    Returns
    -------
    y : ndarray, bool
        Boolean result of the logical XOR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    See Also
    --------
    logical_and, logical_or, logical_not
    bitwise_xor
    numpy.logical_xor
    """
    return ndarray(_ap_logical_xor(x1, x2))


def greater(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the truth value of (x1 > x2) element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.

    dtype : data-type, optional
        Desired data type for the output array. If not specified, defaults to boolean.

    Returns
    -------
    out : ndarray, bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    greater_equal, less, less_equal, equal, not_equal
    numpy.greater
    """
    return ndarray(_ap_greater(x1, x2, _convert_dtype(dtype)))


def greater_equal(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the truth value of (x1 >= x2) element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.

    dtype : data-type, optional
        Desired data type for the output array. If not specified, defaults to boolean.

    Returns
    -------
    out : ndarray, bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    greater, less, less_equal, equal, not_equal
    numpy.greater_equal
    """
    return ndarray(_ap_greater_equal(x1, x2, _convert_dtype(dtype)))


def less(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the truth value of (x1 < x2) element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.
    
    dtype : data-type, optional
        Desired data type for the output array. If not specified, defaults to boolean.
    
    Returns
    -------
    out : ndarray, bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    greater, greater_equal, less_equal, equal, not_equal
    numpy.less
    """
    return ndarray(_ap_less(x1, x2, _convert_dtype(dtype)))


def less_equal(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the truth value of (x1 <= x2) element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.
    
    dtype : data-type, optional
        Desired data type for the output array. If not specified, defaults to boolean.
    
    Returns
    -------
    out : ndarray, bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    greater, greater_equal, less, equal, not_equal
    numpy.less_equal
    """
    return ndarray(_ap_less_equal(x1, x2, _convert_dtype(dtype)))


def equal(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return (x1 == x2) element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.
    
    dtype : data-type, optional
        Desired data type for the output array. If not specified, defaults to boolean.

    Returns
    -------
    out : ndarray, bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less
    numpy.equal
    """
    return ndarray(_ap_equal(x1, x2, _convert_dtype(dtype)))


def not_equal(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return (x1 != x2) element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays. `x1` and `x2` must be broadcastable to the same shape.
    
    dtype : data-type, optional
        Desired data type for the output array. If not specified, defaults to boolean.
        
    Returns
    -------
    out : ndarray, bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    equal, greater_equal, less_equal, greater, less
    numpy.not_equal
    """
    return ndarray(_ap_not_equal(x1, x2, _convert_dtype(dtype)))

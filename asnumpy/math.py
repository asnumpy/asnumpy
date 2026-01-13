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

from typing import Optional, Union
from ._types import ArrayLike, AxisOptional, DTypeLike
from .lib.asnumpy_core.math import (
    absolute as _absolute,
    add as _add,
    amax as _amax,
    amin as _amin,
    around as _around,
    arccos as _arccos,
    arccosh as _arccosh,
    arcsin as _arcsin,
    arcsinh as _arcsinh,
    arctan as _arctan,
    arctan2 as _arctan2,
    arctanh as _arctanh,
    ceil as _ceil,
    clip as _clip,
    copysign as _copysign,
    cos as _cos,
    cosh as _cosh,
    cross as _cross,
    cumprod as _cumprod,
    cumsum as _cumsum,
    degrees as _degrees,
    divide as _divide,
    divmod as _divmod,
    exp as _exp,
    exp2 as _exp2,
    expm1 as _expm1,
    fabs as _fabs,
    fix as _fix,
    float_power as _float_power,
    floor as _floor,
    floor_divide as _floor_divide,
    fmax as _fmax,
    fmin as _fmin,
    fmod as _fmod,
    gcd as _gcd,
    gelu as _gelu,
    heaviside as _heaviside,
    hypot as _hypot,
    lcm as _lcm,
    ldexp as _ldexp,
    log as _log,
    log10 as _log10,
    log1p as _log1p,
    log2 as _log2,
    logaddexp as _logaddexp,
    logaddexp2 as _logaddexp2,
    max as _max,
    maximum as _maximum,
    min as _min,
    minimum as _minimum,
    mod as _mod,
    modf as _modf,
    multiply as _multiply,
    nan_to_num as _nan_to_num,
    nancumprod as _nancumprod,
    nancumsum as _nancumsum,
    nanmax as _nanmax,
    nanprod as _nanprod,
    nansum as _nansum,
    negative as _negative,
    positive as _positive,
    power as _power,
    prod as _prod,
    rad2deg as _rad2deg,
    radians as _radians,
    reciprocal as _reciprocal,
    real as _real,
    relu as _relu,
    remainder as _remainder,
    rint as _rint,
    round_ as _round_,
    sign as _sign,
    signbit as _signbit,
    sin as _sin,
    sinc as _sinc,
    sinh as _sinh,
    sqrt as _sqrt,
    square as _square,
    subtract as _subtract,
    sum as _sum,
    tan as _tan,
    tanh as _tanh,
    true_divide as _true_divide,
    trunc as _trunc,
)
from .utils import ndarray, _convert_dtype


# Trigonometric functions
def sin(x: ArrayLike) -> ndarray:
    """
    Calculate the sine of each element.

    This function computes the sine for every element in the input array `x`.
    Input values are assumed to be in radians.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array containing angles in radians.

    Returns
    -------
    asnumpy.ndarray
        The sine of each element in `x`.

    See Also
    --------
    numpy.sin
    asnumpy.cos
    asnumpy.tan

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.sin(ap.array([0, np.pi/6, np.pi/2]))
    array([0. , 0.5, 1. ])
    """
    return ndarray(_sin(x))


def cos(x: ArrayLike) -> ndarray:
    """
    Calculate the cosine of each element.

    This function computes the cosine for every element in the input array `x`.
    Input values are assumed to be in radians.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array containing angles in radians.

    Returns
    -------
    asnumpy.ndarray
        The cosine of each element in `x`.

    See Also
    --------
    numpy.cos
    asnumpy.sin
    asnumpy.tan

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.cos(ap.array([0, np.pi]))
    array([ 1., -1.])
    """
    return ndarray(_cos(x))


def tan(x: ArrayLike) -> ndarray:
    """
    Calculate the tangent of each element.

    This function computes the tangent for every element in the input array `x`.
    Input values are assumed to be in radians.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array containing angles in radians.

    Returns
    -------
    asnumpy.ndarray
        The tangent of each element in `x`.

    See Also
    --------
    numpy.tan
    asnumpy.sin
    asnumpy.cos

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.tan(ap.array([-np.pi/4, 0, np.pi/4]))
    array([-1.,  0.,  1.])
    """
    return ndarray(_tan(x))


def arcsin(x: ArrayLike) -> ndarray:
    """
    Calculate the inverse sine of each element.

    This function computes the inverse sine (arcsine) for every element in `x`.
    The domain is defined on [-1, 1]. The returned values are in radians, ranging from -pi/2 to pi/2.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array. Elements must be within [-1, 1].

    Returns
    -------
    asnumpy.ndarray
        The inverse sine of each element in `x`.

    See Also
    --------
    numpy.arcsin
    asnumpy.sin
    asnumpy.arccos
    asnumpy.arctan

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.arcsin(ap.array([0, 0.5, 1]))
    array([0.        , 0.52359878, 1.57079633])
    """
    return ndarray(_arcsin(x))


def arccos(x: ArrayLike) -> ndarray:
    """
    Calculate the inverse cosine of each element.

    This function computes the inverse cosine (arccosine) for every element in `x`.
    The domain is defined on [-1, 1]. The returned values are in radians, ranging from 0 to pi.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array. Elements must be within [-1, 1].

    Returns
    -------
    asnumpy.ndarray
        The inverse cosine of each element in `x`.

    See Also
    --------
    numpy.arccos
    asnumpy.cos
    asnumpy.arcsin
    asnumpy.arctan

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.arccos(ap.array([1, 0.5, 0]))
    array([0.        , 1.04719755, 1.57079633])
    """
    return ndarray(_arccos(x))


def arctan(x: ArrayLike) -> ndarray:
    """
    Calculate the inverse tangent of each element.

    This function computes the inverse tangent (arctangent) for every element in `x`.
    The returned values are in radians, ranging from -pi/2 to pi/2.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        The inverse tangent of each element in `x`.

    See Also
    --------
    numpy.arctan
    asnumpy.tan
    asnumpy.arcsin
    asnumpy.arccos
    asnumpy.arctan2

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.arctan(ap.array([0, 1]))
    array([0.        , 0.78539816])
    """
    return ndarray(_arctan(x))


def arctan2(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Calculate the element-wise inverse tangent of the quotient `x1/x2`, adjusting for the quadrant.

    This function computes the inverse tangent of `x1/x2`,
    using the signs of both arguments to determine the correct quadrant of the result.
    The returned values are in radians, ranging from -pi to pi.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        Y-coordinates.
    x2 : asnumpy.ndarray
        X-coordinates.

    Returns
    -------
    asnumpy.ndarray
        Angles in radians.

    See Also
    --------
    numpy.arctan2
    asnumpy.arctan
    asnumpy.tan

    Examples
    --------
    >>> import asnumpy as ap
    >>> y = ap.array([0, 1])
    >>> x = ap.array([-1, 1])
    >>> ap.arctan2(y, x)
    array([3.14159265, 0.78539816])
    """
    return ndarray(_arctan2(x1, x2))


def hypot(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Calculate the hypotenuse given two sides of a right triangle.

    This function computes the hypotenuse for the legs `x1` and `x2`.
    It is mathematically equivalent to ``sqrt(x1**2 + x2**2)``.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        First leg.
    x2 : asnumpy.ndarray
        Second leg.

    Returns
    -------
    asnumpy.ndarray
        The hypotenuse.

    See Also
    --------
    numpy.hypot
    asnumpy.sqrt

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.hypot(3*ap.ones(2), 4*ap.ones(2))
    array([5., 5.])
    """
    return ndarray(_hypot(x1, x2))


def radians(x: ArrayLike) -> ndarray:
    """
    Convert angles from degrees to radians.

    This function converts each element in the input array `x` from degrees to radians.
    The operation is performed element-wise.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array in degrees.

    Returns
    -------
    asnumpy.ndarray
        Output array in radians.

    See Also
    --------
    numpy.radians
    asnumpy.degrees
    asnumpy.deg2rad
    asnumpy.rad2deg

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.radians(ap.array([0, 90, 180]))
    array([0.        , 1.57079633, 3.14159265])
    """
    return ndarray(_radians(x))


def deg2rad(x: ArrayLike) -> ndarray:
    """
    Convert angles from degrees to radians.

    This function converts input angles from degrees to radians element-wise.
    It is an alias for `radians`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array in degrees.

    Returns
    -------
    asnumpy.ndarray
        Output array in radians.

    See Also
    --------
    numpy.deg2rad
    asnumpy.radians
    asnumpy.degrees
    asnumpy.rad2deg

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.deg2rad(ap.array([0, 90, 180]))
    array([0.        , 1.57079633, 3.14159265])
    """
    return ndarray(_radians(x))


def degrees(x: ArrayLike) -> ndarray:
    """
    Convert angles from radians to degrees.

    This function converts each element in the input array `x` from radians to degrees.
    The operation is performed element-wise.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array in radians.

    Returns
    -------
    asnumpy.ndarray
        Output array in degrees.

    See Also
    --------
    numpy.degrees
    asnumpy.radians
    asnumpy.rad2deg
    asnumpy.deg2rad

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.degrees(ap.array([0, np.pi/2, np.pi]))
    array([  0.,  90., 180.])
    """
    return ndarray(_degrees(x))


def rad2deg(x: ArrayLike) -> ndarray:
    """
    Convert angles from radians to degrees.

    This function converts each element in the input array `x` from radians to degrees.
    It is an alias for :func:`asnumpy.degrees`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array in radians.

    Returns
    -------
    asnumpy.ndarray
        Output array in degrees.

    See Also
    --------
    numpy.rad2deg
    asnumpy.degrees
    asnumpy.radians
    asnumpy.deg2rad

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.rad2deg(ap.array([0, np.pi/2, np.pi]))
    array([  0.,  90., 180.])
    """
    return ndarray(_rad2deg(x))


# Miscellaneous functions
def absolute(x: ArrayLike) -> ndarray:
    """
    Calculate the absolute value of each element.

    This function computes the absolute value for every element in the input array `x`.
    If the input is complex, the magnitude is returned.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the absolute value of each element in `x`.

    See Also
    --------
    numpy.absolute
    asnumpy.abs

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.absolute(ap.array([-2.5, 2.5]))
    array([2.5, 2.5])
    >>> ap.absolute(ap.array([3+4j]))
    array([5.])
    """
    return ndarray(_absolute(x))


def fabs(x: ArrayLike) -> ndarray:
    """
    Calculate the absolute value for real-valued elements.

    This function computes the absolute value of each element in `x`.
    It is designed for real numbers and does not handle complex conjugation.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the absolute values of `x`.

    See Also
    --------
    numpy.fabs
    asnumpy.absolute

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.fabs(ap.array([-2.5, 2.5]))
    array([2.5, 2.5])
    """
    return ndarray(_fabs(x))


def sign(x: ArrayLike) -> ndarray:
    """
    Determine the sign of each element.

    This function returns an element-wise indication of the sign of a number: -1 for negative,
    0 for zero, and 1 for positive.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        The sign of each element in `x`.

    See Also
    --------
    numpy.sign

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.sign(ap.array([-3., 2.]))
    array([-1.,  1.])
    >>> ap.sign(0)
    0
    """
    return ndarray(_sign(x))


def heaviside(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Compute the Heaviside step function.

    This function calculates the Heaviside step function for each element in `x1`.
    The value is 0 for negative inputs, 1 for positive inputs, and `x2` when the input is zero.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        Input array.
    x2 : asnumpy.ndarray
        Value to use when `x1` is 0.

    Returns
    -------
    asnumpy.ndarray
        The result of the Heaviside step function.

    See Also
    --------
    numpy.heaviside

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.heaviside(ap.array([-2.0, 0, 1.0]), 0.5)
    array([0. , 0.5, 1. ])
    """
    return ndarray(_heaviside(x1, x2))


def clip(
    a: ArrayLike, a_min: Union[ArrayLike, float], a_max: Union[ArrayLike, float]
) -> ndarray:
    """
    Constrain array values to a given range.

    This function limits the values in `a` to be within the interval [`a_min`, `a_max`].
    Any value less than `a_min` is set to `a_min`, and any value greater than `a_max` is set to `a_max`.

    Arguments
    ---------
    a : asnumpy.ndarray
        Array containing elements to clip.
    a_min : array-like or scalar
        Minimum value.
    a_max : array-like or scalar
        Maximum value.

    Returns
    -------
    asnumpy.ndarray
        An array with the elements of `a` clipped to the specified range.

    See Also
    --------
    numpy.clip

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.arange(5)
    >>> ap.clip(a, 1, 3)
    array([1, 1, 2, 3, 3])
    """
    return ndarray(_clip(a, a_min, a_max))


def nan_to_num(
    x: ArrayLike,
    nan: float = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> ndarray:
    """
    Replace NaN and infinity with finite values.

    This function replaces NaN with zero (or a specified value) and
    infinity with large finite numbers (or specified values).

    Arguments
    ---------
    x : asnumpy.ndarray
        Input data.
    nan : float, optional
        Value to be used to fill NaN values. Default is 0.0.
    posinf : float, optional
        Value to be used to fill positive infinity values. Default is a very large number.
    neginf : float, optional
        Value to be used to fill negative infinity values. Default is a very small (negative) number.

    Returns
    -------
    asnumpy.ndarray
        Array with the same shape as `x` and the same dtype, with replacements applied.

    See Also
    --------
    numpy.nan_to_num

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.nan_to_num(ap.array([np.inf, -np.inf, np.nan]))
    array([ 1.79769313e+308, -1.79769313e+308,  0.00000000e+000])
    """
    return ndarray(_nan_to_num(x, nan, posinf, neginf))


def sqrt(x: ArrayLike) -> ndarray:
    """
    Calculate the non-negative square root of each element.

    This function computes the square root for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        The values whose square-roots are required.

    Returns
    -------
    asnumpy.ndarray
        An array of the same shape as `x`, containing the positive square-root of each element.

    See Also
    --------
    numpy.sqrt
    asnumpy.square

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.sqrt(ap.array([1, 4, 16]))
    array([1., 2., 4.])
    """
    return ndarray(_sqrt(x))


def square(x: ArrayLike) -> ndarray:
    """
    Calculate the square of each element.

    This function computes the square of the input `x` element-wise.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input data.

    Returns
    -------
    asnumpy.ndarray
        Element-wise `x*x`, of the same shape and dtype as `x`.

    See Also
    --------
    numpy.square
    asnumpy.sqrt
    asnumpy.power

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.square(ap.array([2, 3, 4]))
    array([ 4,  9, 16])
    """
    return ndarray(_square(x))


def relu(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the Rectified Linear Unit (ReLU) activation.

    This function applies the ReLU operation element-wise, returning `x` if positive and 0 otherwise.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array with the same shape as `x`, with negative values replaced by 0.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.relu(ap.array([-2, 0, 2]))
    array([0, 0, 2])
    """
    return ndarray(_relu(x, _convert_dtype(dtype)))


def gelu(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the Gaussian Error Linear Unit (GELU) activation.

    This function applies the GELU operation, which weights inputs by their probability under a Gaussian distribution.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The result of the GELU function applied to `x`.

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.gelu(ap.array([-1.0, 0.0, 1.0]))
    array([-0.15865525,  0.        ,  0.84134475])
    """
    return ndarray(_gelu(x, _convert_dtype(dtype)))


# Arithmetic operations
def add(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the sum of two inputs element-wise.

    This function adds `x1` and `x2` element by element.

    Arguments
    ---------
    x1 : array-like or scalar
        First input array or scalar.
    x2 : array-like or scalar
        Second input array or scalar.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The sum of `x1` and `x2`.

    See Also
    --------
    numpy.add
    asnumpy.subtract

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.add(ap.array([10, 20]), ap.array([5, 5]))
    array([15, 25])
    """
    return ndarray(_add(x1, x2, _convert_dtype(dtype)))


def reciprocal(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the reciprocal of each element.

    This function computes the multiplicative inverse, `1 / x`, for every element in the input array.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The reciprocal of each element in `x`.

    See Also
    --------
    numpy.reciprocal
    asnumpy.divide

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.reciprocal(ap.array([1., 2., 4.]))
    array([1.  , 0.5 , 0.25])
    """
    return ndarray(_reciprocal(x, _convert_dtype(dtype)))


def positive(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Apply the unary positive operator element-wise.

    This function returns `+x` for each element. It effectively returns a copy of the array.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The input array with the positive unary operator applied.

    See Also
    --------
    numpy.positive
    asnumpy.negative

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.positive(ap.array([-5, 5]))
    array([-5,  5])
    """
    return ndarray(_positive(x, _convert_dtype(dtype)))


def negative(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the numerical negative element-wise.

    This function negates each element in the input array, returning `-x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The negative of the input array.

    See Also
    --------
    numpy.negative
    asnumpy.positive

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.negative(ap.array([10, -10]))
    array([-10,  10])
    """
    return ndarray(_negative(x, _convert_dtype(dtype)))


def multiply(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the product of two inputs element-wise.

    This function multiplies `x1` and `x2` element by element.

    Arguments
    ---------
    x1 : array-like or scalar
        First input array or scalar.
    x2 : array-like or scalar
        Second input array or scalar.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The product of `x1` and `x2`.

    See Also
    --------
    numpy.multiply
    asnumpy.divide

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.multiply(ap.array([2.0, 4.0]), ap.array([3.0, 0.5]))
    array([6., 2.])
    """
    return ndarray(_multiply(x1, x2, _convert_dtype(dtype)))


def divide(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the division of two inputs element-wise.

    This function divides `x1` by `x2` element by element, performing true division.

    Arguments
    ---------
    x1 : array-like or scalar
        The dividend.
    x2 : array-like or scalar
        The divisor.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The quotient of `x1` divided by `x2`.

    See Also
    --------
    numpy.divide
    asnumpy.multiply
    asnumpy.floor_divide
    asnumpy.true_divide

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.divide(ap.array([6, 12]), ap.array([3, 4]))
    array([2., 3.])
    """
    return ndarray(_divide(x1, x2, _convert_dtype(dtype)))


def true_divide(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the true division of two inputs element-wise.

    This function divides `x1` by `x2` element by element. It is an alias for `divide`.

    Arguments
    ---------
    x1 : array-like or scalar
        The dividend.
    x2 : array-like or scalar
        The divisor.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The quotient of `x1` divided by `x2`.

    See Also
    --------
    numpy.true_divide
    asnumpy.divide
    asnumpy.floor_divide

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.true_divide(ap.array([6, 12]), ap.array([3, 4]))
    array([2., 3.])
    """
    return ndarray(_true_divide(x1, x2, _convert_dtype(dtype)))


def subtract(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the difference between two inputs element-wise.

    This function subtracts `x2` from `x1` element by element.

    Arguments
    ---------
    x1 : array-like or scalar
        The array to subtract from.
    x2 : array-like or scalar
        The array to subtract.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The difference `x1 - x2`.

    See Also
    --------
    numpy.subtract
    asnumpy.add

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.subtract(ap.array([10, 5]), ap.array([2, 2]))
    array([8, 3])
    """
    return ndarray(_subtract(x1, x2, _convert_dtype(dtype)))


def floor_divide(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the floor division of two inputs element-wise.

    This function divides `x1` by `x2` and rounds the quotient down to the nearest integer.
    It corresponds to the `//` operator.

    Arguments
    ---------
    x1 : array-like or scalar
        The dividend.
    x2 : array-like or scalar
        The divisor.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The result of floor division.

    See Also
    --------
    numpy.floor_divide
    asnumpy.divide
    asnumpy.floor
    asnumpy.true_divide

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.floor_divide(ap.array([10, 10]), ap.array([3, 4]))
    array([3, 2])
    """
    return ndarray(_floor_divide(x1, x2, _convert_dtype(dtype)))


def float_power(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the power of bases raised to exponents, promoting to float.

    This function raises elements of `x1` to the power of elements of `x2`.
    It ensures at least float64 precision for the calculation.

    Arguments
    ---------
    x1 : array-like or scalar
        The bases.
    x2 : array-like or scalar
        The exponents.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The result of `x1 ** x2`.

    See Also
    --------
    numpy.float_power
    asnumpy.power

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.float_power(ap.array([2, 5]), ap.array([3, 2]))
    array([ 8., 25.])
    """
    return ndarray(_float_power(x1, x2, _convert_dtype(dtype)))


def fmod(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the floating-point remainder of division.

    This function computes the remainder of `x1` divided by `x2`.
    The result carries the sign of the dividend `x1`, consistent with the C `fmod` function.

    Arguments
    ---------
    x1 : array-like or scalar
        The dividend.
    x2 : array-like or scalar
        The divisor.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The remainder of the division.

    See Also
    --------
    numpy.fmod
    asnumpy.mod
    asnumpy.remainder

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.fmod(ap.array([-4, -4, 4, 4]), ap.array([3, -3, 3, -3]))
    array([-1, -1,  1,  1])
    """
    return ndarray(_fmod(x1, x2, _convert_dtype(dtype)))


def mod(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the remainder of division element-wise.

    This function computes the remainder of `x1` divided by `x2`.
    It behaves like the Python `%` operator, where the result takes the sign of the divisor `x2`.

    Arguments
    ---------
    x1 : array-like or scalar
        The dividend.
    x2 : array-like or scalar
        The divisor.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The remainder of the division.

    See Also
    --------
    numpy.mod
    asnumpy.remainder
    asnumpy.fmod

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.mod(ap.array([-4, -4, 4, 4]), ap.array([3, -3, 3, -3]))
    array([ 2, -1,  1, -2])
    """
    return ndarray(_mod(x1, x2, _convert_dtype(dtype)))


def modf(x: ArrayLike) -> tuple:
    """
    Separate the fractional and integral parts of elements.

    This function splits each element of `x` into its fractional and integral components.
    Both returned parts have the same sign as the input.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    tuple of asnumpy.ndarray
        A tuple containing:
        - The fractional parts of `x`.
        - The integral parts of `x`.

    See Also
    --------
    numpy.modf
    asnumpy.divmod

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.modf(ap.array([1.5, -2.5]))
    (array([ 0.5, -0.5]), array([ 1., -2.]))
    """
    return ndarray(_modf(x))


def remainder(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the remainder of division element-wise.

    This function computes the remainder of `x1` divided by `x2`.
    It is an alias for `mod`, and the result takes the sign of the divisor `x2`.

    Arguments
    ---------
    x1 : array-like or scalar
        The dividend.
    x2 : array-like or scalar
        The divisor.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The remainder of the division.

    See Also
    --------
    numpy.remainder
    asnumpy.mod
    asnumpy.fmod

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.remainder(ap.array([5, -5]), ap.array([3, 3]))
    array([2, 1])
    """
    return ndarray(_remainder(x1, x2, _convert_dtype(dtype)))


def divmod(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> tuple:
    """
    Calculate both the quotient and the remainder.

    This function performs floor division and modulus simultaneously.
    It returns the pair `(x1 // x2, x1 % x2)`.

    Arguments
    ---------
    x1 : array-like or scalar
        The dividend.
    x2 : array-like or scalar
        The divisor.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    tuple of asnumpy.ndarray
        A tuple containing:
        - The element-wise floor quotient.
        - The element-wise remainder.

    See Also
    --------
    numpy.divmod
    asnumpy.floor_divide
    asnumpy.remainder

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.divmod(ap.array([10, 11]), ap.array([3, 3]))
    (array([3, 3]), array([1, 2]))
    """
    return ndarray(_divmod(x1, x2, _convert_dtype(dtype)))


def power(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the power of bases raised to exponents.

    This function raises elements of `x1` to the power of elements of `x2`.

    Arguments
    ---------
    x1 : array-like or scalar
        The bases.
    x2 : array-like or scalar
        The exponents.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The result of `x1 ** x2`.

    See Also
    --------
    numpy.power
    asnumpy.float_power
    asnumpy.square

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.power(ap.array([2, 5]), ap.array([3, 2]))
    array([ 8, 25])
    """
    return ndarray(_power(x1, x2, _convert_dtype(dtype)))


# Sums, products, differences
def prod(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Union[ndarray, float]:
    """
    Calculate the product of elements.

    This function multiplies elements in the input array `a`.
    If an `axis` is provided, the multiplication is performed along that axis.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input array.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, the product of the flattened array is returned.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions with size one.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the elements are multiplied.

    Returns
    -------
    asnumpy.ndarray or scalar
        The product of the elements.

    See Also
    --------
    numpy.prod
    asnumpy.sum

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.prod(ap.array([1., 2.]))
    2.0
    """
    if axis is None:
        return _prod(a)
    return ndarray(_prod(a, axis, keepdims, _convert_dtype(dtype)))


def sum(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Union[ndarray, float]:
    """
    Calculate the sum of elements.

    This function adds up elements in the input array `a`.
    If an `axis` is provided, the summation is performed along that axis.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input array.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, the sum of the flattened array is returned.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions with size one.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the elements are summed.

    Returns
    -------
    asnumpy.ndarray or scalar
        The sum of the elements.

    See Also
    --------
    numpy.sum
    asnumpy.prod

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.sum(ap.array([0.5, 1.5]))
    2.0
    """
    if axis is None:
        return _sum(a)
    return ndarray(_sum(a, axis, keepdims, _convert_dtype(dtype)))


def nanprod(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Union[ndarray, float]:
    """
    Calculate the product of elements, replacing NaNs with one.

    This function multiplies elements in the input array `a`, treating any NaN values as 1.
    This ensures that NaNs do not propagate into the result.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input array.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, the product of the flattened array is returned.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions with size one.
    dtype : data-type, optional
        The type of the returned array and of the accumulator.

    Returns
    -------
    asnumpy.ndarray or scalar
        The product of the elements, with NaNs treated as 1.

    See Also
    --------
    numpy.nanprod
    asnumpy.prod

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.nanprod(ap.array([1, np.nan]))
    1.0
    """
    if axis is None:
        return _nanprod(a)
    return ndarray(_nanprod(a, axis, keepdims, _convert_dtype(dtype)))


def nansum(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Union[ndarray, float]:
    """
    Calculate the sum of elements, replacing NaNs with zero.

    This function adds up elements in the input array `a`, treating any NaN values as 0.
    This ensures that NaNs do not propagate into the result.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input array.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, the sum of the flattened array is returned.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions with size one.
    dtype : data-type, optional
        The type of the returned array and of the accumulator.

    Returns
    -------
    asnumpy.ndarray or scalar
        The sum of the elements, with NaNs treated as 0.

    See Also
    --------
    numpy.nansum
    asnumpy.sum

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.nansum(ap.array([1, np.nan]))
    1.0
    """
    if axis is None:
        return _nansum(a)
    return ndarray(_nansum(a, axis, keepdims, _convert_dtype(dtype)))


def cumprod(
    a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the cumulative product of elements.

    This function computes the running product of elements along the specified axis.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed. By default, the input is flattened.
    dtype : data-type, optional
        Type of the returned array and of the accumulator.

    Returns
    -------
    asnumpy.ndarray
        A new array containing the cumulative product.

    See Also
    --------
    numpy.cumprod
    asnumpy.prod
    asnumpy.cumsum

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1, 2, 3])
    >>> ap.cumprod(a)
    array([1, 2, 6])
    """
    return ndarray(_cumprod(a, axis, _convert_dtype(dtype)))


def cumsum(
    a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the cumulative sum of elements.

    This function computes the running total of elements along the specified axis.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. By default, the input is flattened.
    dtype : data-type, optional
        Type of the returned array and of the accumulator.

    Returns
    -------
    asnumpy.ndarray
        A new array containing the cumulative sum.

    See Also
    --------
    numpy.cumsum
    asnumpy.sum
    asnumpy.cumprod

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1, 2, 3])
    >>> ap.cumsum(a)
    array([1, 3, 6])
    """
    return ndarray(_cumsum(a, axis, _convert_dtype(dtype)))


def nancumprod(
    a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the cumulative product of elements, treating NaNs as one.

    This function computes the running product of elements along the specified axis.
    Any NaN values encountered are treated as 1.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed. By default, the input is flattened.
    dtype : data-type, optional
        Type of the returned array and of the accumulator.

    Returns
    -------
    asnumpy.ndarray
        A new array containing the cumulative product.

    See Also
    --------
    numpy.nancumprod
    asnumpy.cumprod
    asnumpy.nanprod

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.nancumprod(ap.array([1, np.nan]))
    array([1., 1.])
    """
    return ndarray(_nancumprod(a, axis, _convert_dtype(dtype)))


def nancumsum(
    a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the cumulative sum of elements, treating NaNs as zero.

    This function computes the running total of elements along the specified axis.
    Any NaN values encountered are treated as 0.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. By default, the input is flattened.
    dtype : data-type, optional
        Type of the returned array and of the accumulator.

    Returns
    -------
    asnumpy.ndarray
        A new array containing the cumulative sum.

    See Also
    --------
    numpy.nancumsum
    asnumpy.cumsum
    asnumpy.nansum

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.nancumsum(ap.array([1, np.nan]))
    array([1., 1.])
    """
    return ndarray(_nancumsum(a, axis, _convert_dtype(dtype)))


def cross(a: ArrayLike, b: ArrayLike, axis: AxisOptional = None) -> ndarray:
    """
    Calculate the cross product of two vectors.

    This function computes the vector cross product of `a` and `b`.
    It operates on vectors defined by the last axis (or a specified axis), supporting dimensions of 2 or 3.

    Arguments
    ---------
    a : asnumpy.ndarray
        Components of the first vector(s).
    b : asnumpy.ndarray
        Components of the second vector(s).
    axis : int, optional
        Axis that defines the vector(s). By default, the last axis.

    Returns
    -------
    asnumpy.ndarray
        Vector cross product(s).

    See Also
    --------
    numpy.cross

    Examples
    --------
    >>> import asnumpy as ap
    >>> x = ap.array([1, 2, 3])
    >>> y = ap.array([4, 5, 6])
    >>> ap.cross(x, y)
    array([-3,  6, -3])
    """
    return ndarray(_cross(a, b, axis))


# Exponents and logarithms
def exp(x: ArrayLike) -> ndarray:
    """
    Calculate the exponential of each element.

    This function computes `e` raised to the power of each element in `x`,
    where `e` is the base of the natural logarithm.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        Element-wise exponential of ``x``.

    See Also
    --------
    numpy.exp
    asnumpy.expm1

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.exp(ap.array([1., 2.]))
    array([2.71828183, 7.3890561 ])
    """
    return ndarray(_exp(x))


def expm1(x: ArrayLike) -> ndarray:
    """
    Calculate `exp(x) - 1` for each element.

    This function computes the exponential of each element minus one.
    It is designed to be more accurate than `exp(x) - 1` for values of `x` close to zero.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        Element-wise exponential minus one.

    See Also
    --------
    numpy.expm1
    asnumpy.exp

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.expm1(ap.array([1e-10]))
    array([1.0000000e-10])
    """
    return ndarray(_expm1(x))


def exp2(x: ArrayLike) -> ndarray:
    """
    Calculate 2 raised to the power of each element.

    This function computes the base-2 exponential for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        Element-wise 2 to the power ``x``.

    See Also
    --------
    numpy.exp2
    asnumpy.power

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.exp2(ap.array([3]))
    array([8.])
    """
    return ndarray(_exp2(x))


def log(x: ArrayLike) -> ndarray:
    """
    Calculate the natural logarithm of each element.

    This function computes the logarithm to the base `e` for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        The natural logarithm of ``x``, element-wise.

    See Also
    --------
    numpy.log
    asnumpy.log10
    asnumpy.log2
    asnumpy.log1p

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.log(ap.array([ap.e]))
    array([1.])
    """
    return ndarray(_log(x))


def log10(x: ArrayLike) -> ndarray:
    """
    Calculate the base-10 logarithm of each element.

    This function computes the common logarithm (base 10) for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        The base 10 logarithm of ``x``, element-wise.

    See Also
    --------
    numpy.log10
    asnumpy.log
    asnumpy.log2

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.log10(ap.array([100.]))
    array([2.])
    """
    return ndarray(_log10(x))


def log2(x: ArrayLike) -> ndarray:
    """
    Calculate the base-2 logarithm of each element.

    This function computes the binary logarithm (base 2) for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        Base-2 logarithm of ``x``.

    See Also
    --------
    numpy.log2
    asnumpy.log
    asnumpy.log10

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.log2(ap.array([8.]))
    array([3.])
    """
    return ndarray(_log2(x))


def log1p(x: ArrayLike) -> ndarray:
    """
    Calculate the natural logarithm of `1 + x` for each element.

    This function computes `log(1 + x)` element-wise.
    It is designed to provide better precision than `log(1 + x)` when `x` is close to zero.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        Natural logarithm of ``1 + x``, element-wise.

    See Also
    --------
    numpy.log1p
    asnumpy.log

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.log1p(ap.array([1e-99]))
    array([1.e-99])
    """
    return ndarray(_log1p(x))


def logaddexp(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Calculate the logarithm of the sum of exponentials of the inputs.

    This function computes `log(exp(x1) + exp(x2))`.
    It is numerically stable and useful for operations involving probabilities in log-space.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        Input array.
    x2 : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        Logarithm of ``exp(x1) + exp(x2)``.

    See Also
    --------
    numpy.logaddexp
    asnumpy.logaddexp2

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.logaddexp(ap.array([0]), ap.array([0]))
    array([0.69314718])
    """
    return ndarray(_logaddexp(x1, x2))


def logaddexp2(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Calculate the base-2 logarithm of the sum of base-2 exponentials of the inputs.

    This function computes `log2(2**x1 + 2**x2)`.
    It is a base-2 analog of `logaddexp`.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        Input array.
    x2 : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        Base-2 logarithm of ``2**x1 + 2**x2``.

    See Also
    --------
    numpy.logaddexp2
    asnumpy.logaddexp

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.logaddexp2(ap.array([1]), ap.array([1]))
    array([2.])
    """
    return ndarray(_logaddexp2(x1, x2))


# Handling complex numbers
def real(x: ArrayLike) -> ndarray:
    """
    Return the real part of the complex argument.

    This function extracts the real component of the elements in `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.

    Returns
    -------
    asnumpy.ndarray
        The real part of the complex argument.

    See Also
    --------
    numpy.real
    asnumpy.imag

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.real(ap.array([1+5j]))
    array([1.])
    """
    return ndarray(_real(x))


# Floating point routines
def signbit(x: ArrayLike) -> ndarray:
    """
    Check if the sign bit is set for each element.

    This function returns True where the sign bit is set (indicating a negative number) and False otherwise.

    Arguments
    ---------
    x : asnumpy.ndarray
        The input array.

    Returns
    -------
    asnumpy.ndarray
        Boolean array with the same shape as ``x``.

    See Also
    --------
    numpy.signbit
    asnumpy.sign

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.signbit(ap.array([-2.5, 3.5]))
    array([ True, False])
    """
    return ndarray(_signbit(x))


def ldexp(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Calculate `x1 * (2**x2)` element-wise.

    This function computes the product of `x1` and 2 raised to the power of `x2`.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        Array of multipliers.
    x2 : asnumpy.ndarray
        Array of exponents.

    Returns
    -------
    asnumpy.ndarray
        The result of ``x1 * 2**x2``.

    See Also
    --------
    numpy.ldexp
    asnumpy.frexp

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.ldexp(ap.array([3]), ap.array([2]))
    array([12.])
    """
    return ndarray(_ldexp(x1, x2))


def copysign(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """
    Change the sign of `x1` to that of `x2` element-wise.

    This function returns a value with the magnitude of `x1` and the sign of `x2`.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        Values to change the sign of.
    x2 : asnumpy.ndarray
        The sign of ``x2`` is copied to ``x1``.

    Returns
    -------
    asnumpy.ndarray
        The values of ``x1`` with the sign of ``x2``.

    See Also
    --------
    numpy.copysign
    asnumpy.sign

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.copysign(ap.array([1.5]), ap.array([-1]))
    array([-1.5])
    """
    return ndarray(_copysign(x1, x2))


# Hyperbolic functions
def sinh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the hyperbolic sine of each element.

    This function computes the hyperbolic sine for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the hyperbolic sine of each element in ``x``.

    See Also
    --------
    numpy.sinh
    asnumpy.cosh
    asnumpy.tanh

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.sinh(ap.array([0., 1.]))
    array([0.        , 1.17520119])
    """
    return ndarray(_sinh(x, _convert_dtype(dtype)))


def cosh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the hyperbolic cosine of each element.

    This function computes the hyperbolic cosine for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the hyperbolic cosine of each element in ``x``.

    See Also
    --------
    numpy.cosh
    asnumpy.sinh
    asnumpy.tanh

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.cosh(ap.array([0., 1.]))
    array([1.        , 1.54308063])
    """
    return ndarray(_cosh(x, _convert_dtype(dtype)))


def tanh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the hyperbolic tangent of each element.

    This function computes the hyperbolic tangent for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the hyperbolic tangent of each element in ``x``.

    See Also
    --------
    numpy.tanh
    asnumpy.sinh
    asnumpy.cosh

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.tanh(ap.array([0., 1.]))
    array([0.        , 0.76159416])
    """
    return ndarray(_tanh(x, _convert_dtype(dtype)))


def arcsinh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the inverse hyperbolic sine of each element.

    This function computes the inverse hyperbolic sine for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the inverse hyperbolic sine of each element in ``x``.

    See Also
    --------
    numpy.arcsinh
    asnumpy.sinh

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.arcsinh(ap.array([0., 1.17520119]))
    array([0., 1.])
    """
    return ndarray(_arcsinh(x, _convert_dtype(dtype)))


def arccosh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the inverse hyperbolic cosine of each element.

    This function computes the inverse hyperbolic cosine for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the inverse hyperbolic cosine of each element in ``x``.

    See Also
    --------
    numpy.arccosh
    asnumpy.cosh

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.arccosh(ap.array([1., 1.54308063]))
    array([0., 1.])
    """
    return ndarray(_arccosh(x, _convert_dtype(dtype)))


def arctanh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the inverse hyperbolic tangent of each element.

    This function computes the inverse hyperbolic tangent for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array containing the inverse hyperbolic tangent of each element in ``x``.

    See Also
    --------
    numpy.arctanh
    asnumpy.tanh

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.arctanh(ap.array([0., 0.76159416]))
    array([0., 1.])
    """
    return ndarray(_arctanh(x, _convert_dtype(dtype)))


# Other special functions
def sinc(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the normalized sinc function of each element.

    This function computes the normalized sinc function,
    `sin(pi * x) / (pi * x)`, for every element in the input array `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The normalized sinc function evaluated at ``x``.

    See Also
    --------
    numpy.sinc

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.sinc(ap.array([0., 0.5]))
    array([1.        , 0.63661977])
    """
    return ndarray(_sinc(x, _convert_dtype(dtype)))


# Rational routines
def gcd(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the greatest common divisor of the inputs.

    This function computes the greatest common divisor (GCD) of the absolute values of `x1` and `x2` element-wise.

    Arguments
    ---------
    x1 : array-like or scalar
        First input array.
    x2 : array-like or scalar
        Second input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The greatest common divisor of the absolute values of the inputs.

    See Also
    --------
    numpy.gcd
    asnumpy.lcm

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.gcd(ap.array([10]), ap.array([25]))
    array([5])
    """
    return ndarray(_gcd(x1, x2, _convert_dtype(dtype)))


def lcm(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the least common multiple of the inputs.

    This function computes the least common multiple (LCM) of the absolute values of `x1` and `x2` element-wise.

    Arguments
    ---------
    x1 : array-like or scalar
        First input array.
    x2 : array-like or scalar
        Second input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The least common multiple of the absolute values of the inputs.

    See Also
    --------
    numpy.lcm
    asnumpy.gcd

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.lcm(ap.array([4]), ap.array([6]))
    array([12])
    """
    return ndarray(_lcm(x1, x2, _convert_dtype(dtype)))


# Rounding
def around(x: ArrayLike, decimals: int = 0, dtype: DTypeLike = None) -> ndarray:
    """
    Round elements to a specified number of decimal places.

    This function rounds each element in `x` to the given number of decimals.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).
        If decimals is negative, it specifies the number of positions to the left of the decimal point.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of the same type as ``x``, containing the rounded values.

    See Also
    --------
    numpy.around
    asnumpy.round_
    asnumpy.ceil
    asnumpy.floor

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.around(ap.array([0.55, 1.55]), decimals=1)
    array([0.6, 1.6])
    """
    return ndarray(_around(x, decimals, _convert_dtype(dtype)))


def round_(x: ArrayLike, decimals: int = 0, dtype: DTypeLike = None) -> ndarray:
    """
    Round elements to a specified number of decimal places.

    This function rounds each element in `x` to the given number of decimals. It is an alias for `around`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).
        If decimals is negative, it specifies the number of positions to the left of the decimal point.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of the same type as ``x``, containing the rounded values.

    See Also
    --------
    numpy.round_
    asnumpy.around
    asnumpy.ceil
    asnumpy.floor

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.round_(ap.array([0.55, 1.55]), decimals=1)
    array([0.6, 1.6])
    """
    return ndarray(_round_(x, decimals, _convert_dtype(dtype)))


def rint(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Round elements to the nearest integer.

    This function rounds each element in the input array `x` to the closest integer value.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        Output array with the same shape and type as ``x``.

    See Also
    --------
    numpy.rint
    asnumpy.floor
    asnumpy.ceil
    asnumpy.trunc

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.rint(ap.array([-1.2, 1.2]))
    array([-1.,  1.])
    """
    return ndarray(_rint(x, _convert_dtype(dtype)))


def fix(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Round elements towards zero.

    This function rounds each floating-point element to the nearest integer closer to zero.

    Arguments
    ---------
    x : asnumpy.ndarray
        An array of floats to be rounded.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The array of rounded numbers.

    See Also
    --------
    numpy.fix
    asnumpy.trunc
    asnumpy.floor
    asnumpy.ceil

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.fix(ap.array([2.9, -2.9]))
    array([ 2., -2.])
    """
    return ndarray(_fix(x, _convert_dtype(dtype)))


def floor(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the floor of each element.

    This function returns the largest integer less than or equal to each element in `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input data.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The floor of each element in ``x``.

    See Also
    --------
    numpy.floor
    asnumpy.ceil
    asnumpy.trunc

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.floor(ap.array([-1.5, 1.5]))
    array([-2.,  1.])
    """
    return ndarray(_floor(x, _convert_dtype(dtype)))


def ceil(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Calculate the ceiling of each element.

    This function returns the smallest integer greater than or equal to each element in `x`.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input data.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The ceiling of each element in ``x``.

    See Also
    --------
    numpy.ceil
    asnumpy.floor
    asnumpy.trunc

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.ceil(ap.array([-1.5, 1.5]))
    array([-1.,  2.])
    """
    return ndarray(_ceil(x, _convert_dtype(dtype)))


def trunc(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    """
    Truncate elements to their integer part.

    This function returns the integer portion of each element in `x`, effectively discarding the fractional part.

    Arguments
    ---------
    x : asnumpy.ndarray
        Input data.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The truncated value of each element in ``x``.

    See Also
    --------
    numpy.trunc
    asnumpy.floor
    asnumpy.ceil

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.trunc(ap.array([-1.5, 1.5]))
    array([-1.,  1.])
    """
    return ndarray(_trunc(x, _convert_dtype(dtype)))


# Extrema finding
def maximum(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the element-wise maximum of the inputs.

    This function compares `x1` and `x2` and returns the larger value for each element.

    Arguments
    ---------
    x1 : array-like or scalar
        The first input array.
    x2 : array-like or scalar
        The second input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The maximum of ``x1`` and ``x2``, element-wise.

    See Also
    --------
    numpy.maximum
    asnumpy.minimum
    asnumpy.fmax
    asnumpy.amax

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.maximum(ap.array([2, 3]), ap.array([1, 5]))
    array([2, 5])
    """
    return ndarray(_maximum(x1, x2, _convert_dtype(dtype)))


def minimum(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the element-wise minimum of the inputs.

    This function compares `x1` and `x2` and returns the smaller value for each element.

    Arguments
    ---------
    x1 : array-like or scalar
        The first input array.
    x2 : array-like or scalar
        The second input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The minimum of ``x1`` and ``x2``, element-wise.

    See Also
    --------
    numpy.minimum
    asnumpy.maximum
    asnumpy.fmin
    asnumpy.amin

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.minimum(ap.array([2, 3]), ap.array([1, 5]))
    array([1, 3])
    """
    return ndarray(_minimum(x1, x2, _convert_dtype(dtype)))


def fmax(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the element-wise maximum of the inputs, ignoring NaNs.

    This function compares `x1` and `x2` and returns the larger value.
    If a NaN is encountered, the other value is returned.

    Arguments
    ---------
    x1 : array-like or scalar
        The first input array.
    x2 : array-like or scalar
        The second input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The maximum of ``x1`` and ``x2``, element-wise.

    See Also
    --------
    numpy.fmax
    asnumpy.fmin
    asnumpy.maximum
    asnumpy.amax

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.fmax(ap.array([np.nan, 2]), ap.array([1, np.nan]))
    array([1., 2.])
    """
    return ndarray(_fmax(x1, x2, _convert_dtype(dtype)))


def fmin(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    """
    Calculate the element-wise minimum of the inputs, ignoring NaNs.

    This function compares `x1` and `x2` and returns the smaller value.
    If a NaN is encountered, the other value is returned.

    Arguments
    ---------
    x1 : array-like or scalar
        The first input array.
    x2 : array-like or scalar
        The second input array.
    dtype : data-type, optional
        The desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The minimum of ``x1`` and ``x2``, element-wise.

    See Also
    --------
    numpy.fmin
    asnumpy.fmax
    asnumpy.minimum
    asnumpy.amin

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> ap.fmin(ap.array([np.nan, 2]), ap.array([1, np.nan]))
    array([1., 2.])
    """
    return ndarray(_fmin(x1, x2, _convert_dtype(dtype)))


def max(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Calculate the maximum value of the array.

    This function finds the largest value in the array `a`.
    If an `axis` is provided, the maximum is computed along that axis.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input data.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    asnumpy.ndarray or scalar
        Maximum of ``a``.

    See Also
    --------
    numpy.max
    asnumpy.min
    asnumpy.maximum
    asnumpy.amax

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1, 2, 3])
    >>> ap.max(a)
    3
    """
    if axis is None:
        return _max(a)
    return ndarray(_max(a, axis, keepdims))


def amax(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Calculate the maximum value of the array.

    This function finds the largest value in the array `a`.
    It is an alias for `max`.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input data.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    asnumpy.ndarray or scalar
        Maximum of ``a``.

    See Also
    --------
    numpy.amax
    asnumpy.amin
    asnumpy.maximum
    asnumpy.max

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1, 2, 3])
    >>> ap.amax(a)
    3
    """
    if axis is None:
        return _amax(a)
    return ndarray(_amax(a, axis, keepdims))


def nanmax(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Calculate the maximum value of the array, ignoring NaNs.

    This function finds the largest value in the array `a`, skipping any NaN values.
    If an `axis` is provided, the maximum is computed along that axis.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input data.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    asnumpy.ndarray or scalar
        Maximum of ``a``.

    See Also
    --------
    numpy.nanmax
    asnumpy.nanmin
    asnumpy.max
    asnumpy.amax

    Examples
    --------
    >>> import asnumpy as ap
    >>> import numpy as np
    >>> a = ap.array([1, np.nan])
    >>> ap.nanmax(a)
    1.0
    """
    if axis is None:
        return _nanmax(a)
    return ndarray(_nanmax(a, axis, keepdims))


def min(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Calculate the minimum value of the array.

    This function finds the smallest value in the array `a`.
    If an `axis` is provided, the minimum is computed along that axis.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input data.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    asnumpy.ndarray or scalar
        Minimum of ``a``.

    See Also
    --------
    numpy.min
    asnumpy.max
    asnumpy.minimum
    asnumpy.amin

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1, 2, 3])
    >>> ap.min(a)
    1
    """
    if axis is None:
        return _min(a)
    return ndarray(_min(a, axis, keepdims))


def amin(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Calculate the minimum value of the array.

    This function finds the smallest value in the array `a`.
    It is an alias for `min`.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input data.
    axis : int or sequence of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    asnumpy.ndarray or scalar
        Minimum of ``a``.

    See Also
    --------
    numpy.amin
    asnumpy.amax
    asnumpy.minimum
    asnumpy.min

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1, 2, 3])
    >>> ap.amin(a)
    1
    """
    if axis is None:
        return _amin(a)
    return ndarray(_amin(a, axis, keepdims))

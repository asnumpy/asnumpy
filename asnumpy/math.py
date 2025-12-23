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
from .lib.asnumpy_core.math import (
    absolute as _ap_absolute,
    add as _ap_add,
    amax as _ap_amax,
    amin as _ap_amin,
    around as _ap_around,
    arccos as _ap_arccos,
    arccosh as _ap_arccosh,
    arcsin as _ap_arcsin,
    arcsinh as _ap_arcsinh,
    arctan as _ap_arctan,
    arctan2 as _ap_arctan2,
    arctanh as _ap_arctanh,
    ceil as _ap_ceil,
    clip as _ap_clip,
    copysign as _ap_copysign,
    cos as _ap_cos,
    cosh as _ap_cosh,
    cross as _ap_cross,
    cumprod as _ap_cumprod,
    cumsum as _ap_cumsum,
    degrees as _ap_degrees,
    divide as _ap_divide,
    divmod as _ap_divmod,
    exp as _ap_exp,
    exp2 as _ap_exp2,
    expm1 as _ap_expm1,
    fabs as _ap_fabs,
    fix as _ap_fix,
    float_power as _ap_float_power,
    floor as _ap_floor,
    floor_divide as _ap_floor_divide,
    fmax as _ap_fmax,
    fmin as _ap_fmin,
    fmod as _ap_fmod,
    gcd as _ap_gcd,
    gelu as _ap_gelu,
    heaviside as _ap_heaviside,
    hypot as _ap_hypot,
    lcm as _ap_lcm,
    ldexp as _ap_ldexp,
    log as _ap_log,
    log10 as _ap_log10,
    log1p as _ap_log1p,
    log2 as _ap_log2,
    logaddexp as _ap_logaddexp,
    logaddexp2 as _ap_logaddexp2,
    max as _ap_max,
    maximum as _ap_maximum,
    min as _ap_min,
    minimum as _ap_minimum,
    mod as _ap_mod,
    modf as _ap_modf,
    multiply as _ap_multiply,
    nan_to_num as _ap_nan_to_num,
    nancumprod as _ap_nancumprod,
    nancumsum as _ap_nancumsum,
    nanmax as _ap_nanmax,
    nanprod as _ap_nanprod,
    nansum as _ap_nansum,
    negative as _ap_negative,
    positive as _ap_positive,
    power as _ap_power,
    prod as _ap_prod,
    rad2deg as _ap_rad2deg,
    radians as _ap_radians,
    reciprocal as _ap_reciprocal,
    real as _ap_real,
    relu as _ap_relu,
    remainder as _ap_remainder,
    rint as _ap_rint,
    round_ as _ap_round_,
    sign as _ap_sign,
    signbit as _ap_signbit,
    sin as _ap_sin,
    sinc as _ap_sinc,
    sinh as _ap_sinh,
    sqrt as _ap_sqrt,
    square as _ap_square,
    subtract as _ap_subtract,
    sum as _ap_sum,
    tan as _ap_tan,
    tanh as _ap_tanh,
    true_divide as _ap_true_divide,
    trunc as _ap_trunc,
)
from .utils import ndarray, _convert_dtype


# Trigonometric functions
def sin(x: ndarray) -> ndarray:
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    x : ndarray
        Angle, in radians.

    Returns
    -------
    y : ndarray
        The sine of each element of x.

    See Also
    --------
    numpy.sin
    """
    return ndarray(_ap_sin(x))


def cos(x: ndarray) -> ndarray:
    """
    Cosine element-wise.

    Parameters
    ----------
    x : ndarray
        Input array in radians.

    Returns
    -------
    y : ndarray
        The corresponding cosine values.

    See Also
    --------
    numpy.cos
    """
    return ndarray(_ap_cos(x))


def tan(x: ndarray) -> ndarray:
    """
    Compute tangent element-wise.

    Parameters
    ----------
    x : ndarray
        Input array in radians.

    Returns
    -------
    y : ndarray
        The corresponding tangent values.

    See Also
    --------
    numpy.tan
    """
    return ndarray(_ap_tan(x))


def arcsin(x: ndarray) -> ndarray:
    """
    Inverse sine, element-wise.

    Parameters
    ----------
    x : ndarray
        y-coordinate on the unit circle.

    Returns
    -------
    angle : ndarray
        The inverse sine of each element in x, in radians and in the closed
        interval ``[-pi/2, pi/2]``.

    See Also
    --------
    numpy.arcsin
    """
    return ndarray(_ap_arcsin(x))


def arccos(x: ndarray) -> ndarray:
    """
    Trigonometric inverse cosine, element-wise.

    Parameters
    ----------
    x : ndarray
        x-coordinate on the unit circle. For real arguments, the domain is
        [-1, 1].

    Returns
    -------
    angle : ndarray
        The angle of the ray intersecting the unit circle at the given
        x-coordinate in radians [0, pi].

    See Also
    --------
    numpy.arccos
    """
    return ndarray(_ap_arccos(x))


def arctan(x: ndarray) -> ndarray:
    """
    Trigonometric inverse tangent, element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    out : ndarray
        Array of the same shape as `x`.

    See Also
    --------
    numpy.arctan
    """
    return ndarray(_ap_arctan(x))


def arctan2(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    Parameters
    ----------
    x1 : ndarray
        y-coordinates.
    x2 : ndarray
        x-coordinates.

    Returns
    -------
    angle : ndarray
        Array of angles in radians, in the range ``[-pi, pi]``.

    See Also
    --------
    numpy.arctan2
    """
    return ndarray(_ap_arctan2(x1, x2))


def hypot(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Given the "legs" of a right triangle, return its hypotenuse.

    Parameters
    ----------
    x1, x2 : ndarray
        Leg of the triangle(s).

    Returns
    -------
    z : ndarray
        The hypotenuse of the triangle(s).

    See Also
    --------
    numpy.hypot
    """
    return ndarray(_ap_hypot(x1, x2))


def radians(x: ndarray) -> ndarray:
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : ndarray
        Input array in degrees.

    Returns
    -------
    y : ndarray
        The corresponding radian values.

    See Also
    --------
    numpy.radians
    """
    return ndarray(_ap_radians(x))


def deg2rad(x: ndarray) -> ndarray:
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : ndarray
        Input array in degrees.

    Returns
    -------
    y : ndarray
        The corresponding radian values.

    See Also
    --------
    numpy.deg2rad
    """
    return ndarray(_ap_radians(x))


def degrees(x: ndarray) -> ndarray:
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : ndarray
        Input array in radians.

    Returns
    -------
    y : ndarray
        The corresponding degree values.

    See Also
    --------
    numpy.degrees
    """
    return ndarray(_ap_degrees(x))


def rad2deg(x: ndarray) -> ndarray:
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : ndarray
        Input array in radians.

    Returns
    -------
    y : ndarray
        The corresponding degree values.

    See Also
    --------
    numpy.rad2deg
    """
    return ndarray(_ap_rad2deg(x))


# Miscellaneous functions
def absolute(x: ndarray) -> ndarray:
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    absolute : ndarray
        An ndarray containing the absolute value of each element in `x`.

    See Also
    --------
    numpy.absolute
    """
    return ndarray(_ap_absolute(x))


def fabs(x: ndarray) -> ndarray:
    """
    Compute the absolute values element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    y : ndarray
        The absolute values of `x`.

    See Also
    --------
    numpy.fabs
    """
    return ndarray(_ap_fabs(x))


def sign(x: ndarray) -> ndarray:
    """
    Returns an element-wise indication of the sign of a number.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    y : ndarray
        The sign of `x`.

    See Also
    --------
    numpy.sign
    """
    return ndarray(_ap_sign(x))


def heaviside(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Compute the Heaviside step function.

    Parameters
    ----------
    x1 : ndarray
        Input values.
    x2 : ndarray
        The value of the function when x1 is 0.

    Returns
    -------
    out : ndarray
        The output array, element-wise Heaviside step function of x1.

    See Also
    --------
    numpy.heaviside
    """
    return ndarray(_ap_heaviside(x1, x2))


def clip(
    a: ndarray, a_min: Union[ndarray, float], a_max: Union[ndarray, float]
) -> ndarray:
    """
    Clip (limit) the values in an array.

    Parameters
    ----------
    a : ndarray
        Array containing elements to clip.
    a_min, a_max : ndarray or None
        Minimum and maximum value. If ``None``, clipping is not performed on
        corresponding edge. Only one of `a_min` and `a_max` may be ``None``.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values < `a_min` are
        replaced with `a_min`, and those > `a_max` with `a_max`.

    See Also
    --------
    numpy.clip
    """
    return ndarray(_ap_clip(a, a_min, a_max))


def nan_to_num(
    x: ndarray,
    nan: float = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> ndarray:
    """
    Replace NaN with zero and infinity with large finite numbers.

    Parameters
    ----------
    x : ndarray
        Input data.
    nan : int, float, optional
        Value to be used to fill NaN values. If no value is passed then NaN
        values will be replaced with 0.0.
    posinf : int, float, optional
        Value to be used to fill positive infinity values. If no value is
        passed then positive infinity values will be replaced with a very
        large number.
    neginf : int, float, optional
        Value to be used to fill negative infinity values. If no value is
        passed then negative infinity values will be replaced with a very
        small (or negative) number.

    Returns
    -------
    out : ndarray
        Array with the same shape as `x` and dtype of the element in `x` with
        the greatest precision.

    See Also
    --------
    numpy.nan_to_num
    """
    return ndarray(_ap_nan_to_num(x, nan, posinf, neginf))


def sqrt(x: ndarray) -> ndarray:
    """
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    x : ndarray
        The values whose square-roots are required.

    Returns
    -------
    y : ndarray
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`.

    See Also
    --------
    numpy.sqrt
    """
    return ndarray(_ap_sqrt(x))


def square(x: ndarray) -> ndarray:
    """
    Return the element-wise square of the input.

    Parameters
    ----------
    x : ndarray
        Input data.

    Returns
    -------
    out : ndarray
        Element-wise `x*x`, of the same shape and dtype as `x`.

    See Also
    --------
    numpy.square
    """
    return ndarray(_ap_square(x))


def relu(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Rectified Linear Unit.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        Output array.
    """
    return ndarray(_ap_relu(x, _convert_dtype(dtype)))


def gelu(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Gaussian Error Linear Unit.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        Output array.
    """
    return ndarray(_ap_gelu(x, _convert_dtype(dtype)))


# Arithmetic operations
def add(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        The arrays to be added.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The sum of `x1` and `x2`, element-wise.

    See Also
    --------
    numpy.add
    """
    return ndarray(_ap_add(x1, x2, _convert_dtype(dtype)))


def reciprocal(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return the reciprocal of the argument, element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        Return the reciprocal of `x`.

    See Also
    --------
    numpy.reciprocal
    """
    return ndarray(_ap_reciprocal(x, _convert_dtype(dtype)))


def positive(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Numerical positive, element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        Returned array or scalar: `y = +x`.

    See Also
    --------
    numpy.positive
    """
    return ndarray(_ap_positive(x, _convert_dtype(dtype)))


def negative(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Numerical negative, element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        Returned array or scalar: `y = -x`.

    See Also
    --------
    numpy.negative
    """
    return ndarray(_ap_negative(x, _convert_dtype(dtype)))


def multiply(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays to be multiplied.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The product of `x1` and `x2`, element-wise.

    See Also
    --------
    numpy.multiply
    """
    return ndarray(_ap_multiply(x1, x2, _convert_dtype(dtype)))


def divide(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Returns a true division of the inputs, element-wise.

    Parameters
    ----------
    x1 : ndarray
        Dividend array.
    x2 : ndarray
        Divisor array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The quotient `x1/x2`, element-wise.

    See Also
    --------
    numpy.divide
    """
    return ndarray(_ap_divide(x1, x2, _convert_dtype(dtype)))


def true_divide(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Returns a true division of the inputs, element-wise.

    Parameters
    ----------
    x1 : ndarray
        Dividend array.
    x2 : ndarray
        Divisor array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The quotient `x1/x2`, element-wise.

    See Also
    --------
    numpy.true_divide
    """
    return ndarray(_ap_true_divide(x1, x2, _convert_dtype(dtype)))


def subtract(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Subtract arguments, element-wise.

    Parameters
    ----------
    x1, x2 : ndarray
        The arrays to be subtracted from each other.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The difference of `x1` and `x2`, element-wise.

    See Also
    --------
    numpy.subtract
    """
    return ndarray(_ap_subtract(x1, x2, _convert_dtype(dtype)))


def floor_divide(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the largest integer smaller or equal to the division of the inputs.

    Parameters
    ----------
    x1 : ndarray
        Numerator.
    x2 : ndarray
        Denominator.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The floor division of `x1` and `x2`, element-wise.

    See Also
    --------
    numpy.floor_divide
    """
    return ndarray(_ap_floor_divide(x1, x2, _convert_dtype(dtype)))


def float_power(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    First array elements raised to powers from second array, element-wise.

    Parameters
    ----------
    x1 : ndarray
        The bases.
    x2 : ndarray
        The exponents.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The bases in `x1` raised to the exponents in `x2`.

    See Also
    --------
    numpy.float_power
    """
    return ndarray(_ap_float_power(x1, x2, _convert_dtype(dtype)))


def fmod(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the element-wise remainder of division.

    Parameters
    ----------
    x1 : ndarray
        Dividend.
    x2 : ndarray
        Divisor.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The remainder of the division of `x1` by `x2`.

    See Also
    --------
    numpy.fmod
    """
    return ndarray(_ap_fmod(x1, x2, _convert_dtype(dtype)))


def mod(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return element-wise remainder of division.

    Parameters
    ----------
    x1 : ndarray
        Dividend.
    x2 : ndarray
        Divisor.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The remainder of the division of `x1` by `x2`.

    See Also
    --------
    numpy.mod
    """
    return ndarray(_ap_mod(x1, x2, _convert_dtype(dtype)))


def modf(x: ndarray) -> tuple:
    """
    Return the fractional and integral parts of an array, element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    y1 : ndarray
        Fractional part of `x`.
    y2 : ndarray
        Integral part of `x`.

    See Also
    --------
    numpy.modf
    """
    return ndarray(_ap_modf(x))


def remainder(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return element-wise remainder of division.

    Parameters
    ----------
    x1 : ndarray
        Dividend.
    x2 : ndarray
        Divisor.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The remainder of the division of `x1` by `x2`.

    See Also
    --------
    numpy.remainder
    """
    return ndarray(_ap_remainder(x1, x2, _convert_dtype(dtype)))


def divmod(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> tuple:
    """
    Return element-wise quotient and remainder simultaneously.

    Parameters
    ----------
    x1 : ndarray
        Dividend.
    x2 : ndarray
        Divisor.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out1 : ndarray
        Element-wise quotient.
    out2 : ndarray
        Element-wise remainder.

    See Also
    --------
    numpy.divmod
    """
    return ndarray(_ap_divmod(x1, x2, _convert_dtype(dtype)))


def power(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    First array elements raised to powers from second array, element-wise.

    Parameters
    ----------
    x1 : ndarray
        The bases.
    x2 : ndarray
        The exponents.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The bases in `x1` raised to the exponents in `x2`.

    See Also
    --------
    numpy.power
    """
    return ndarray(_ap_power(x1, x2, _convert_dtype(dtype)))


# Sums, products, differences
def prod(
    a: ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[ndarray, float]:
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed. The default,
        axis=None, will calculate the product of all the elements in the
        input array.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are multiplied.

    Returns
    -------
    product_along_axis : ndarray
        An array shaped as `a` but with the specified axis removed.

    See Also
    --------
    numpy.prod
    """
    if axis is None:
        return _ap_prod(a)
    return ndarray(_ap_prod(a, axis, keepdims, _convert_dtype(dtype)))


def sum(
    a: ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[ndarray, float]:
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : ndarray
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis removed.

    See Also
    --------
    numpy.sum
    """
    if axis is None:
        return _ap_sum(a)
    return ndarray(_ap_sum(a, axis, keepdims, _convert_dtype(dtype)))


def nanprod(
    a: ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[ndarray, float]:
    """
    Return the product of array elements over a given axis treating Not a
    Numbers (NaNs) as ones.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are multiplied.

    Returns
    -------
    product_along_axis : ndarray
        An array shaped as `a` but with the specified axis removed.

    See Also
    --------
    numpy.nanprod
    """
    if axis is None:
        return _ap_nanprod(a)
    return ndarray(_ap_nanprod(a, axis, keepdims, _convert_dtype(dtype)))


def nansum(
    a: ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[ndarray, float]:
    """
    Return the sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.

    Returns
    -------
    sum_along_axis : ndarray
        An array shaped as `a` but with the specified axis removed.

    See Also
    --------
    numpy.nansum
    """
    if axis is None:
        return _ap_nansum(a)
    return ndarray(_ap_nansum(a, axis, keepdims, _convert_dtype(dtype)))


def cumprod(
    a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the cumulative product of elements along a given axis.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed. By default the
        input is flattened.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.

    Returns
    -------
    out : ndarray
        A new array holding the result.

    See Also
    --------
    numpy.cumprod
    """
    return ndarray(_ap_cumprod(a, axis, _convert_dtype(dtype)))


def cumsum(
    a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. By default the input
        is flattened.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.

    Returns
    -------
    out : ndarray
        A new array holding the result.

    See Also
    --------
    numpy.cumsum
    """
    return ndarray(_ap_cumsum(a, axis, _convert_dtype(dtype)))


def nancumprod(
    a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the cumulative product of array elements over a given axis treating
    Not a Numbers (NaNs) as one.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed. By default the
        input is flattened.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.

    Returns
    -------
    out : ndarray
        A new array holding the result.

    See Also
    --------
    numpy.nancumprod
    """
    return ndarray(_ap_nancumprod(a, axis, _convert_dtype(dtype)))


def nancumsum(
    a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Return the cumulative sum of array elements over a given axis treating
    Not a Numbers (NaNs) as zero.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. By default the input
        is flattened.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.

    Returns
    -------
    out : ndarray
        A new array holding the result.

    See Also
    --------
    numpy.nancumsum
    """
    return ndarray(_ap_nancumsum(a, axis, _convert_dtype(dtype)))


def cross(a: ndarray, b: ndarray, axis: Optional[int] = None) -> ndarray:
    """
    Return the cross product of two (arrays of) vectors.

    Parameters
    ----------
    a : ndarray
        Components of the first vector(s).
    b : ndarray
        Components of the second vector(s).
    axis : int, optional
        Axis along which to take the cross product.

    Returns
    -------
    c : ndarray
        Vector cross product(s).

    See Also
    --------
    numpy.cross
    """
    return ndarray(_ap_cross(a, b, axis))


# Exponents and logarithms
def exp(x: ndarray) -> ndarray:
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    out : ndarray
        Element-wise exponential of `x`.

    See Also
    --------
    numpy.exp
    """
    return ndarray(_ap_exp(x))


def expm1(x: ndarray) -> ndarray:
    """
    Calculate ``exp(x) - 1`` for all elements in the array.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    out : ndarray
        Element-wise exponential minus one: ``out = exp(x) - 1``.

    See Also
    --------
    numpy.expm1
    """
    return ndarray(_ap_expm1(x))


def exp2(x: ndarray) -> ndarray:
    """
    Calculate `2**p` for all `p` in the input array.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    out : ndarray
        Element-wise 2 to the power `x`.

    See Also
    --------
    numpy.exp2
    """
    return ndarray(_ap_exp2(x))


def log(x: ndarray) -> ndarray:
    """
    Natural logarithm, element-wise.

    Parameters
    ----------
    x : ndarray
        Input value.

    Returns
    -------
    y : ndarray
        The natural logarithm of `x`, element-wise.

    See Also
    --------
    numpy.log
    """
    return ndarray(_ap_log(x))


def log10(x: ndarray) -> ndarray:
    """
    Return the base 10 logarithm of the input array, element-wise.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    y : ndarray
        The base 10 logarithm of `x`, element-wise.

    See Also
    --------
    numpy.log10
    """
    return ndarray(_ap_log10(x))


def log2(x: ndarray) -> ndarray:
    """
    Base-2 logarithm of `x`.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    y : ndarray
        Base-2 logarithm of `x`.

    See Also
    --------
    numpy.log2
    """
    return ndarray(_ap_log2(x))


def log1p(x: ndarray) -> ndarray:
    """
    Return the natural logarithm of one plus the input array, element-wise.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    y : ndarray
        Natural logarithm of `1 + x`, element-wise.

    See Also
    --------
    numpy.log1p
    """
    return ndarray(_ap_log1p(x))


def logaddexp(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Logarithm of the sum of exponentiations of the inputs.

    Parameters
    ----------
    x1, x2 : ndarray
        Input values.

    Returns
    -------
    result : ndarray
        Logarithm of ``exp(x1) + exp(x2)``.

    See Also
    --------
    numpy.logaddexp
    """
    return ndarray(_ap_logaddexp(x1, x2))


def logaddexp2(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Logarithm of the sum of exponentiations of the inputs in base-2.

    Parameters
    ----------
    x1, x2 : ndarray
        Input values.

    Returns
    -------
    result : ndarray
        Base-2 logarithm of ``2**x1 + 2**x2``.

    See Also
    --------
    numpy.logaddexp2
    """
    return ndarray(_ap_logaddexp2(x1, x2))


# Handling complex numbers
def real(x: ndarray) -> ndarray:
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    out : ndarray
        The real part of the complex argument.

    See Also
    --------
    numpy.real
    """
    return ndarray(_ap_real(x))


# Floating point routines
def signbit(x: ndarray) -> ndarray:
    """
    Returns element-wise True where signbit is set (less than zero).

    Parameters
    ----------
    x : ndarray
        The input value(s).

    Returns
    -------
    result : ndarray
        Output array, or reference to `out` if that was supplied.

    See Also
    --------
    numpy.signbit
    """
    return ndarray(_ap_signbit(x))


def ldexp(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Returns x1 * 2**x2, element-wise.

    Parameters
    ----------
    x1 : ndarray
        Array of multipliers.
    x2 : ndarray
        Array of exponents.

    Returns
    -------
    y : ndarray
        The result of ``x1 * 2**x2``.

    See Also
    --------
    numpy.ldexp
    """
    return ndarray(_ap_ldexp(x1, x2))


def copysign(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Change the sign of x1 to that of x2, element-wise.

    Parameters
    ----------
    x1 : ndarray
        Values to change the sign of.
    x2 : ndarray
        The sign of `x2` is copied to `x1`.

    Returns
    -------
    out : ndarray
        The values of `x1` with the sign of `x2`.

    See Also
    --------
    numpy.copysign
    """
    return ndarray(_ap_copysign(x1, x2))


# Hyperbolic functions
def sinh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Hyperbolic sine, element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The corresponding hyperbolic sine values.

    See Also
    --------
    numpy.sinh
    """
    return ndarray(_ap_sinh(x, _convert_dtype(dtype)))


def cosh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        Output array of same shape as `x`.

    See Also
    --------
    numpy.cosh
    """
    return ndarray(_ap_cosh(x, _convert_dtype(dtype)))


def tanh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Compute hyperbolic tangent element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The corresponding hyperbolic tangent values.

    See Also
    --------
    numpy.tanh
    """
    return ndarray(_ap_tanh(x, _convert_dtype(dtype)))


def arcsinh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Inverse hyperbolic sine element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        Array of the same shape as `x`.

    See Also
    --------
    numpy.arcsinh
    """
    return ndarray(_ap_arcsinh(x, _convert_dtype(dtype)))


def arccosh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Inverse hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        Array of the same shape as `x`.

    See Also
    --------
    numpy.arccosh
    """
    return ndarray(_ap_arccosh(x, _convert_dtype(dtype)))


def arctanh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Inverse hyperbolic tangent element-wise.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        Array of the same shape as `x`.

    See Also
    --------
    numpy.arctanh
    """
    return ndarray(_ap_arctanh(x, _convert_dtype(dtype)))


# Other special functions
def sinc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return the sinc function.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        Sinc function of `x`.

    See Also
    --------
    numpy.sinc
    """
    return ndarray(_ap_sinc(x, _convert_dtype(dtype)))


# Rational routines
def gcd(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Returns the greatest common divisor of ``|x1|`` and ``|x2|``.

    Parameters
    ----------
    x1, x2 : ndarray
        Arrays of values.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The greatest common divisor of the absolute value of the inputs.

    See Also
    --------
    numpy.gcd
    """
    return ndarray(_ap_gcd(x1, x2, _convert_dtype(dtype)))


def lcm(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Returns the lowest common multiple of ``|x1|`` and ``|x2|``.

    Parameters
    ----------
    x1, x2 : ndarray
        Arrays of values.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The lowest common multiple of the absolute value of the inputs.

    See Also
    --------
    numpy.lcm
    """
    return ndarray(_ap_lcm(x1, x2, _convert_dtype(dtype)))


# Rounding
def around(x: ndarray, decimals: int = 0, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Round an array to the given number of decimals.

    Parameters
    ----------
    x : ndarray
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0). If decimals is
        negative, it specifies the number of positions to the left of the
        decimal point.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    rounded_array : ndarray
        An array of the same type as `x`, containing the rounded values.

    See Also
    --------
    numpy.around
    """
    return ndarray(_ap_around(x, decimals, _convert_dtype(dtype)))


def round_(x: ndarray, decimals: int = 0, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Round an array to the given number of decimals.

    Parameters
    ----------
    x : ndarray
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0). If decimals is
        negative, it specifies the number of positions to the left of the
        decimal point.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    rounded_array : ndarray
        An array of the same type as `x`, containing the rounded values.

    See Also
    --------
    numpy.round_
    """
    return ndarray(_ap_round_(x, decimals, _convert_dtype(dtype)))


def rint(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : ndarray
        Input array.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        Output array is same shape and type as `x`.

    See Also
    --------
    numpy.rint
    """
    return ndarray(_ap_rint(x, _convert_dtype(dtype)))


def fix(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Round to nearest integer towards zero.

    Parameters
    ----------
    x : ndarray
        An array of floats to be rounded.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    out : ndarray
        The array of rounded numbers.

    See Also
    --------
    numpy.fix
    """
    return ndarray(_ap_fix(x, _convert_dtype(dtype)))


def floor(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return the floor of the input, element-wise.

    Parameters
    ----------
    x : ndarray
        Input data.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The floor of each element in `x`.

    See Also
    --------
    numpy.floor
    """
    return ndarray(_ap_floor(x, _convert_dtype(dtype)))


def ceil(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return the ceiling of the input, element-wise.

    Parameters
    ----------
    x : ndarray
        Input data.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The ceiling of each element in `x`.

    See Also
    --------
    numpy.ceil
    """
    return ndarray(_ap_ceil(x, _convert_dtype(dtype)))


def trunc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    """
    Return the truncated value of the input, element-wise.

    Parameters
    ----------
    x : ndarray
        Input data.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The truncated value of each element in `x`.

    See Also
    --------
    numpy.trunc
    """
    return ndarray(_ap_trunc(x, _convert_dtype(dtype)))


# Extrema finding
def maximum(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Element-wise maximum of array elements.

    Parameters
    ----------
    x1, x2 : ndarray
        The arrays holding the elements to be compared.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The maximum of `x1` and `x2`, element-wise.

    See Also
    --------
    numpy.maximum
    """
    return ndarray(_ap_maximum(x1, x2, _convert_dtype(dtype)))


def minimum(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Element-wise minimum of array elements.

    Parameters
    ----------
    x1, x2 : ndarray
        The arrays holding the elements to be compared.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The minimum of `x1` and `x2`, element-wise.

    See Also
    --------
    numpy.minimum
    """
    return ndarray(_ap_minimum(x1, x2, _convert_dtype(dtype)))


def fmax(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Element-wise maximum of array elements.

    Parameters
    ----------
    x1, x2 : ndarray
        The arrays holding the elements to be compared.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The maximum of `x1` and `x2`, element-wise.

    See Also
    --------
    numpy.fmax
    """
    return ndarray(_ap_fmax(x1, x2, _convert_dtype(dtype)))


def fmin(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    """
    Element-wise minimum of array elements.

    Parameters
    ----------
    x1, x2 : ndarray
        The arrays holding the elements to be compared.
    dtype : dtype, optional
        The type of the output array.

    Returns
    -------
    y : ndarray
        The minimum of `x1` and `x2`, element-wise.

    See Also
    --------
    numpy.fmin
    """
    return ndarray(_ap_fmin(x1, x2, _convert_dtype(dtype)))


def max(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is
        used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.

    Returns
    -------
    max : ndarray
        Maximum of `a`.

    See Also
    --------
    numpy.max
    """
    if axis is None:
        return _ap_max(a)
    return ndarray(_ap_max(a, axis, keepdims))


def amax(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is
        used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.

    Returns
    -------
    amax : ndarray
        Maximum of `a`.

    See Also
    --------
    numpy.amax
    """
    if axis is None:
        return _ap_amax(a)
    return ndarray(_ap_amax(a, axis, keepdims))


def nanmax(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Return the maximum of an array or maximum along an axis, ignoring any
    NaNs.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is
        used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.

    Returns
    -------
    nanmax : ndarray
        Maximum of `a`.

    See Also
    --------
    numpy.nanmax
    """
    if axis is None:
        return _ap_nanmax(a)
    return ndarray(_ap_nanmax(a, axis, keepdims))


def min(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is
        used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.

    Returns
    -------
    min : ndarray
        Minimum of `a`.

    See Also
    --------
    numpy.min
    """
    if axis is None:
        return _ap_min(a)
    return ndarray(_ap_min(a, axis, keepdims))


def amin(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is
        used.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one.

    Returns
    -------
    amin : ndarray
        Minimum of `a`.

    See Also
    --------
    numpy.amin
    """
    if axis is None:
        return _ap_amin(a)
    return ndarray(_ap_amin(a, axis, keepdims))
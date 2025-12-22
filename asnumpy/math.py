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
    return ndarray(_ap_sin(x))


def cos(x: ndarray) -> ndarray:
    return ndarray(_ap_cos(x))


def tan(x: ndarray) -> ndarray:
    return ndarray(_ap_tan(x))


def arcsin(x: ndarray) -> ndarray:
    return ndarray(_ap_arcsin(x))


def arccos(x: ndarray) -> ndarray:
    return ndarray(_ap_arccos(x))


def arctan(x: ndarray) -> ndarray:
    return ndarray(_ap_arctan(x))


def arctan2(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_arctan2(x1, x2))


def hypot(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_hypot(x1, x2))


def radians(x: ndarray) -> ndarray:
    return ndarray(_ap_radians(x))


def deg2rad(x: ndarray) -> ndarray:
    return ndarray(_ap_radians(x))


def degrees(x: ndarray) -> ndarray:
    return ndarray(_ap_degrees(x))


def rad2deg(x: ndarray) -> ndarray:
    return ndarray(_ap_rad2deg(x))


# Miscellaneous functions
def absolute(x: ndarray) -> ndarray:
    return ndarray(_ap_absolute(x))


def fabs(x: ndarray) -> ndarray:
    return ndarray(_ap_fabs(x))


def sign(x: ndarray) -> ndarray:
    return ndarray(_ap_sign(x))


def heaviside(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_heaviside(x1, x2))


def clip(
    a: ndarray, a_min: Union[ndarray, float], a_max: Union[ndarray, float]
) -> ndarray:
    return ndarray(_ap_clip(a, a_min, a_max))


def nan_to_num(
    x: ndarray,
    nan: float = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> ndarray:
    return ndarray(_ap_nan_to_num(x, nan, posinf, neginf))


def sqrt(x: ndarray) -> ndarray:
    return ndarray(_ap_sqrt(x))


def square(x: ndarray) -> ndarray:
    return ndarray(_ap_square(x))


def relu(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_relu(x, _convert_dtype(dtype)))


def gelu(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_gelu(x, _convert_dtype(dtype)))


# Arithmetic operations
def add(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_add(x1, x2, _convert_dtype(dtype)))


def reciprocal(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_reciprocal(x, _convert_dtype(dtype)))


def positive(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_positive(x, _convert_dtype(dtype)))


def negative(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_negative(x, _convert_dtype(dtype)))


def multiply(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_multiply(x1, x2, _convert_dtype(dtype)))


def divide(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_divide(x1, x2, _convert_dtype(dtype)))


def true_divide(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_true_divide(x1, x2, _convert_dtype(dtype)))


def subtract(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_subtract(x1, x2, _convert_dtype(dtype)))


def floor_divide(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_floor_divide(x1, x2, _convert_dtype(dtype)))


def float_power(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_float_power(x1, x2, _convert_dtype(dtype)))


def fmod(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_fmod(x1, x2, _convert_dtype(dtype)))


def mod(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_mod(x1, x2, _convert_dtype(dtype)))


def modf(x: ndarray) -> tuple:
    return ndarray(_ap_modf(x))


def remainder(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_remainder(x1, x2, _convert_dtype(dtype)))


def divmod(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> tuple:
    return ndarray(_ap_divmod(x1, x2, _convert_dtype(dtype)))


def power(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_power(x1, x2, _convert_dtype(dtype)))


# Sums, products, differences
def prod(
    a: ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_prod(a)
    return ndarray(_ap_prod(a, axis, keepdims, _convert_dtype(dtype)))


def sum(
    a: ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_sum(a)
    return ndarray(_ap_sum(a, axis, keepdims, _convert_dtype(dtype)))


def nanprod(
    a: ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_nanprod(a)
    return ndarray(_ap_nanprod(a, axis, keepdims, _convert_dtype(dtype)))


def nansum(
    a: ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_nansum(a)
    return ndarray(_ap_nansum(a, axis, keepdims, _convert_dtype(dtype)))


def cumprod(
    a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_cumprod(a, axis, _convert_dtype(dtype)))


def cumsum(
    a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_cumsum(a, axis, _convert_dtype(dtype)))


def nancumprod(
    a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_nancumprod(a, axis, _convert_dtype(dtype)))


def nancumsum(
    a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_nancumsum(a, axis, _convert_dtype(dtype)))


def cross(a: ndarray, b: ndarray, axis: Optional[int] = None) -> ndarray:
    return ndarray(_ap_cross(a, b, axis))


# Exponents and logarithms
def exp(x: ndarray) -> ndarray:
    return ndarray(_ap_exp(x))


def expm1(x: ndarray) -> ndarray:
    return ndarray(_ap_expm1(x))


def exp2(x: ndarray) -> ndarray:
    return ndarray(_ap_exp2(x))


def log(x: ndarray) -> ndarray:
    return ndarray(_ap_log(x))


def log10(x: ndarray) -> ndarray:
    return ndarray(_ap_log10(x))


def log2(x: ndarray) -> ndarray:
    return ndarray(_ap_log2(x))


def log1p(x: ndarray) -> ndarray:
    return ndarray(_ap_log1p(x))


def logaddexp(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_logaddexp(x1, x2))


def logaddexp2(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_logaddexp2(x1, x2))


# Handling complex numbers
def real(x: ndarray) -> ndarray:
    return ndarray(_ap_real(x))


# Floating point routines
def signbit(x: ndarray) -> ndarray:
    return ndarray(_ap_signbit(x))


def ldexp(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_ldexp(x1, x2))


def copysign(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_copysign(x1, x2))


# Hyperbolic functions
def sinh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_sinh(x, _convert_dtype(dtype)))


def cosh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_cosh(x, _convert_dtype(dtype)))


def tanh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_tanh(x, _convert_dtype(dtype)))


def arcsinh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_arcsinh(x, _convert_dtype(dtype)))


def arccosh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_arccosh(x, _convert_dtype(dtype)))


def arctanh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_arctanh(x, _convert_dtype(dtype)))


# Other special functions
def sinc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_sinc(x, _convert_dtype(dtype)))


# Rational routines
def gcd(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_gcd(x1, x2, _convert_dtype(dtype)))


def lcm(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_lcm(x1, x2, _convert_dtype(dtype)))


# Rounding
def around(x: ndarray, decimals: int = 0, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_around(x, decimals, _convert_dtype(dtype)))


def round_(x: ndarray, decimals: int = 0, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_round_(x, decimals, _convert_dtype(dtype)))


def rint(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_rint(x, _convert_dtype(dtype)))


def fix(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_fix(x, _convert_dtype(dtype)))


def floor(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_floor(x, _convert_dtype(dtype)))


def ceil(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_ceil(x, _convert_dtype(dtype)))


def trunc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_trunc(x, _convert_dtype(dtype)))


# Extrema finding
def maximum(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_maximum(x1, x2, _convert_dtype(dtype)))


def minimum(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_minimum(x1, x2, _convert_dtype(dtype)))


def fmax(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_fmax(x1, x2, _convert_dtype(dtype)))


def fmin(
    x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_fmin(x1, x2, _convert_dtype(dtype)))


def max(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_max(a)
    return ndarray(_ap_max(a, axis, keepdims))


def amax(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_amax(a)
    return ndarray(_ap_amax(a, axis, keepdims))


def nanmax(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_nanmax(a)
    return ndarray(_ap_nanmax(a, axis, keepdims))


def min(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_min(a)
    return ndarray(_ap_min(a, axis, keepdims))


def amin(
    a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _ap_amin(a)
    return ndarray(_ap_amin(a, axis, keepdims))
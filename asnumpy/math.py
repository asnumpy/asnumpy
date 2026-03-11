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
    return ndarray(_sin(x))


def cos(x: ArrayLike) -> ndarray:
    return ndarray(_cos(x))


def tan(x: ArrayLike) -> ndarray:
    return ndarray(_tan(x))


def arcsin(x: ArrayLike) -> ndarray:
    return ndarray(_arcsin(x))


def arccos(x: ArrayLike) -> ndarray:
    return ndarray(_arccos(x))


def arctan(x: ArrayLike) -> ndarray:
    return ndarray(_arctan(x))


def arctan2(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_arctan2(x1, x2))


def hypot(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_hypot(x1, x2))


def radians(x: ArrayLike) -> ndarray:
    return ndarray(_radians(x))


def deg2rad(x: ArrayLike) -> ndarray:
    return ndarray(_radians(x))


def degrees(x: ArrayLike) -> ndarray:
    return ndarray(_degrees(x))


def rad2deg(x: ArrayLike) -> ndarray:
    return ndarray(_rad2deg(x))


# Miscellaneous functions
def absolute(x: ArrayLike) -> ndarray:
    return ndarray(_absolute(x))


def fabs(x: ArrayLike) -> ndarray:
    return ndarray(_fabs(x))


def sign(x: ArrayLike) -> ndarray:
    return ndarray(_sign(x))


def heaviside(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_heaviside(x1, x2))


def clip(
    a: ArrayLike, a_min: Union[ArrayLike, float], a_max: Union[ArrayLike, float]
) -> ndarray:
    return ndarray(_clip(a, a_min, a_max))


def nan_to_num(
    x: ArrayLike,
    nan: float = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> ndarray:
    return ndarray(_nan_to_num(x, nan, posinf, neginf))


def sqrt(x: ArrayLike) -> ndarray:
    return ndarray(_sqrt(x))


def square(x: ArrayLike) -> ndarray:
    return ndarray(_square(x))


def relu(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_relu(x, _convert_dtype(dtype)))


def gelu(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_gelu(x, _convert_dtype(dtype)))


# Arithmetic operations
def add(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_add(x1, x2, _convert_dtype(dtype)))


def reciprocal(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_reciprocal(x, _convert_dtype(dtype)))


def positive(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_positive(x, _convert_dtype(dtype)))


def negative(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_negative(x, _convert_dtype(dtype)))


def multiply(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_multiply(x1, x2, _convert_dtype(dtype)))


def divide(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_divide(x1, x2, _convert_dtype(dtype)))


def true_divide(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_true_divide(x1, x2, _convert_dtype(dtype)))


def subtract(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_subtract(x1, x2, _convert_dtype(dtype)))


def floor_divide(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_floor_divide(x1, x2, _convert_dtype(dtype)))


def float_power(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_float_power(x1, x2, _convert_dtype(dtype)))


def fmod(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_fmod(x1, x2, _convert_dtype(dtype)))


def mod(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_mod(x1, x2, _convert_dtype(dtype)))


def modf(x: ArrayLike) -> tuple:
    frac, inte = _modf(x)
    return [ndarray(frac), ndarray(inte)]


def remainder(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_remainder(x1, x2, _convert_dtype(dtype)))


def divmod(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> tuple:
    res1, res2 = _divmod(x1, x2)
    _type = _convert_dtype(dtype)
    return [ndarray(res1, _type), ndarray(res2, _type)]


def power(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_power(x1, x2, _convert_dtype(dtype)))


# Sums, products, differences
def prod(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Union[ndarray, float]:
    if axis is None:
        return _prod(a)
    return ndarray(_prod(a, axis, keepdims, _convert_dtype(dtype)))


def sum(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Union[ndarray, float]:
    if axis is None:
        return _sum(a)
    return ndarray(_sum(a, axis, keepdims, _convert_dtype(dtype)))


def nanprod(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Union[ndarray, float]:
    if axis is None:
        return _nanprod(a)
    return ndarray(_nanprod(a, axis, keepdims, _convert_dtype(dtype)))


def nansum(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Union[ndarray, float]:
    if axis is None:
        return _nansum(a)
    return ndarray(_nansum(a, axis, keepdims, _convert_dtype(dtype)))


def cumprod(
    a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_cumprod(a, axis, _convert_dtype(dtype)))


def cumsum(
    a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_cumsum(a, axis, _convert_dtype(dtype)))


def nancumprod(
    a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_nancumprod(a, axis, _convert_dtype(dtype)))


def nancumsum(
    a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_nancumsum(a, axis, _convert_dtype(dtype)))


def cross(a: ArrayLike, b: ArrayLike, axis: AxisOptional = None) -> ndarray:
    return ndarray(_cross(a, b, axis))


# Exponents and logarithms
def exp(x: ArrayLike) -> ndarray:
    return ndarray(_exp(x))


def expm1(x: ArrayLike) -> ndarray:
    return ndarray(_expm1(x))


def exp2(x: ArrayLike) -> ndarray:
    return ndarray(_exp2(x))


def log(x: ArrayLike) -> ndarray:
    return ndarray(_log(x))


def log10(x: ArrayLike) -> ndarray:
    return ndarray(_log10(x))


def log2(x: ArrayLike) -> ndarray:
    return ndarray(_log2(x))


def log1p(x: ArrayLike) -> ndarray:
    return ndarray(_log1p(x))


def logaddexp(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logaddexp(x1, x2))


def logaddexp2(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logaddexp2(x1, x2))


# Handling complex numbers
def real(x: ArrayLike) -> ndarray:
    return ndarray(_real(x))


# Floating point routines
def signbit(x: ArrayLike) -> ndarray:
    result = ndarray(_signbit(x))
    # CANN's aclnnSignbit does not handle IEEE 754 negative zero (-0.0).
    # Detect -0.0 via numpy and patch the result.
    import numpy as np
    np_x = x.to_numpy()
    neg_zero_mask = np.signbit(np_x) & (np_x == 0)
    if neg_zero_mask.any():
        np_result = result.to_numpy().astype(np.bool_)
        np_result |= neg_zero_mask
        return ndarray(ndarray.from_numpy(np_result))
    return result


def ldexp(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_ldexp(x1, x2))


def copysign(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_copysign(x1, x2))


# Hyperbolic functions
def sinh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_sinh(x, _convert_dtype(dtype)))


def cosh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_cosh(x, _convert_dtype(dtype)))


def tanh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_tanh(x, _convert_dtype(dtype)))


def arcsinh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_arcsinh(x, _convert_dtype(dtype)))


def arccosh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_arccosh(x, _convert_dtype(dtype)))


def arctanh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_arctanh(x, _convert_dtype(dtype)))


# Other special functions
def sinc(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_sinc(x, _convert_dtype(dtype)))


# Rational routines
def gcd(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_gcd(x1, x2, _convert_dtype(dtype)))


def lcm(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_lcm(x1, x2, _convert_dtype(dtype)))


# Rounding
def around(x: ArrayLike, decimals: int = 0, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_around(x, decimals, _convert_dtype(dtype)))


def round_(x: ArrayLike, decimals: int = 0, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_round_(x, decimals, _convert_dtype(dtype)))


def rint(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_rint(x, _convert_dtype(dtype)))


def fix(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_fix(x, _convert_dtype(dtype)))


def floor(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_floor(x, _convert_dtype(dtype)))


def ceil(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_ceil(x, _convert_dtype(dtype)))


def trunc(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_trunc(x, _convert_dtype(dtype)))


# Extrema finding
def maximum(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_maximum(x1, x2, _convert_dtype(dtype)))


def minimum(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_minimum(x1, x2, _convert_dtype(dtype)))


def fmax(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_fmax(x1, x2, _convert_dtype(dtype)))


def fmin(
    x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None
) -> ndarray:
    return ndarray(_fmin(x1, x2, _convert_dtype(dtype)))


def max(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _max(a)
    return ndarray(_max(a, axis, keepdims))


def amax(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _amax(a)
    return ndarray(_amax(a, axis, keepdims))


def nanmax(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _nanmax(a)
    return ndarray(_nanmax(a, axis, keepdims))


def min(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _min(a)
    return ndarray(_min(a, axis, keepdims))


def amin(
    a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False
) -> Union[ndarray, float]:
    if axis is None:
        return _amin(a)
    return ndarray(_amin(a, axis, keepdims))

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
from .asnumpy_core.math import (
    absolute as ap_absolute,
    add as ap_add,
    amax as ap_amax,
    around as ap_around,
    arccos as ap_arccos,
    arccosh as ap_arccosh,
    arcsin as ap_arcsin,
    arcsinh as ap_arcsinh,
    arctan as ap_arctan,
    arctan2 as ap_arctan2,
    arctanh as ap_arctanh,
    ceil as ap_ceil,
    clip as ap_clip,
    copysign as ap_copysign,
    cos as ap_cos,
    cosh as ap_cosh,
    cross as ap_cross,
    cumprod as ap_cumprod,
    cumsum as ap_cumsum,
    degrees as ap_degrees,
    divide as ap_divide,
    divmod as ap_divmod,
    exp as ap_exp,
    exp2 as ap_exp2,
    expm1 as ap_expm1,
    fabs as ap_fabs,
    fix as ap_fix,
    float_power as ap_float_power,
    floor as ap_floor,
    floor_divide as ap_floor_divide,
    fmax as ap_fmax,
    fmin as ap_fmin,
    fmod as ap_fmod,
    gcd as ap_gcd,
    gelu as ap_gelu,
    heaviside as ap_heaviside,
    hypot as ap_hypot,
    lcm as ap_lcm,
    ldexp as ap_ldexp,
    log as ap_log,
    log10 as ap_log10,
    log1p as ap_log1p,
    log2 as ap_log2,
    logaddexp as ap_logaddexp,
    logaddexp2 as ap_logaddexp2,
    max as ap_max,
    maximum as ap_maximum,
    minimum as ap_minimum,
    mod as ap_mod,
    modf as ap_modf,
    multiply as ap_multiply,
    nan_to_num as ap_nan_to_num,
    nancumprod as ap_nancumprod,
    nancumsum as ap_nancumsum,
    nanmax as ap_nanmax,
    nanprod as ap_nanprod,
    nansum as ap_nansum,
    negative as ap_negative,
    positive as ap_positive,
    power as ap_power,
    prod as ap_prod,
    rad2deg as ap_rad2deg,
    radians as ap_radians,
    reciprocal as ap_reciprocal,
    real as ap_real,
    relu as ap_relu,
    remainder as ap_remainder,
    rint as ap_rint,
    round_ as ap_round_,
    sign as ap_sign,
    signbit as ap_signbit,
    sin as ap_sin,
    sinc as ap_sinc,
    sinh as ap_sinh,
    sqrt as ap_sqrt,
    square as ap_square,
    subtract as ap_subtract,
    sum as ap_sum,
    tan as ap_tan,
    tanh as ap_tanh,
    true_divide as ap_true_divide,
    trunc as ap_trunc
)
from .utils import ndarray, _convert_dtype


def _convert_axis(axis: Optional[Union[int, Sequence[int]]]) -> Optional[Union[int, Sequence[int]]]:
    return axis

# Trigonometric functions
def sin(x: ndarray) -> ndarray:
    return ndarray(ap_sin(x._impl))

def cos(x: ndarray) -> ndarray:
    return ndarray(ap_cos(x._impl))

def tan(x: ndarray) -> ndarray:
    return ndarray(ap_tan(x._impl))

def arcsin(x: ndarray) -> ndarray:
    return ndarray(ap_arcsin(x._impl))

def arccos(x: ndarray) -> ndarray:
    return ndarray(ap_arccos(x._impl))

def arctan(x: ndarray) ->ndarray:
    return ndarray(ap_arctan(x._impl))

def arctan2(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_arctan2(x1._impl, x2._impl))

def hypot(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_hypot(x1._impl, x2._impl))

def radians(x: ndarray) -> ndarray:
    return ndarray(ap_radians(x._impl))

def deg2rad(x: ndarray) -> ndarray:
    return ndarray(ap_radians(x._impl))

def degrees(x: ndarray) -> ndarray:
    return ndarray(ap_degrees(x._impl))

def rad2deg(x: ndarray) -> ndarray:
    return ndarray(ap_rad2deg(x._impl))

# Miscellaneous functions
def absolute(x: ndarray) -> ndarray:
    return ndarray(ap_absolute(x._impl))

def fabs(x: ndarray) -> ndarray:
    return ndarray(ap_fabs(x._impl))

def sign(x: ndarray) -> ndarray:
    return ndarray(ap_sign(x._impl))

def heaviside(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_heaviside(x1._impl, x2._impl))

def clip(a: ndarray, a_min: Union[ndarray, float], a_max: Union[ndarray, float]) -> ndarray:
    a_min_impl = a_min._impl if isinstance(a_min, ndarray) else a_min
    a_max_impl = a_max._impl if isinstance(a_max, ndarray) else a_max
    return ndarray(ap_clip(a._impl, a_min_impl, a_max_impl))

def nan_to_num(x: ndarray, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None) -> ndarray:
    return ndarray(ap_nan_to_num(x._impl, nan, posinf, neginf))

def sqrt(x: ndarray) -> ndarray:
    return ndarray(ap_sqrt(x._impl))

def square(x: ndarray) -> ndarray:
    return ndarray(ap_square(x._impl))

def relu(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_relu(x._impl, _convert_dtype(dtype)))

def gelu(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_gelu(x._impl, _convert_dtype(dtype)))

# Arithmetic operations
def add(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_add(x1_impl, x2_impl, _convert_dtype(dtype)))

def reciprocal(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_reciprocal(x._impl, _convert_dtype(dtype)))

def positive(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_positive(x._impl, _convert_dtype(dtype)))

def negative(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_negative(x._impl, _convert_dtype(dtype)))

def multiply(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_multiply(x1_impl, x2_impl, _convert_dtype(dtype)))

def divide(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_divide(x1_impl, x2_impl, _convert_dtype(dtype)))

def true_divide(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_true_divide(x1_impl, x2_impl, _convert_dtype(dtype)))

def subtract(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_subtract(x1_impl, x2_impl, _convert_dtype(dtype)))

def floor_divide(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_floor_divide(x1_impl, x2_impl, _convert_dtype(dtype)))

def float_power(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_float_power(x1_impl, x2_impl, _convert_dtype(dtype)))

def fmod(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_fmod(x1_impl, x2_impl, _convert_dtype(dtype)))

def mod(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_mod(x1_impl, x2_impl, _convert_dtype(dtype)))

def modf(x: ndarray) -> tuple:
    return ndarray(ap_modf(x._impl))

def remainder(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_remainder(x1_impl, x2_impl, _convert_dtype(dtype)))

def divmod(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> tuple:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_divmod(x1_impl, x2_impl, _convert_dtype(dtype)))

def power(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_power(x1_impl, x2_impl, _convert_dtype(dtype)))

# Sums, products, differences
def prod(a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, dtype: Optional[np.dtype] = None) -> Union[ndarray, float]:
    if axis is None:
        return ap_prod(a._impl)
    return ndarray(ap_prod(a._impl, axis, keepdims, _convert_dtype(dtype)))

def sum(a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, dtype: Optional[np.dtype] = None) -> Union[ndarray, float]:
    if axis is None:
        return ap_sum(a._impl)
    return ndarray(ap_sum(a._impl, axis, keepdims, _convert_dtype(dtype)))

def nanprod(a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, dtype: Optional[np.dtype] = None) -> Union[ndarray, float]:
    if axis is None:
        return ap_nanprod(a._impl)
    return ndarray(ap_nanprod(a._impl, axis, keepdims, _convert_dtype(dtype)))

def nansum(a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, dtype: Optional[np.dtype] = None) -> Union[ndarray, float]:
    if axis is None:
        return ap_nansum(a._impl)
    return ndarray(ap_nansum(a._impl, axis, keepdims, _convert_dtype(dtype)))

def cumprod(a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_cumprod(a._impl, axis, _convert_dtype(dtype)))

def cumsum(a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_cumsum(a._impl, axis, _convert_dtype(dtype)))

def nancumprod(a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_nancumprod(a._impl, axis, _convert_dtype(dtype)))

def nancumsum(a: ndarray, axis: Optional[int] = None, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_nancumsum(a._impl, axis, _convert_dtype(dtype)))

def cross(a: ndarray, b: ndarray, axis: Optional[int] = None) -> ndarray:
    return ndarray(ap_cross(a._impl, b._impl, axis))

# Exponents and logarithms
def exp(x: ndarray) -> ndarray:
    return ndarray(ap_exp(x._impl))

def expm1(x: ndarray) -> ndarray:
    return ndarray(ap_expm1(x._impl))

def exp2(x: ndarray) -> ndarray:
    return ndarray(ap_exp2(x._impl))

def log(x: ndarray) -> ndarray:
    return ndarray(ap_log(x._impl))

def log10(x: ndarray) -> ndarray:
    return ndarray(ap_log10(x._impl))

def log2(x: ndarray) -> ndarray:
    return ndarray(ap_log2(x._impl))

def log1p(x: ndarray) -> ndarray:
    return ndarray(ap_log1p(x._impl))

def logaddexp(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_logaddexp(x1._impl, x2._impl))

def logaddexp2(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_logaddexp2(x1._impl, x2._impl))

# Handling complex numbers
def real(x: ndarray) -> ndarray:
    return ndarray(ap_real(x._impl))

# Floating point routines
def signbit(x: ndarray) -> ndarray:
    return ndarray(ap_signbit(x._impl))

def ldexp(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_ldexp(x1._impl, x2._impl))

def copysign(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_copysign(x1._impl, x2._impl))

# Hyperbolic functions
def sinh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_sinh(x._impl, _convert_dtype(dtype)))

def cosh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_cosh(x._impl, _convert_dtype(dtype)))

def tanh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_tanh(x._impl, _convert_dtype(dtype)))

def arcsinh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_arcsinh(x._impl, _convert_dtype(dtype)))

def arccosh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_arccosh(x._impl, _convert_dtype(dtype)))

def arctanh(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_arctanh(x._impl, _convert_dtype(dtype)))

# Other special functions
def sinc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_sinc(x._impl, _convert_dtype(dtype)))

# Rational routines
def gcd(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_gcd(x1_impl, x2_impl, _convert_dtype(dtype)))

def lcm(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_lcm(x1_impl, x2_impl, _convert_dtype(dtype)))

# Rounding
def around(x: ndarray, decimals: int = 0, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_around(x._impl, decimals, _convert_dtype(dtype)))

def round_(x: ndarray, decimals: int = 0, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_round_(x._impl, decimals, _convert_dtype(dtype)))

def rint(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_rint(x._impl, _convert_dtype(dtype)))

def fix(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_fix(x._impl, _convert_dtype(dtype)))

def floor(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_floor(x._impl, _convert_dtype(dtype)))

def ceil(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_ceil(x._impl, _convert_dtype(dtype)))

def trunc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_trunc(x._impl, _convert_dtype(dtype)))

# Extrema finding
def maximum(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_maximum(x1_impl, x2_impl, _convert_dtype(dtype)))

def minimum(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_minimum(x1_impl, x2_impl, _convert_dtype(dtype)))

def fmax(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_fmax(x1_impl, x2_impl, _convert_dtype(dtype)))

def fmin(x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None) -> ndarray:
    x1_impl = x1._impl if isinstance(x1, ndarray) else x1
    x2_impl = x2._impl if isinstance(x2, ndarray) else x2
    return ndarray(ap_fmin(x1_impl, x2_impl, _convert_dtype(dtype)))

def max(a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Union[ndarray, float]:
    if axis is None:
        return ap_amax(a._impl)
    return ndarray(ap_max(a._impl, axis, keepdims))

def amax(a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Union[ndarray, float]:
    if axis is None:
        return ap_amax(a._impl)
    return ndarray(ap_amax(a._impl, axis, keepdims))

def nanmax(a: ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Union[ndarray, float]:
    if axis is None:
        return ap_nanmax(a._impl)
    return ndarray(ap_nanmax(a._impl, axis, keepdims))
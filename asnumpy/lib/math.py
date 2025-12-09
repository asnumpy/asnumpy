from typing import Optional, Union, Sequence, Any
import numpy as np
from .asnumpy_core.math import (
    sin as ap_sin,
    cos as ap_cos,
    tan as ap_tan,
    arcsin as ap_arcsin,
    arccos as ap_arccos,
    arctan as ap_arctan,
    arctan2 as ap_arctan2,
    hypot as ap_hypot,
    radians as ap_radians,
    degrees as ap_degrees,
    absolute as ap_absolute,
    fabs as ap_fabs,
    sign as ap_sign,
    heaviside as ap_heaviside,
    clip as ap_clip,
    nan_to_num as ap_nan_to_num,
    sqrt as ap_sqrt,
    square as ap_square,
    relu as ap_relu,
    gelu as ap_gelu,
    add as ap_add,
    reciprocal as ap_reciprocal,
    positive as ap_positive,
    negative as ap_negative,
    multiply as ap_multiply,
    divide as ap_divide,
    true_divide as ap_true_divide,
    subtract as ap_subtract,
    floor_divide as ap_floor_divide,
    float_power as ap_float_power,
    fmod as ap_fmod,
    mod as ap_mod,
    modf as ap_modf,
    remainder as ap_remainder,
    divmod as ap_divmod,
    power as ap_power,
    prod as ap_prod,
    sum as ap_sum,
    nanprod as ap_nanprod,
    nansum as ap_nansum,
    cumprod as ap_cumprod,
    cumsum as ap_cumsum,
    nancumprod as ap_nancumprod,
    nancumsum as ap_nancumsum,
    cross as ap_cross,
    exp as ap_exp,
    expm1 as ap_expm1,
    exp2 as ap_exp2,
    log as ap_log,
    log10 as ap_log10,
    log2 as ap_log2,
    log1p as ap_log1p,
    logaddexp as ap_logaddexp,
    logaddexp2 as ap_logaddexp2,
    real as ap_real,
    signbit as ap_signbit,
    ldexp as ap_ldexp,
    copysign as ap_copysign,
    sinh as ap_sinh,
    cosh as ap_cosh,
    tanh as ap_tanh,
    arcsinh as ap_arcsinh,
    arccosh as ap_arccosh,
    arctanh as ap_arctanh,
    sinc as ap_sinc,
    gcd as ap_gcd,
    lcm as ap_lcm,
    around as ap_around,
    rad2deg as ap_rad2deg,
    round_ as ap_round_,
    rint as ap_rint,
    fix as ap_fix,
    floor as ap_floor,
    ceil as ap_ceil,
    trunc as ap_trunc,
    maximum as ap_maximum,
    minimum as ap_minimum,
    fmax as ap_fmax,
    fmin as ap_fmin,
    max as ap_max,
    amax as ap_amax,
    nanmax as ap_nanmax
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
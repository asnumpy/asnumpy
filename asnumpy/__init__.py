# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

from .array import (
    empty,
    empty_like,
    eye,
    full,
    full_like,
    identity,
    linspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
)

from .cann import finalize, init, reset_device, reset_device_force, set_device

from . import linalg

from .linalg.direct import dot, einsum, inner, matmul, outer, vdot, _direct_all_

from .logic import (
    all,
    any,
    equal,
    greater,
    greater_equal,
    isfinite,
    isinf,
    isneginf,
    isposinf,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    not_equal,
)

from .math import (
    absolute,
    add,
    amax,
    amin,
    around,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    ceil,
    clip,
    copysign,
    cos,
    cosh,
    cross,
    cumprod,
    cumsum,
    deg2rad,
    degrees,
    divide,
    divmod,
    exp,
    exp2,
    expm1,
    fabs,
    fix,
    floor,
    floor_divide,
    fmax,
    fmin,
    fmod,
    float_power,
    gelu,
    gcd,
    heaviside,
    hypot,
    lcm,
    ldexp,
    log,
    log10,
    log1p,
    log2,
    logaddexp,
    logaddexp2,
    max,
    maximum,
    min,
    minimum,
    mod,
    modf,
    multiply,
    nan_to_num,
    nancumprod,
    nancumsum,
    nanmax,
    nanprod,
    nansum,
    negative,
    power,
    positive,
    prod,
    rad2deg,
    radians,
    real,
    reciprocal,
    relu,
    remainder,
    rint,
    round_,
    sign,
    signbit,
    sin,
    sinc,
    sinh,
    sqrt,
    square,
    subtract,
    sum,
    tan,
    tanh,
    true_divide,
    trunc,
)

from . import random

from .sorting import sort

from .statistics import mean

from .nn import softmax

from .utils import broadcast_shape, ndarray

from .io import save, savez, savez_compressed, load


# Get version from package metadata (defined in pyproject.toml)
try:
    from importlib.metadata import version

    __version__ = version("asnumpy")
except Exception:
    # Fallback for development mode or if package is not installed
    __version__ = "0.2.0"


__all__ = [
    # .array
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "identity",
    "linspace",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    # .cann
    "finalize",
    "init",
    "reset_device",
    "reset_device_force",
    "set_device",
    # .linalg
    "linalg",
    # .logic
    "all",
    "any",
    "equal",
    "greater",
    "greater_equal",
    "isfinite",
    "isinf",
    "isneginf",
    "isposinf",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "not_equal",
    # .math
    "absolute",
    "add",
    "amax",
    "amin",
    "around",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "ceil",
    "clip",
    "copysign",
    "cos",
    "cosh",
    "cross",
    "cumprod",
    "cumsum",
    "deg2rad",
    "degrees",
    "divide",
    "divmod",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "fix",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "float_power",
    "gelu",
    "gcd",
    "heaviside",
    "hypot",
    "lcm",
    "ldexp",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "max",
    "maximum",
    "min",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "nan_to_num",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanprod",
    "nansum",
    "negative",
    "power",
    "positive",
    "prod",
    "rad2deg",
    "radians",
    "real",
    "reciprocal",
    "relu",
    "remainder",
    "rint",
    "round_",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "sum",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
    # .random
    "random",
    # .sorting
    "sort",
    # .statistics
    "mean",
    # .nn
    "softmax",
    # .utils
    "broadcast_shape",
    "ndarray",
    # .io
    "load",
    "save",
    "savez",
    "savez_compressed",
]

__all__.extend(_direct_all_)

import atexit


@atexit.register
def reset():
    reset_device(0)
    finalize()


init()
set_device(0)

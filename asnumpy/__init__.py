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

import sys
import os
import numpy as np
from loguru import logger
from .array import (
    array,
    asarray,
    asanyarray,
    copy,
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
from . import testing

from .sorting import sort

from .statistics import mean

from ._types import (
    ArrayLike,
    DTypeLike,
    ShapeLike,
    AxisLike,
    AxisOptional,
    ScalarLike,
)

from .nn import softmax

from .utils import broadcast_shape, ndarray

from .io import save, savez, savez_compressed, load

# NumPy-compatible constants
from numpy import e, euler_gamma, inf, nan, newaxis, pi

# NumPy-compatible dtype types
from numpy import (
    bool_, int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64,
    complex64, complex128,
    dtype, finfo, iinfo,
)

# NumPy-compatible dtype helper functions
from numpy import issubdtype, promote_types, can_cast, result_type


# Get version from package metadata (defined in pyproject.toml)
try:
    from importlib.metadata import version

    __version__ = version("asnumpy")
except Exception:
    # Fallback for development mode or if package is not installed
    __version__ = "0.2.0"


__all__ = [
    # .array
    "array",
    "asarray",
    "asanyarray",
    "copy",
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
    # types
    "ArrayLike",
    "DTypeLike",
    "ShapeLike",
    "AxisLike",
    "AxisOptional",
    "ScalarLike",
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
    # numpy constants
    "e", "euler_gamma", "inf", "nan", "newaxis", "pi",
    # numpy dtype types
    "bool_", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "complex64", "complex128",
    "dtype", "finfo", "iinfo",
    # numpy dtype helpers
    "issubdtype", "promote_types", "can_cast", "result_type",
]

__all__.extend(_direct_all_)


# ---------------------------------------------------------------------------
# Module-level __getattr__: automatic numpy fallback for unimplemented APIs
# ---------------------------------------------------------------------------
# Python's module attribute lookup order:
#   1. Explicit imports / __dict__  →  found? return immediately
#   2. Not found?  →  call __getattr__(name) if defined on the module
#
# Example: when a user writes `ap.tri(3)`, Python cannot find "tri" in
# asnumpy's explicit imports, so it calls __getattr__("tri"). This function
# then delegates to numpy.tri and wraps the returned np.ndarray into an
# asnumpy.ndarray, so the user gets a seamless experience.
#
# Guard: if a name is declared in __all__ but has no actual implementation
# (e.g. a planned-but-not-yet-implemented operator), we raise AttributeError
# immediately instead of silently falling back to numpy. This prevents real
# bugs from being masked — if we claimed to support "sin", it MUST work on
# NPU, not quietly run on CPU via numpy.
# ---------------------------------------------------------------------------

def __getattr__(name):
    """Fallback to numpy for APIs not yet natively implemented in asnumpy.

    Lookup flow:
        1. Guard check — if *name* is in __all__, it means asnumpy claims
           to support it, so raise AttributeError to signal a real bug
           (e.g. the import is missing or the implementation is broken).
        2. Delegate to numpy — try ``getattr(np, name)``.
        3. If the result is callable, wrap it so that any np.ndarray
           returned by numpy is automatically converted to asnumpy.ndarray
           via ``_wrap_result``. Non-callable attributes (e.g. constants)
           are returned as-is.

    This is the same pattern used by CuPy for APIs it has not yet ported
    to GPU: the user gets a working API immediately, and asnative
    implementations can be added incrementally without breaking existing
    user code.
    """
    # Guard: names in __all__ must have real asnumpy implementations.
    # If we reach here, something is wrong — either a missing import
    # or a broken implementation. Fail loudly to avoid silent bugs.
    if name in __all__:
        raise AttributeError(
            f"module 'asnumpy' has no attribute {name!r}"
        )

    # Delegate to numpy
    try:
        attr = getattr(np, name)
    except AttributeError:
        raise AttributeError(
            f"module 'asnumpy' has no attribute {name!r}"
        )

    # Wrap callables so that np.ndarray return values become asnumpy.ndarray.
    # Non-callable attributes (e.g. np.ndarray subclass types) pass through.
    if callable(attr):
        def _wrapped(*args, **kwargs):
            result = attr(*args, **kwargs)
            return _wrap_result(result)
        # Preserve introspection so that tracebacks and help() look correct
        _wrapped.__name__ = name
        _wrapped.__qualname__ = f'asnumpy.{name}'
        _wrapped.__module__ = 'asnumpy'
        return _wrapped

    return attr


def _wrap_result(result):
    """Recursively convert np.ndarray values to asnumpy.ndarray.

    This handles the common return types from numpy functions:
        - np.ndarray       → asnumpy.ndarray (via from_numpy)
        - tuple of arrays  → tuple of asnumpy.ndarray
        - list of arrays   → list of asnumpy.ndarray
        - scalars / other  → returned as-is
    """
    if isinstance(result, np.ndarray):
        return ndarray.from_numpy(result)
    if isinstance(result, tuple):
        return tuple(_wrap_result(r) for r in result)
    if isinstance(result, list):
        return [_wrap_result(r) for r in result]
    return result


logger.disable("asnumpy")


def enable_logging(level="INFO", log_dir=None):
    """
    Enable low-level logging for asnumpy.
    :param level: Logging level (e.g., "DEBUG", "INFO").
    :param log_dir: If specified, logs will be saved to a file in this directory alongside console output.
    """
    # Enable logging for the current module
    logger.enable("asnumpy")
    logger.remove() # Remove Loguru's default handler

    # Add safe console output (use sys.__stderr__ to avoid closed stream errors during atexit)
    logger.add(sys.__stderr__, level=level,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
               catch=True)

    # Write to file only if log_dir is provided by the user
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "asnumpy_{time:YYYY-MM-DD_HHmmss}.log")
        logger.add(log_file, retention="7 days", level=level, catch=True)
        logger.info(f"ASNumPy file logging enabled: {log_file}")


if os.getenv("ASNUMPY_DEBUG", "0") == "1":
    enable_logging(level="DEBUG", log_dir=os.getenv("ASNUMPY_LOG_DIR", None))


import atexit


@atexit.register
def reset():
    reset_device(0)
    finalize()


init()
set_device(0)

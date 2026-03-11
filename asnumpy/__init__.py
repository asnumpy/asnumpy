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
from loguru import logger
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
]

__all__.extend(_direct_all_)


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

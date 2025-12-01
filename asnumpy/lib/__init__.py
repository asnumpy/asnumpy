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

from .asnumpy_core import *
from .asnumpy_core.math import * 
from .asnumpy_core.random import *
from .asnumpy_core.cann import * 
from .asnumpy_core.array import *
from .asnumpy_core.logic import * 
from .asnumpy_core.sorting import * 
from .asnumpy_core import linalg  
# linalg模块内部分需要ap.linalg.xxx调用，部分ap.yyy调用，
# yyy类函数分到了.asnumpy_core根模块中


__all__ = [
    "zeros",
    "zeros_like",
    "full",
    "full_like",
    "empty",
    "empty_like",
    "eye",
    "ones",
    "ones_like",
    "identity",
    "ndarray",
    "linspace",
    "init",
    "finalize",
    "set_device",
    "reset_device",
    "broadcast_shape",
    "absolute",
    "fabs",
    "sign",
    "heaviside",
    "linalg",  # linalg整个子模块
    "dot",
    "vdot",
    "inner",
    "outer",
    "matmul",
    "einsum",
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "power",
    "float_power",
    "fmod",
    "mod",
    "remainder",
    "modf",
    "divmod",
    "positive",
    "negative",
    "reciprocal",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "hypot",
    "arctan2",
    "radians",
    "deg2rad",
    "degrees",
    "rad2deg",
    "prod",
    "sum",
    "nanprod",
    "nansum",
    "cumprod",
    "cumsum",
    "nancumprod",
    "nancumsum",
    "cross",
    "exp",
    "expm1",
    "exp2",
    "log",
    "log10",
    "log2",
    "log1p",
    "logaddexp",
    "logaddexp2",
    "real",
    "around",
    "round_",
    "sinc",
    "lcm",
    "gcd",
    "rint",
    "fix",
    "floor",
    "ceil",
    "trunc",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "signbit",
    "ldexp",
    "copysign",
    "clip",
    "sqrt",
    "square",
    "nan_to_num",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "max",
    "amax",
    "nanmax",
    "relu",
    "gelu",
    "pareto",
    "rayleigh",
    "normal",
    "uniform",
    "standard_normal",
    "standard_cauchy",
    "weibull",
    "binomial",
    "exponential",
    "geometric",
    "gumbel",
    "laplace",
    "logistic",
    "lognormal",
    # logic
    "all",
    "any",
    "isfinite",
    "isinf",
    "isneginf",
    "isposinf",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "sort"
]




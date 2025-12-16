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
from .asnumpy_core import linalg  
# linalg模块内部分需要ap.linalg.xxx调用，部分ap.yyy调用，
# yyy类函数分到了.asnumpy_core根模块中
import numpy as np
from .asnumpy_core import dtypes
from .asnumpy_core.dtypes import float8_e5m2 as _float8_e5m2_type, bfloat16 as _bfloat16_type
from ._dtype_wrappers import _CallableDtype

# 创建可调用的 dtype 包装对象
# 使用 _CallableDtype 包装，使其同时支持：
# 1. 调用创建标量: ap.bfloat16(3.14)
# 2. 直接作为 dtype: dtype=ap.bfloat16
# 3. 用于结构化数组: np.dtype([('x', ap.bfloat16)])
float8_e5m2 = _CallableDtype(np.dtype(_float8_e5m2_type), _float8_e5m2_type)
bfloat16 = _CallableDtype(np.dtype(_bfloat16_type), _bfloat16_type)
# ============================================================================
# 包装 pybind11 函数和类以支持 _CallableDtype
# ============================================================================

from ._pybind11_wrappers import (
    wrap_array_creation_functions,
    wrap_ndarray_class,
)

# 包装数组创建函数
_wrapped_functions = wrap_array_creation_functions(
    ones_func=ones,
    zeros_func=zeros,
    empty_func=empty,
    full_func=full,
    eye_func=eye,
    identity_func=identity,
    ones_like_func=ones_like,
    zeros_like_func=zeros_like,
    full_like_func=full_like,
    empty_like_func=empty_like,
)

# 替换原始函数为包装后的版本
ones = _wrapped_functions['ones']
zeros = _wrapped_functions['zeros']
empty = _wrapped_functions['empty']
full = _wrapped_functions['full']
eye = _wrapped_functions['eye']
identity = _wrapped_functions['identity']
ones_like = _wrapped_functions['ones_like']
zeros_like = _wrapped_functions['zeros_like']
full_like = _wrapped_functions['full_like']
empty_like = _wrapped_functions['empty_like']

# 包装 ndarray 类
ndarray = wrap_ndarray_class(ndarray)

__all__ = [
    "dtypes",
    "float8_e5m2",
    "bfloat16",
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
    "clip",
    "square",
    "nan_to_num",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
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
]




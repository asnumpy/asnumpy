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

from .._core.random import (
    binomial as _binomial,
)
from .._core.random import (
    exponential as _exponential,
)
from .._core.random import (
    geometric as _geometric,
)
from .._core.random import (
    gumbel as _gumbel,
)
from .._core.random import (
    laplace as _laplace,
)
from .._core.random import (
    logistic as _logistic,
)
from .._core.random import (
    lognormal as _lognormal,
)
from .._core.random import (
    normal as _normal,
)
from .._core.random import (
    pareto as _pareto,
)
from .._core.random import (
    rayleigh as _rayleigh,
)
from .._core.random import (
    standard_cauchy as _standard_cauchy,
)
from .._core.random import (
    standard_normal as _standard_normal,
)
from .._core.random import (
    uniform as _uniform,
)
from .._core.random import (
    weibull as _weibull,
)
from .._types import ShapeLike
from ..utils import _convert_size, ndarray


def pareto(a: float, size: ShapeLike) -> ndarray:
    return ndarray(_pareto(a, _convert_size(size)))


def rayleigh(scale: float, size: ShapeLike) -> ndarray:
    return ndarray(_rayleigh(scale, _convert_size(size)))


def normal(loc: float, scale: float, size: ShapeLike) -> ndarray:
    return ndarray(_normal(loc, scale, _convert_size(size)))


def uniform(low: float, high: float, size: ShapeLike) -> ndarray:
    return ndarray(_uniform(low, high, _convert_size(size)))


def standard_normal(size: ShapeLike) -> ndarray:
    return ndarray(_standard_normal(_convert_size(size)))


def standard_cauchy(size: ShapeLike) -> ndarray:
    return ndarray(_standard_cauchy(_convert_size(size)))


def weibull(a: float, size: ShapeLike) -> ndarray:
    return ndarray(_weibull(a, _convert_size(size)))


def binomial(n: int, p: float, size: ShapeLike) -> ndarray:
    return ndarray(_binomial(n, p, _convert_size(size)))


def exponential(scale: float, size: ShapeLike) -> ndarray:
    return ndarray(_exponential(scale, _convert_size(size)))


def geometric(p: float, size: ShapeLike) -> ndarray:
    return ndarray(_geometric(p, _convert_size(size)))


def gumbel(loc: float, scale: float, size: ShapeLike) -> ndarray:
    return ndarray(_gumbel(loc, scale, _convert_size(size)))


def laplace(loc: float, scale: float, size: ShapeLike) -> ndarray:
    return ndarray(_laplace(loc, scale, _convert_size(size)))


def logistic(loc: float, scale: float, size: ShapeLike) -> ndarray:
    return ndarray(_logistic(loc, scale, _convert_size(size)))


def lognormal(mean: float, sigma: float, size: ShapeLike) -> ndarray:
    return ndarray(_lognormal(mean, sigma, _convert_size(size)))

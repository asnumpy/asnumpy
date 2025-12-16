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

from typing import Union, Sequence
from ..lib.asnumpy_core.random import (
    binomial as _ap_binomial,
    exponential as _ap_exponential,
    geometric as _ap_geometric,
    gumbel as _ap_gumbel,
    laplace as _ap_laplace,
    lognormal as _ap_lognormal,
    logistic as _ap_logistic,
    normal as _ap_normal,
    pareto as _ap_pareto,
    rayleigh as _ap_rayleigh,
    standard_cauchy as _ap_standard_cauchy,
    standard_normal as _ap_standard_normal,
    uniform as _ap_uniform,
    weibull as _ap_weibull,
)
from ..utils import ndarray


def pareto(a: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_pareto(a, size))


def rayleigh(scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_rayleigh(scale, size))


def normal(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_normal(loc, scale, size))


def uniform(low: float, high: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_uniform(low, high, size))


def standard_normal(size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_standard_normal(size))


def standard_cauchy(size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_standard_cauchy(size))


def weibull(a: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_weibull(a, size))


def binomial(n: int, p: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_binomial(n, p, size))


def exponential(scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_exponential(scale, size))


def geometric(p: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_geometric(p, size))


def gumbel(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_gumbel(loc, scale, size))


def laplace(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_laplace(loc, scale, size))


def logistic(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_logistic(loc, scale, size))


def lognormal(mean: float, sigma: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(_ap_lognormal(mean, sigma, size))

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
from ..utils import ndarray, _convert_size


def pareto(a: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a Pareto II or Lomax distribution with specified shape.

    Parameters
    ----------
    a : float
        Shape of the distribution. Must be positive.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized Pareto distribution.

    See Also
    --------
    numpy.random.pareto
    """
    return ndarray(_ap_pareto(a, _convert_size(size)))


def rayleigh(scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a Rayleigh distribution.

    Parameters
    ----------
    scale : float
        Scale, also equals the mode. Must be non-negative.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized Rayleigh distribution.

    See Also
    --------
    numpy.random.rayleigh
    """
    return ndarray(_ap_rayleigh(scale, _convert_size(size)))


def normal(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw random samples from a normal (Gaussian) distribution.

    Parameters
    ----------
    loc : float
        Mean ("centre") of the distribution.
    scale : float
        Standard deviation (spread or "width") of the distribution. Must be
        non-negative.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized normal distribution.

    See Also
    --------
    numpy.random.normal
    """
    return ndarray(_ap_normal(loc, scale, _convert_size(size)))


def uniform(low: float, high: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a uniform distribution.

    Parameters
    ----------
    low : float
        Lower boundary of the output interval. All values generated will be
        greater than or equal to low.
    high : float
        Upper boundary of the output interval. All values generated will be
        less than or equal to high.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized uniform distribution.

    See Also
    --------
    numpy.random.uniform
    """
    return ndarray(_ap_uniform(low, high, _convert_size(size)))


def standard_normal(size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a standard Normal distribution (mean=0, stdev=1).

    Parameters
    ----------
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples.

    See Also
    --------
    numpy.random.standard_normal
    """
    return ndarray(_ap_standard_normal(_convert_size(size)))


def standard_cauchy(size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a standard Cauchy distribution with mode = 0.

    Parameters
    ----------
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples.

    See Also
    --------
    numpy.random.standard_cauchy
    """
    return ndarray(_ap_standard_cauchy(_convert_size(size)))


def weibull(a: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a Weibull distribution.

    Parameters
    ----------
    a : float
        Shape parameter of the distribution. Must be non-negative.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized Weibull distribution.

    See Also
    --------
    numpy.random.weibull
    """
    return ndarray(_ap_weibull(a, _convert_size(size)))


def binomial(n: int, p: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a binomial distribution.

    Parameters
    ----------
    n : int
        Parameter of the distribution, >= 0.
    p : float
        Parameter of the distribution, >= 0 and <=1.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized binomial distribution.

    See Also
    --------
    numpy.random.binomial
    """
    return ndarray(_ap_binomial(n, p, _convert_size(size)))


def exponential(scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from an exponential distribution.

    Parameters
    ----------
    scale : float
        The scale parameter, \beta = 1/\lambda. Must be non-negative.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized exponential distribution.

    See Also
    --------
    numpy.random.exponential
    """
    return ndarray(_ap_exponential(scale, _convert_size(size)))


def geometric(p: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from the geometric distribution.

    Parameters
    ----------
    p : float
        The probability of success of an individual trial.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized geometric distribution.

    See Also
    --------
    numpy.random.geometric
    """
    return ndarray(_ap_geometric(p, _convert_size(size)))


def gumbel(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a Gumbel distribution.

    Parameters
    ----------
    loc : float
        The location of the mode of the distribution.
    scale : float
        The scale parameter of the distribution.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized Gumbel distribution.

    See Also
    --------
    numpy.random.gumbel
    """
    return ndarray(_ap_gumbel(loc, scale, _convert_size(size)))


def laplace(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from the Laplace or double exponential distribution with
    specified location (or mean) and scale (decay).

    Parameters
    ----------
    loc : float
        The position, \mu, of the distribution peak.
    scale : float
        The scale, \lambda, of the exponential decay.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized Laplace distribution.

    See Also
    --------
    numpy.random.laplace
    """
    return ndarray(_ap_laplace(loc, scale, _convert_size(size)))


def logistic(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a logistic distribution.

    Parameters
    ----------
    loc : float
        Parameter of the distribution.
    scale : float
        Parameter of the distribution.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized logistic distribution.

    See Also
    --------
    numpy.random.logistic
    """
    return ndarray(_ap_logistic(loc, scale, _convert_size(size)))


def lognormal(mean: float, sigma: float, size: Union[int, Sequence[int]]) -> ndarray:
    """
    Draw samples from a log-normal distribution.

    Parameters
    ----------
    mean : float
        Mean value of the underlying normal distribution.
    sigma : float
        Standard deviation of the underlying normal distribution.
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized log-normal distribution.

    See Also
    --------
    numpy.random.lognormal
    """
    return ndarray(_ap_lognormal(mean, sigma, _convert_size(size)))

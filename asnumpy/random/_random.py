# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

from ..lib.asnumpy_core.random import (
    binomial as _binomial,
    exponential as _exponential,
    geometric as _geometric,
    gumbel as _gumbel,
    laplace as _laplace,
    lognormal as _lognormal,
    logistic as _logistic,
    normal as _normal,
    pareto as _pareto,
    rayleigh as _rayleigh,
    standard_cauchy as _standard_cauchy,
    standard_normal as _standard_normal,
    uniform as _uniform,
    weibull as _weibull,
)
from ..utils import ndarray, _convert_size
from .._types import ShapeLike


def pareto(a: float, size: ShapeLike) -> ndarray:
    """
    Draw random samples from a Pareto II (Lomax) distribution.

    Generates random samples from a Pareto II distribution with the specified shape parameter `a`.
    The samples are drawn such that they follow the probability density function defined by the shape parameter.

    Arguments
    ---------
    a : float
        The shape parameter of the distribution. Must be greater than zero.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the Pareto distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.pareto

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.pareto(a=3.0, size=5)
    array([0.123, 0.456, 0.789, 0.012, 0.345])  # random
    """
    return ndarray(_pareto(a, _convert_size(size)))


def rayleigh(scale: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a Rayleigh distribution.

    Generates random samples from a Rayleigh distribution.
    The distribution is determined by the `scale` parameter, which corresponds to the mode of the distribution.

    Arguments
    ---------
    scale : float
        The scale parameter of the distribution. Must be non-negative.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the Rayleigh distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.rayleigh

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.rayleigh(scale=2.0, size=(2, 2))
    array([[1.5, 2.1],
           [0.8, 3.2]])  # random
    """
    return ndarray(_rayleigh(scale, _convert_size(size)))


def normal(loc: float, scale: float, size: ShapeLike) -> ndarray:
    """
    Draw random samples from a normal (Gaussian) distribution.

    Generates random samples from a normal distribution characterized by
    its mean (`loc`) and standard deviation (`scale`).

    Arguments
    ---------
    loc : float
        The mean (center) of the distribution.
    scale : float
        The standard deviation (spread) of the distribution. Must be non-negative.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the normal distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.normal

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.normal(loc=0.0, scale=1.0, size=3)
    array([-0.5,  1.2, -0.1])  # random
    """
    return ndarray(_normal(loc, scale, _convert_size(size)))


def uniform(low: float, high: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a uniform distribution.

    Generates random samples from a uniform distribution over the interval [`low`, `high`).
    Any value within the given interval is equally likely to be drawn.

    Arguments
    ---------
    low : float
        The lower boundary of the output interval.
    high : float
        The upper boundary of the output interval.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the uniform distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.uniform

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.uniform(low=0.0, high=10.0, size=4)
    array([2.5, 8.1, 0.3, 5.9])  # random
    """
    return ndarray(_uniform(low, high, _convert_size(size)))


def standard_normal(size: ShapeLike) -> ndarray:
    """
    Draw samples from a standard Normal distribution.

    Generates random samples from a standard Normal distribution, which has a mean of 0 and a standard deviation of 1.

    Arguments
    ---------
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the standard Normal distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.standard_normal

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.standard_normal(size=(2, 2))
    array([[ 0.5, -1.2],
           [ 0.1,  0.8]])  # random
    """
    return ndarray(_standard_normal(_convert_size(size)))


def standard_cauchy(size: ShapeLike) -> ndarray:
    """
    Draw samples from a standard Cauchy distribution.

    Generates random samples from a standard Cauchy distribution with mode equal to 0.

    Arguments
    ---------
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the standard Cauchy distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.standard_cauchy

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.standard_cauchy(size=3)
    array([ 0.1, -2.5,  0.8])  # random
    """
    return ndarray(_standard_cauchy(_convert_size(size)))


def weibull(a: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a Weibull distribution.

    Generates random samples from a Weibull distribution with the specified shape parameter `a`.

    Arguments
    ---------
    a : float
        The shape parameter of the distribution. Must be non-negative.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the Weibull distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.weibull

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.weibull(a=2.0, size=4)
    array([0.5, 1.2, 0.8, 1.5])  # random
    """
    return ndarray(_weibull(a, _convert_size(size)))


def binomial(n: int, p: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a binomial distribution.

    Generates random samples from a binomial distribution.
    This distribution models the number of successes in `n` independent trials,
    where each trial has a probability `p` of success.

    Arguments
    ---------
    n : int
        The number of trials. Must be non-negative.
    p : float
        The probability of success for each trial. Must be in the interval [0, 1].
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the binomial distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.binomial

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.binomial(n=10, p=0.5, size=5)
    array([5, 6, 4, 5, 7])  # random
    """
    return ndarray(_binomial(n, p, _convert_size(size)))


def exponential(scale: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from an exponential distribution.

    Generates random samples from an exponential distribution.
    The distribution is characterized by the `scale` parameter, which is the inverse of the rate parameter lambda.

    Arguments
    ---------
    scale : float
        The scale parameter of the distribution. Must be non-negative.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the exponential distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.exponential

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.exponential(scale=2.0, size=3)
    array([1.5, 0.8, 3.2])  # random
    """
    return ndarray(_exponential(scale, _convert_size(size)))


def geometric(p: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a geometric distribution.

    Generates random samples from a geometric distribution.
    This distribution models the number of trials needed to achieve the first success,
    where each trial has a probability `p` of success.

    Arguments
    ---------
    p : float
        The probability of success for an individual trial. Must be in the interval (0, 1].
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the geometric distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.geometric

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.geometric(p=0.3, size=4)
    array([2, 5, 1, 3])  # random
    """
    return ndarray(_geometric(p, _convert_size(size)))


def gumbel(loc: float, scale: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a Gumbel distribution.

    Generates random samples from a Gumbel distribution.
    This distribution is often used to model the distribution of the maximum (or the minimum) of
    a number of samples of various distributions.

    Arguments
    ---------
    loc : float
        The location parameter, representing the mode of the distribution.
    scale : float
        The scale parameter of the distribution. Must be non-negative.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the Gumbel distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.gumbel

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.gumbel(loc=0.0, scale=1.0, size=3)
    array([0.5, 1.2, -0.3])  # random
    """
    return ndarray(_gumbel(loc, scale, _convert_size(size)))


def laplace(loc: float, scale: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a Laplace distribution.

    Generates random samples from a Laplace distribution, also known as the double exponential distribution.
    It is specified by its location parameter (mean) and scale parameter (decay).

    Arguments
    ---------
    loc : float
        The location parameter, representing the peak of the distribution.
    scale : float
        The scale parameter, representing the exponential decay. Must be non-negative.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the Laplace distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.laplace

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.laplace(loc=0.0, scale=1.0, size=3)
    array([0.2, -1.5, 0.8])  # random
    """
    return ndarray(_laplace(loc, scale, _convert_size(size)))


def logistic(loc: float, scale: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a logistic distribution.

    Generates random samples from a logistic distribution.
    This distribution is similar to the normal distribution but has heavier tails.

    Arguments
    ---------
    loc : float
        The location parameter of the distribution.
    scale : float
        The scale parameter of the distribution. Must be non-negative.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the logistic distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.logistic

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.logistic(loc=0.0, scale=1.0, size=3)
    array([0.5, -0.2, 1.1])  # random
    """
    return ndarray(_logistic(loc, scale, _convert_size(size)))


def lognormal(mean: float, sigma: float, size: ShapeLike) -> ndarray:
    """
    Draw samples from a log-normal distribution.

    Generates random samples from a log-normal distribution.
    A variable is log-normally distributed if its natural logarithm is normally distributed.

    Arguments
    ---------
    mean : float
        The mean of the underlying normal distribution.
    sigma : float
        The standard deviation of the underlying normal distribution. Must be non-negative.
    size : int or sequence of ints
        The shape of the output array.

    Returns
    -------
    asnumpy.ndarray
        An array of random samples drawn from the log-normal distribution.
        The array may be allocated on the accelerator device.

    See Also
    --------
    numpy.random.lognormal

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.random.lognormal(mean=0.0, sigma=1.0, size=3)
    array([1.5, 0.6, 2.3])  # random
    """
    return ndarray(_lognormal(mean, sigma, _convert_size(size)))

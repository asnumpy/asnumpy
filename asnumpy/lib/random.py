from typing import Union, Sequence
import numpy as np
from .asnumpy_core.random import (
    pareto as ap_pareto,
    rayleigh as ap_rayleigh,
    normal as ap_normal,
    uniform as ap_uniform,
    standard_normal as ap_standard_normal,
    standard_cauchy as ap_standard_cauchy,
    weibull as ap_weibull,
    binomial as ap_binomial,
    exponential as ap_exponential,
    geometric as ap_geometric,
    gumbel as ap_gumbel,
    laplace as ap_laplace,
    logistic as ap_logistic,
    lognormal as ap_lognormal
)
from .utils import ndarray

def pareto(a: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_pareto(a, size))

def rayleigh(scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_rayleigh(scale, size))

def normal(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_normal(loc, scale, size))

def uniform(low: float, high: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_uniform(low, high, size))

def standard_normal(size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_standard_normal(size))

def standard_cauchy(size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_standard_cauchy(size))

def weibull(a: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_weibull(a, size))

def binomial(n: int, p: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_binomial(n, p, size))

def exponential(scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_exponential(scale, size))

def geometric(p: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_geometric(p, size))

def gumbel(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_gumbel(loc, scale, size))

def laplace(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_laplace(loc, scale, size))

def logistic(loc: float, scale: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_logistic(loc, scale, size))

def lognormal(mean: float, sigma: float, size: Union[int, Sequence[int]]) -> ndarray:
    return ndarray(ap_lognormal(mean, sigma, size))
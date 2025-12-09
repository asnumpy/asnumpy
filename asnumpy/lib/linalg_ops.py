from .asnumpy_core import (
    dot as ap_dot,
    inner as ap_inner,
    outer as ap_outer,
    vdot as ap_vdot,
    matmul as ap_matmul,
    einsum as ap_einsum
)
from .utils import ndarray

def dot(a: ndarray, b: ndarray) -> ndarray:
    return ndarray(ap_dot(a._impl, b._impl))

def inner(a: ndarray, b: ndarray) -> ndarray:
    return ndarray(ap_inner(a._impl, b._impl))

def outer(a: ndarray, b: ndarray) -> ndarray:
    return ndarray(ap_outer(a._impl, b._impl))

def vdot(a: ndarray, b: ndarray) -> ndarray:
    return ndarray(ap_vdot(a._impl, b._impl))

def matmul(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(ap_matmul(x1._impl, x2._impl))

def einsum(subscripts: str, *operands: ndarray) -> ndarray:
    return ndarray(ap_einsum(subscripts, *operands))
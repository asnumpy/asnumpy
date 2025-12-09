from typing import Optional, Union, Sequence
from .asnumpy_core.linalg import (
    matrix_power as ap_matrix_power,
    qr as ap_qr,
    norm as ap_norm,
    det as ap_det,
    slogdet as ap_slogdet,
    inv as ap_inv
)
from .utils import ndarray


def matrix_power(a: ndarray, n: int) -> ndarray:
    return ndarray(ap_matrix_power(a._impl, n))

def qr(a: ndarray, mode: str = "reduced") -> Union[ndarray, tuple]:
    arr, t = ap_qr(a._impl, mode)
    return [ndarray(arr),t]

def norm(a: ndarray, ord: Optional[Union[str, int, float]] = None,
         axis: Optional[Union[int, Sequence[int]]] = None,
         keepdims: bool = False) -> ndarray:
    return ndarray(ap_norm(a._impl, ord, axis, keepdims))

def det(a: ndarray) -> ndarray:
    return ndarray(ap_det(a._impl))

def slogdet(a: ndarray) -> tuple:
    return ap_slogdet(a._impl)

def inv(a: ndarray) -> ndarray:
    return ndarray(ap_inv(a._impl))
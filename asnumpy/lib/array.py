from typing import Union, Optional, Sequence, Any
import numpy as np
from .asnumpy_core.array import (
    zeros as ap_zeros,
    zeros_like as ap_zeros_like,
    full as ap_full,
    full_like as ap_full_like,
    empty as ap_empty,
    empty_like as ap_empty_like,
    eye as ap_eye,
    ones as ap_ones,
    ones_like as ap_ones_like,
    identity as ap_identity
)
from .utils import ndarray, _convert_dtype


def zeros(shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_zeros(shape, _convert_dtype(dtype)))

def zeros_like(other: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    other_impl = other._impl if isinstance(other, ndarray) else other
    return ndarray(ap_zeros_like(other_impl, _convert_dtype(dtype)))

def full(shape: Union[int, Sequence[int]], value: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_full(shape, value, _convert_dtype(dtype)))

def full_like(other: Any, value: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    other_impl = other._impl if isinstance(other, ndarray) else other
    return ndarray(ap_full_like(other_impl, value, _convert_dtype(dtype)))

def empty(shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_empty(shape, _convert_dtype(dtype)))

def empty_like(prototype: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    prototype_impl = prototype._impl if isinstance(prototype, ndarray) else prototype
    return ndarray(ap_empty_like(prototype_impl, _convert_dtype(dtype)))

def eye(n: int, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_eye(n, _convert_dtype(dtype)))

def ones(shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_ones(shape, _convert_dtype(dtype)))

def ones_like(other: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    other_impl = other._impl if isinstance(other, ndarray) else other
    return ndarray(ap_ones_like(other_impl, _convert_dtype(dtype)))

def identity(n: int, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(ap_identity(n, _convert_dtype(dtype)))
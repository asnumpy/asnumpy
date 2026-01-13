from typing import Union, Optional, Sequence, Any
import numpy as np
from .lib.asnumpy_core.array import (
    empty as _ap_empty,
    empty_like as _ap_empty_like,
    eye as _ap_eye,
    full as _ap_full,
    full_like as _ap_full_like,
    identity as _ap_identity,
    linspace as _ap_linspace,
    ones as _ap_ones,
    ones_like as _ap_ones_like,
    zeros as _ap_zeros,
    zeros_like as _ap_zeros_like,
)
# 引入转换函数
from .utils import ndarray, _convert_dtype, _convert_size

def zeros(
    shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None
) -> ndarray:
    # 核心修复在这里：_convert_size(shape)
    raw_obj = _ap_zeros(_convert_size(shape), _convert_dtype(dtype))
    return ndarray(raw_obj)

def zeros_like(other: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    raw_obj = _ap_zeros_like(other, _convert_dtype(dtype))
    return ndarray(raw_obj)

def full(
    shape: Union[int, Sequence[int]], value: Any, dtype: Optional[np.dtype] = None
) -> ndarray:
    raw_obj = _ap_full(_convert_size(shape), value, _convert_dtype(dtype))
    return ndarray(raw_obj)

def full_like(other: Any, value: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    raw_obj = _ap_full_like(other, value, _convert_dtype(dtype))
    return ndarray(raw_obj)

def empty(
    shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None
) -> ndarray:
    raw_obj = _ap_empty(_convert_size(shape), _convert_dtype(dtype))
    return ndarray(raw_obj)

def empty_like(prototype: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    raw_obj = _ap_empty_like(prototype, _convert_dtype(dtype))
    return ndarray(raw_obj)

def eye(n: int, dtype: Optional[np.dtype] = None) -> ndarray:
    # eye 本身就接受 int，不需要 _convert_size
    raw_obj = _ap_eye(n, _convert_dtype(dtype))
    return ndarray(raw_obj)

def ones(shape: Union[int, Sequence[int]], dtype: Optional[np.dtype] = None) -> ndarray:
    raw_obj = _ap_ones(_convert_size(shape), _convert_dtype(dtype))
    return ndarray(raw_obj)

def ones_like(other: Any, dtype: Optional[np.dtype] = None) -> ndarray:
    raw_obj = _ap_ones_like(other, _convert_dtype(dtype))
    return ndarray(raw_obj)

def identity(n: int, dtype: Optional[np.dtype] = None) -> ndarray:
    raw_obj = _ap_identity(n, _convert_dtype(dtype))
    return ndarray(raw_obj)

def linspace(
    start: Union[int, float],
    end: Union[int, float],
    steps: int = 50,
    dtype: Optional[np.dtype] = None,
) -> ndarray:
    raw_obj = _ap_linspace(start, end, steps, _convert_dtype(dtype))
    return ndarray(raw_obj)
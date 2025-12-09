from typing import Sequence
import numpy as np
from .asnumpy_core import ndarray as _ndarray
from .asnumpy_core import broadcast_shape as _broadcast_shape


class ndarray:
    def __init__(self, shape: Sequence[int], dtype: np.dtype):
        self._impl = _ndarray(shape, dtype)
    
    def __init__(self, other: _ndarray):
        self._impl = other
    
    @property
    def shape(self) -> tuple:
        return tuple(self._impl.shape)
    
    @property
    def dtype(self) -> np.dtype:
        return self._impl.dtype
    
    @property
    def aclDtype(self) -> int:
        return self._impl.aclDtype
    
    def to_numpy(self) -> np.ndarray:
        return self._impl.to_numpy()
    
    @classmethod
    def from_numpy(cls, host_data: np.ndarray) -> 'ndarray':
        result = cls.__new__(cls)
        result._impl = _ndarray.from_numpy(host_data)
        return result
    
    def __repr__(self) -> str:
        return f"ndarray(shape={self.shape}, dtype={self.dtype})"
    
    def __str__(self) -> str:
        return self.__repr__()


def broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple:
    return _broadcast_shape(shape_a, shape_b)


def _convert_dtype(dtype):
    """Convert dtype parameter to appropriate format if needed"""
    if dtype is None:
        return None
    if not isinstance(dtype, np.dtype):
        return np.dtype(dtype)
    return dtype


__all__ = ['ndarray', 'broadcast_shape']
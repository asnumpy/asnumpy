from typing import Optional, Union, Sequence
from .asnumpy_core.sorting import sort as ap_sort
from .utils import ndarray

def sort(a: ndarray, axis: Optional[int] = -1, stable: bool = False) -> ndarray:
    return ndarray(ap_sort(a._impl, axis, stable))
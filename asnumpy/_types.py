from typing import Union, Sequence, TypeVar
import numpy as np


ArrayLike = Union[
    "ndarray",  # NPUArray
    np.ndarray,
    int,
    float,
    complex,
    bool,
    Sequence,
]

DTypeLike = Union[np.dtype, str, type, None]

ShapeLike = Union[int, Sequence[int]]

AxisLike = Union[int, Sequence[int], None]

AxisOptional = Union[int, Sequence[int], None]

ScalarLike = Union[int, float, complex, bool]

T = TypeVar("T")


__all__ = [
    "ArrayLike",
    "DTypeLike",
    "ShapeLike",
    "AxisLike",
    "AxisOptional",
    "ScalarLike",
    "T",
]

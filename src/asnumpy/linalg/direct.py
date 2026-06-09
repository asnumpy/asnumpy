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

import numpy as np

from .._conversion import asarray, asnumpy
from .._core import (
    dot as _dot,
)
from .._core import (
    einsum as _einsum,
)
from .._core import (
    vdot as _vdot,
)
from .._types import ArrayLike
from ..utils import ndarray


def _requires_fp64_fallback(*arrays: ArrayLike) -> bool:
    return any(asnumpy(arr).dtype == np.float64 for arr in arrays)


def dot(a: ArrayLike, b: ArrayLike) -> ndarray:
    if _requires_fp64_fallback(a, b):
        return asarray(np.dot(asnumpy(a), asnumpy(b)))
    return ndarray(_dot(a, b))


def inner(a: ArrayLike, b: ArrayLike) -> ndarray:
    return asarray(np.inner(asnumpy(a), asnumpy(b)))


def outer(a: ArrayLike, b: ArrayLike) -> ndarray:
    return asarray(np.outer(asnumpy(a), asnumpy(b)))


def vdot(a: ArrayLike, b: ArrayLike) -> ndarray:
    if _requires_fp64_fallback(a, b):
        return asarray(np.vdot(asnumpy(a), asnumpy(b)))
    return ndarray(_vdot(a, b))


def matmul(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return asarray(np.matmul(asnumpy(x1), asnumpy(x2)))


def einsum(subscripts: str, *operands: ArrayLike) -> ndarray:
    return ndarray(_einsum(subscripts, *operands))


_direct_all_ = [
    "dot",
    "einsum",
    "inner",
    "matmul",
    "outer",
    "vdot",
]

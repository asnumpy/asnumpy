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

from ..lib.asnumpy_core import (
    dot as _dot,
    vdot as _vdot,
    matmul as _matmul,
    einsum as _einsum,
)
from ..utils import ndarray
from .._types import ArrayLike


def dot(a: ArrayLike, b: ArrayLike) -> ndarray:
    return ndarray(_dot(a, b))


def inner(a: ArrayLike, b: ArrayLike) -> ndarray:
    na = a.to_numpy() if hasattr(a, 'to_numpy') else np.asarray(a)
    nb = b.to_numpy() if hasattr(b, 'to_numpy') else np.asarray(b)
    return ndarray.from_numpy(np.asarray(np.inner(na, nb)))


def outer(a: ArrayLike, b: ArrayLike) -> ndarray:
    na = a.to_numpy() if hasattr(a, 'to_numpy') else np.asarray(a)
    nb = b.to_numpy() if hasattr(b, 'to_numpy') else np.asarray(b)
    return ndarray.from_numpy(np.asarray(np.outer(na, nb)))


def vdot(a: ArrayLike, b: ArrayLike) -> ndarray:
    return ndarray(_vdot(a, b))


def matmul(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_matmul(x1, x2))


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

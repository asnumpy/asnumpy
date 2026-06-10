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

from .._core import (
    dot as _dot,
)
from .._core import (
    einsum as _einsum,
)
from .._core import (
    vdot as _vdot,
)
from .._fallback import fallback_to_numpy
from .._types import ArrayLike
from ..utils import as_host_array, ndarray, to_asnumpy_array


def _requires_fp64_fallback(*arrays: ArrayLike) -> bool:
    return any(np.asarray(as_host_array(arr)).dtype == np.float64 for arr in arrays)


@fallback_to_numpy
def dot(a: ArrayLike, b: ArrayLike) -> ndarray:
    if _requires_fp64_fallback(a, b):
        return to_asnumpy_array(np.dot(as_host_array(a), as_host_array(b)))
    return ndarray(_dot(a, b))


def inner(a: ArrayLike, b: ArrayLike) -> ndarray:
    na = as_host_array(a)
    nb = as_host_array(b)
    return ndarray.from_numpy(np.asarray(np.inner(na, nb)))


def outer(a: ArrayLike, b: ArrayLike) -> ndarray:
    na = as_host_array(a)
    nb = as_host_array(b)
    return ndarray.from_numpy(np.asarray(np.outer(na, nb)))


@fallback_to_numpy
def vdot(a: ArrayLike, b: ArrayLike) -> ndarray:
    if _requires_fp64_fallback(a, b):
        return to_asnumpy_array(np.vdot(as_host_array(a), as_host_array(b)))
    return ndarray(_vdot(a, b))


def matmul(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    na = as_host_array(x1)
    nb = as_host_array(x2)
    return ndarray.from_numpy(np.asarray(np.matmul(na, nb)))


@fallback_to_numpy
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

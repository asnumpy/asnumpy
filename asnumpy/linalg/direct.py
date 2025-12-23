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

from ..lib.asnumpy_core import (
    dot as _ap_dot,
    inner as _ap_inner,
    outer as _ap_outer,
    vdot as _ap_vdot,
    matmul as _ap_matmul,
    einsum as _ap_einsum,
)
from ..utils import ndarray


def dot(a: ndarray, b: ndarray) -> ndarray:
    return ndarray(_ap_dot(a, b))


def inner(a: ndarray, b: ndarray) -> ndarray:
    return ndarray(_ap_inner(a, b))


def outer(a: ndarray, b: ndarray) -> ndarray:
    return ndarray(_ap_outer(a, b))


def vdot(a: ndarray, b: ndarray) -> ndarray:
    return ndarray(_ap_vdot(a, b))


def matmul(x1: ndarray, x2: ndarray) -> ndarray:
    return ndarray(_ap_matmul(x1, x2))


def einsum(subscripts: str, *operands: ndarray) -> ndarray:
    return ndarray(_ap_einsum(subscripts, *operands))


_direct_all_ = [
    "dot",
    "einsum",
    "inner",
    "matmul",
    "outer",
    "vdot",
]

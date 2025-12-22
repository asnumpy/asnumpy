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

from typing import Optional, Union, Sequence
import numpy as np
from ..lib.asnumpy_core.linalg import (
    det as ap_det,
    inv as ap_inv,
    matrix_power as ap_matrix_power,
    norm as ap_norm,
    slogdet as ap_slogdet,
)
from ..utils import ndarray


def matrix_power(a: ndarray, n: int) -> ndarray:
    return ndarray(ap_matrix_power(a, n))


def qr(a: ndarray, mode: str = "reduced") -> Union[ndarray, tuple]:
    return ndarray.from_numpy(np.linalg.qr(a, mode))


def norm(
    a: ndarray,
    ord: Optional[Union[str, int, float]] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> ndarray:
    return ndarray(ap_norm(a, ord, axis, keepdims))


def det(a: ndarray) -> ndarray:
    return ndarray(ap_det(a))


def slogdet(a: ndarray) -> tuple:
    return ap_slogdet(a)


def inv(a: ndarray) -> ndarray:
    return ndarray(ap_inv(a))

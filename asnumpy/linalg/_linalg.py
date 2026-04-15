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

from typing import Optional, Union
import numpy as np
from ..lib.asnumpy_core.linalg import (
    det as _det,
    inv as _inv,
    matrix_power as _matrix_power,
    norm as _norm,
    qr as _qr,
    slogdet as _slogdet,
)
from ..utils import ndarray
from .._types import ArrayLike, AxisLike


def matrix_power(a: ArrayLike, n: int) -> ndarray:
    return ndarray(_matrix_power(a, n))


def qr(a: ArrayLike, mode: str = "reduced") -> Union[ndarray, tuple]:
    result = _qr(a, mode)
    if isinstance(result, tuple):
        q, r = result
        return (ndarray(q), ndarray(r))
    return ndarray(result)


def norm(
    a: ArrayLike,
    ord: Optional[Union[str, int, float]] = None,
    axis: AxisLike = None,
    keepdims: bool = False,
) -> ndarray:
    return ndarray(_norm(a, ord, axis, keepdims))


def det(a: ArrayLike) -> ndarray:
    return ndarray(_det(a))


def slogdet(a: ArrayLike) -> tuple:
    # CANN's double-precision slogdet may produce different results from NumPy
    # for inputs containing nan/inf. Fall back to NumPy for such cases.
    host = a.to_numpy() if hasattr(a, 'to_numpy') else np.asarray(a)
    if np.issubdtype(host.dtype, np.floating) and (np.any(np.isnan(host)) or np.any(np.isinf(host))):
        return np.linalg.slogdet(host)
    sign, logdet = _slogdet(a)
    return (ndarray(sign), ndarray(logdet))


def inv(a: ArrayLike) -> ndarray:
    return ndarray(_inv(a))

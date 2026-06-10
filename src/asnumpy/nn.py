# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

from ._core.nn import softmax as _softmax
from ._fallback import fallback_to_numpy
from ._types import ArrayLike, DTypeLike
from .utils import _convert_dtype, ndarray


def _numpy_softmax(x, axis=-1, dtype=None):
    """NumPy softmax implementation for fallback."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    s = np.sum(e, axis=axis, keepdims=True)
    result = e / s
    if dtype is not None:
        result = result.astype(dtype)
    return result


@fallback_to_numpy(numpy_func=_numpy_softmax)
def softmax(x: ArrayLike, axis: int = -1, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_softmax(x, axis, _convert_dtype(dtype)))

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

from typing import Optional
import numpy as np
from .lib.asnumpy_core.nn import softmax as _ap_softmax
from .utils import ndarray, _convert_dtype


def softmax(
    x: ndarray, axis: int = -1, dtype: Optional[np.dtype] = None
) -> ndarray:
    return ndarray(_ap_softmax(x, axis, _convert_dtype(dtype)))

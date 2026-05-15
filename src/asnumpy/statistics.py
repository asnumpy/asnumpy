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

from ._core.statistics import mean as _mean
from ._registry import register_op
from ._types import ArrayLike, AxisLike, DTypeLike
from .utils import _convert_dtype, ndarray, validate_axis


@register_op("mean", module="statistics", category="reduction")
def mean(
    a: ArrayLike,
    axis: AxisLike = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> ndarray | float:
    """Compute the arithmetic mean along the specified axis.

    NumPy-compatible interface running the reduction on Ascend NPU.

    Args:
        a: Input array.
        axis: Axis or axes along which the means are computed.
        keepdims: If True, the reduced axes are retained as size-1 dimensions.
        dtype: Type to use in computing the mean.

    Returns:
        ndarray or Python scalar (when axis=None).

    Raises:
        np.exceptions.AxisError: axis is out of bounds.
    """
    # Convert to ndarray if needed (C++ _mean expects NPUArray)
    if isinstance(a, ndarray):
        arr = a
    elif isinstance(a, np.ndarray):
        arr = ndarray.from_numpy(a)
    else:
        arr = ndarray.from_numpy(np.asarray(a))

    ndim = arr.ndim

    if axis is not None:
        validated_axis = validate_axis(axis, ndim)
    else:
        validated_axis = None

    out_dtype = _convert_dtype(dtype)
    if validated_axis is None:
        result = _mean(arr)
        target_dt = out_dtype if out_dtype is not None else arr.dtype
        if np.issubdtype(target_dt, np.floating):
            return target_dt.type(result)
        return result
    return ndarray(_mean(arr, validated_axis, keepdims, out_dtype))

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

import operator
import warnings

import numpy as np

from ._core import ndarray as _core_ndarray
from ._core.statistics import mean as _mean
from ._types import DTypeLike
from .utils import ndarray


def _normalize_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    """Normalize a NumPy-style axis argument without touching device data.

    ``None`` expands to all dimensions while ``()`` remains empty.  Keeping
    those cases distinct is important because CANN interprets an empty axis
    array as "reduce all", whereas NumPy defines ``axis=()`` as no reduction.
    """

    if axis is None:
        return tuple(range(ndim))

    raw_axes = axis if isinstance(axis, tuple) else (axis,)
    normalized: list[int] = []
    for raw_axis in raw_axes:
        # Python bool is an int subclass, but NumPy does not accept it as a
        # reduction axis.  Reject NumPy's scalar bool for the same reason.
        if isinstance(raw_axis, (bool, np.bool_)):
            raise TypeError("an integer is required for each axis")
        try:
            value = operator.index(raw_axis)
        except TypeError:
            raise TypeError("an integer is required for each axis") from None

        original = value
        if value < 0:
            value += ndim
        if value < 0 or value >= ndim:
            raise np.exceptions.AxisError(original, ndim=ndim)
        if value in normalized:
            raise ValueError("duplicate value in 'axis'")
        normalized.append(value)

    return tuple(normalized)


def _normalize_keepdims(keepdims: bool) -> bool:
    """Normalize ``keepdims`` using NumPy's integer-index protocol."""

    # NumPy accepts Python bool and any integer implementing ``__index__``,
    # then applies normal truth-value conversion.  np.bool_ deliberately does
    # not participate in that protocol for reduction keyword arguments.
    if isinstance(keepdims, np.bool_):
        raise TypeError("keepdims must be an integer")
    try:
        return bool(operator.index(keepdims))
    except TypeError:
        raise TypeError("keepdims must be an integer") from None


def _mean_dtypes(input_dtype: DTypeLike, dtype: DTypeLike) -> tuple[np.dtype, np.dtype]:
    """Resolve separate accumulator and result dtypes, following CuPy."""

    source = np.dtype(input_dtype)
    if dtype is None:
        if source.kind in "biu":
            promoted = np.dtype(np.float64)
            return promoted, promoted
        if source == np.dtype(np.float16):
            return np.dtype(np.float32), source
        return source, source

    result = np.dtype(dtype)
    if result.kind in "biu":
        return np.dtype(np.float64), result
    return result, result


def _reduction_shape(
    shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool
) -> tuple[int, ...]:
    """Return the exact output shape for a normalized reduction."""

    reduced = set(axes)
    if keepdims:
        return tuple(1 if index in reduced else extent for index, extent in enumerate(shape))
    return tuple(extent for index, extent in enumerate(shape) if index not in reduced)


def mean(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike = None,
    out: ndarray | None = None,
    keepdims: bool = False,
) -> ndarray:
    """Return the arithmetic mean along one or more axes.

    The parameter order and return convention match :func:`cupy.mean`.  In
    particular, a full reduction returns a zero-dimensional device array, not
    a Python float, so computing a scalar does not force a device-to-host copy.

    Integer and boolean inputs use ``float64`` by default.  A default
    ``float16`` result accumulates in ``float32``; an explicitly requested
    integer or boolean result accumulates in ``float64``, following CuPy.
    ``axis`` accepts ``None``, one integer, or a tuple of unique integers.
    When ``out`` is supplied, it must have exactly the result shape; its dtype
    controls final storage and the same object is returned.

    Args:
        a: Input asnumpy array.
        axis: Axis or axes to reduce. ``None`` reduces all dimensions and
            ``()`` performs no reduction.
        dtype: Requested result dtype; accumulation may use a wider dtype.
        out: Destination asnumpy array.
        keepdims: Retain reduced dimensions with extent one.
    """

    if not isinstance(a, _core_ndarray):
        raise TypeError("mean input must be an asnumpy.ndarray")

    normalized_keepdims = _normalize_keepdims(keepdims)
    normalized_axes = _normalize_axes(axis, a.ndim)
    compute_dtype, result_dtype = _mean_dtypes(a.dtype, dtype)
    if normalized_axes and any(a.shape[index] == 0 for index in normalized_axes):
        warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

    if out is not None:
        if not isinstance(out, ndarray):
            raise TypeError("out must be an asnumpy.ndarray")
        expected_shape = _reduction_shape(a.shape, normalized_axes, normalized_keepdims)
        if out.shape != expected_shape:
            raise ValueError(
                f"output shape {out.shape} does not match mean result shape {expected_shape}"
            )

    result = _mean(
        a,
        normalized_axes,
        normalized_keepdims,
        compute_dtype,
        result_dtype,
        out,
    )
    if out is not None:
        return out
    return ndarray(result)

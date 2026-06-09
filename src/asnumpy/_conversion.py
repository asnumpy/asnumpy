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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._core import ndarray as _core_ndarray
from ._dtype import normalize_dtype

if TYPE_CHECKING:
    from .utils import ndarray


def array(obj: Any, dtype: Any | None = None, copy: bool = True) -> ndarray:
    """Create an asnumpy ``ndarray`` from *obj*.

    Parameters
    ----------
    obj : array_like
        Data to convert.  May be a list, tuple, NumPy ndarray, or any
        object exposing a ``to_numpy()`` method.
    dtype : dtype-like, optional
        Desired output dtype.
    copy : bool, default ``True``
        If ``True``, the data is always copied.  If ``False`` a
        no-copy path is attempted when *obj* is already an
        ``asnumpy.ndarray`` with matching *dtype*.
    """
    from .utils import ndarray

    # Fast path: already an asnumpy ndarray with matching dtype and no-copy.
    if isinstance(obj, ndarray) and dtype is None and not copy:
        return obj

    # Convert to host NumPy array first.
    if isinstance(obj, _core_ndarray):
        host = obj.to_numpy()
    else:
        host = np.array(obj, dtype=normalize_dtype(dtype), copy=copy)

    # Ensure dtype consistency when an explicit dtype was requested.
    if dtype is not None and host.dtype != np.dtype(dtype):
        host = host.astype(np.dtype(dtype), copy=False)

    # Ensure contiguous layout required by device transfer.
    if not host.flags.c_contiguous:
        host = np.ascontiguousarray(host)
    return ndarray.from_numpy(host)


def asarray(obj: Any, dtype: Any | None = None, copy: bool | None = None) -> ndarray:
    """Convert *obj* to an asnumpy ``ndarray`` (avoiding copies when possible).

    Unlike :func:`array`, this defaults to **not** copying data.
    """
    from .utils import ndarray

    target_dtype = normalize_dtype(dtype)

    if isinstance(obj, ndarray):
        if target_dtype is None or obj.dtype == target_dtype:
            if copy is True:
                return ndarray(obj)  # explicit copy
            return obj  # reuse — no copy
        # dtype mismatch: need conversion.
        return array(asnumpy(obj), dtype=target_dtype, copy=True)

    host = np.asarray(obj, dtype=target_dtype)
    if copy is True:
        host = host.copy()
    # Ensure contiguous layout required by device transfer.
    if not host.flags.c_contiguous:
        host = np.ascontiguousarray(host)
    return ndarray.from_numpy(host)


def asnumpy(obj: Any) -> np.ndarray:
    """Return *obj* as a NumPy ``ndarray``.

    If *obj* has a ``to_numpy()`` method it will be used; otherwise
    ``np.asarray`` is called as a fallback.
    """
    if hasattr(obj, "to_numpy"):
        return obj.to_numpy()
    return np.asarray(obj)

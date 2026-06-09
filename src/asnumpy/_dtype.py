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

from typing import Any

import numpy as np

# Set of dtypes fully supported by asnumpy / CANN.
SUPPORTED_DTYPES = frozenset(
    {
        np.dtype(t)
        for t in (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    }
)


def normalize_dtype(dtype: Any | None) -> np.dtype | None:
    """Coerce *dtype* to ``np.dtype`` or return ``None``."""
    if dtype is None:
        return None
    return np.dtype(dtype)


def dtype_of(value: Any) -> np.dtype:
    """Return the dtype of *value* (array-like or scalar)."""
    if hasattr(value, "dtype"):
        return np.dtype(value.dtype)
    return np.asarray(value).dtype


def result_dtype(*values: Any, dtype: Any | None = None) -> np.dtype:
    """Resolve the result dtype following NumPy promotion rules.

    If an explicit *dtype* is given it is validated and returned;
    otherwise ``np.result_type`` is used to promote the input dtypes.
    """
    explicit = normalize_dtype(dtype)
    if explicit is not None:
        return require_supported_dtype(explicit)
    resolved = np.result_type(
        *[dtype_of(v) if hasattr(v, "dtype") else v for v in values]
    )
    return require_supported_dtype(resolved)


def require_supported_dtype(dtype: Any) -> np.dtype:
    """Return *dtype* as ``np.dtype`` or raise ``TypeError`` if unsupported."""
    normalized = np.dtype(dtype)
    if normalized not in SUPPORTED_DTYPES:
        raise TypeError(f"dtype {normalized} is not supported by asnumpy")
    return normalized


def can_cast_to_output(from_dtype: Any, to_dtype: Any, casting: str = "same_kind") -> bool:
    """Check whether *from_dtype* can be safely cast to *to_dtype*."""
    return bool(np.can_cast(np.dtype(from_dtype), np.dtype(to_dtype), casting=casting))

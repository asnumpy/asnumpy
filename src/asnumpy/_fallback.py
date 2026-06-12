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

import functools
from typing import TYPE_CHECKING, overload

import numpy as np

from ._config import get_fallback_state, warn_copy
from .utils import ndarray, to_asnumpy_array

if TYPE_CHECKING:
    pass


def fallback_to_numpy(func=None, *, numpy_func=None):
    """Decorator that automatically falls back to NumPy when an NPU operator fails.

    Wraps a function that calls a C++ ``_core`` operator.  If the wrapped function
    raises a ``RuntimeError`` (NPU/CANN operator failure or ``NotImplementedError``)
    and fallback is enabled (see :func:`asnumpy.auto_fallback`), the decorator
    converts inputs to host NumPy arrays, calls the equivalent ``numpy.<funcname>``
    function, and optionally copies the result back to the NPU.  Other exception
    types (``TypeError``, ``ValueError``, …) propagate unchanged.

    Parameters
    ----------
    numpy_func : callable, optional
        The NumPy function to use as fallback.  When *None* (the default),
        ``getattr(np, func.__name__)`` is used, which works for every operator
        whose asnumpy name matches its NumPy name (e.g. ``sin`` → ``np.sin``).

    Examples
    --------
    >>> @fallback_to_numpy
    ... def sin(x):
    ...     return ndarray(_sin(x))

    >>> @fallback_to_numpy(numpy_func=np.random.exponential)
    ... def exponential(scale=1.0, size=None):
    ...     return ndarray(_exponential(scale, size))
    """

    if func is None:
        return lambda f: fallback_to_numpy(f, numpy_func=numpy_func)

    _numpy_func = numpy_func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError:
            # NPU/CANN operator failures surface as RuntimeError (CannError is a
            # subclass) and "not implemented" as NotImplementedError (also a
            # RuntimeError subclass).  Narrow on purpose: TypeError/ValueError/
            # IndexError are genuine bugs or bad inputs and must not be silently
            # masked by a CPU fallback.
            state = get_fallback_state()
            if not state.enabled:
                raise

            np_func = _numpy_func or getattr(np, func.__name__)

            # Only NPUArrays need copying to host; everything else (scalars,
            # axis ints, einsum subscript strings, raw lists) is passed through
            # untouched — NumPy accepts these natively.  Coercing them with
            # np.asarray would, e.g., turn an einsum subscript string into a
            # 0-d array that np.einsum rejects.
            np_args = [_to_host(a) for a in args]
            np_kwargs = {k: _to_host(v) for k, v in kwargs.items()}

            warn_copy(f"falling back to numpy.{func.__name__}")

            # Many asnumpy operators carry extra kwargs (e.g. ``dtype``) that
            # the corresponding NumPy function does not accept.  Try the full
            # call first; fall back to positional-only on TypeError.
            try:
                result = np_func(*np_args, **np_kwargs)
            except TypeError:
                result = np_func(*np_args)

            if state.return_device:
                return _wrap_asnumpy(result)
            return result

    return wrapper


def _to_host(v):
    """Convert an NPUArray to a host numpy array; pass everything else through."""
    if hasattr(v, "to_numpy"):
        return v.to_numpy()
    return v


@overload
def _wrap_asnumpy(value: np.ndarray) -> ndarray: ...


@overload
def _wrap_asnumpy(value: tuple[np.ndarray, ...]) -> tuple[ndarray, ...]: ...


def _wrap_asnumpy(value):
    """Wrap a numpy array or tuple of arrays back to asnumpy ndarray(s)."""
    if isinstance(value, tuple):
        return tuple(to_asnumpy_array(v) for v in value)
    return to_asnumpy_array(value)

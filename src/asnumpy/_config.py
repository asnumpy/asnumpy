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

import contextvars
import warnings
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class FallbackState:
    enabled: bool = False
    return_device: bool = True
    warn_on_copy: bool = True


_fallback_state: contextvars.ContextVar[FallbackState | None] = contextvars.ContextVar(
    "asnumpy_fallback_state",
    default=None,
)


def get_fallback_state() -> FallbackState:
    state = _fallback_state.get()
    if state is None:
        return FallbackState()
    return state


def auto_fallback(*, enable: bool, return_device: bool = True, warn_on_copy: bool = True) -> None:
    _fallback_state.set(
        FallbackState(
            enabled=bool(enable),
            return_device=bool(return_device),
            warn_on_copy=bool(warn_on_copy),
        )
    )


@contextmanager
def numpy_fallback(*, enable: bool, return_device: bool = True, warn_on_copy: bool = True):
    token = _fallback_state.set(
        FallbackState(
            enabled=bool(enable),
            return_device=bool(return_device),
            warn_on_copy=bool(warn_on_copy),
        )
    )
    try:
        yield
    finally:
        _fallback_state.reset(token)


def warn_copy(message: str) -> None:
    if get_fallback_state().warn_on_copy:
        warnings.warn(message, RuntimeWarning, stacklevel=2)

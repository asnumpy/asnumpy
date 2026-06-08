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

import warnings

import asnumpy as ap


def test_auto_fallback_defaults_are_strict():
    state = ap.get_fallback_state()
    assert state.enabled is False
    assert state.return_device is True
    assert state.warn_on_copy is True


def test_auto_fallback_sets_global_defaults():
    previous = ap.get_fallback_state()
    try:
        ap.auto_fallback(enable=True, return_device=False, warn_on_copy=False)
        state = ap.get_fallback_state()
        assert state.enabled is True
        assert state.return_device is False
        assert state.warn_on_copy is False
    finally:
        ap.auto_fallback(
            enable=previous.enabled,
            return_device=previous.return_device,
            warn_on_copy=previous.warn_on_copy,
        )


def test_numpy_fallback_context_restores_state():
    previous = ap.get_fallback_state()
    try:
        ap.auto_fallback(enable=False, return_device=True, warn_on_copy=True)
        with ap.numpy_fallback(enable=True, return_device=False, warn_on_copy=False):
            state = ap.get_fallback_state()
            assert state.enabled is True
            assert state.return_device is False
            assert state.warn_on_copy is False
        state = ap.get_fallback_state()
        assert state.enabled is False
        assert state.return_device is True
        assert state.warn_on_copy is True
    finally:
        ap.auto_fallback(
            enable=previous.enabled,
            return_device=previous.return_device,
            warn_on_copy=previous.warn_on_copy,
        )


def test_warn_copy_uses_runtime_warning():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        ap.warn_copy("test copy path")
    assert len(records) > 0, "Expected at least one RuntimeWarning"
    assert any(issubclass(item.category, RuntimeWarning) for item in records)
    assert "test copy path" in str(records[0].message)

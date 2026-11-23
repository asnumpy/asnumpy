# *****************************************************************************
# Copyright (c) 2026 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

"""Host regression tests for generic assert_equal with array operands."""

import importlib.util
from pathlib import Path

import numpy
import pytest


def _load_assertions():
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "asnumpy"
        / "testing"
        / "_assertions.py"
    )
    spec = importlib.util.spec_from_file_location("assertions_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assertions = _load_assertions()
assert_equal = assertions.assert_equal


def test_equal_arrays_pass():
    assert_equal(numpy.array([1, 2, 3]), numpy.array([1, 2, 3]))


def test_unequal_arrays_raise_assertion_error():
    with pytest.raises(AssertionError):
        assert_equal(numpy.array([1, 2, 3]), numpy.array([1, 2, 4]))


def test_unequal_shapes_raise_assertion_error():
    with pytest.raises(AssertionError):
        assert_equal(numpy.array([1, 2, 3]), numpy.array([[1, 2, 3]]))


def test_err_msg_included_in_failure():
    with pytest.raises(AssertionError, match="custom message"):
        assert_equal(numpy.array([1]), numpy.array([2]), err_msg="custom message")


def test_list_of_arrays():
    assert_equal(
        [numpy.array([1, 2]), numpy.array([3, 4])],
        [numpy.array([1, 2]), numpy.array([3, 4])],
    )
    with pytest.raises(AssertionError):
        assert_equal(
            [numpy.array([1, 2]), numpy.array([3, 4])],
            [numpy.array([1, 2]), numpy.array([3, 5])],
        )


def test_dict_of_arrays():
    assert_equal({"a": numpy.array([1, 2])}, {"a": numpy.array([1, 2])})
    with pytest.raises(AssertionError):
        assert_equal({"a": numpy.array([1, 2])}, {"a": numpy.array([1, 3])})


def test_scalars_shapes_and_types_still_work():
    assert_equal(3, 3)
    assert_equal((3, 4), (3, 4))
    assert_equal("abc", "abc")
    assert_equal(type(numpy.array([1])), numpy.ndarray)
    with pytest.raises(AssertionError):
        assert_equal(3, 4)
    with pytest.raises(AssertionError):
        assert_equal((3, 4), (3, 5))


def test_nan_at_same_position_is_equal():
    assert_equal(
        numpy.array([0.0, numpy.nan, numpy.inf]),
        numpy.array([0.0, numpy.nan, numpy.inf]),
    )
    assert_equal(
        {"out": numpy.array([numpy.nan])},
        {"out": numpy.array([numpy.nan])},
    )


def test_nan_at_different_positions_is_not_equal():
    with pytest.raises(AssertionError):
        assert_equal(
            numpy.array([numpy.nan, 0.0]),
            numpy.array([0.0, numpy.nan]),
        )

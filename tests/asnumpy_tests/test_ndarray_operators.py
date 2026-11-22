# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

"""Python arithmetic operator coverage for asnumpy.ndarray."""

import operator

import numpy as np
import pytest

import asnumpy as ap


@pytest.mark.parametrize(
    ("op", "reverse"),
    [
        (operator.add, False),
        (operator.sub, False),
        (operator.mul, False),
        (operator.truediv, False),
        (operator.floordiv, False),
        (operator.mod, False),
        (operator.pow, False),
        (operator.add, True),
        (operator.sub, True),
        (operator.mul, True),
        (operator.truediv, True),
        (operator.floordiv, True),
        (operator.mod, True),
        (operator.pow, True),
    ],
)
def test_binary_operators_match_numpy(op, reverse):
    host_a = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    host_b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    device_a = ap.asarray(host_a)
    device_b = ap.asarray(host_b)
    left, right = (device_b, device_a) if reverse else (device_a, device_b)
    expected_left, expected_right = (host_b, host_a) if reverse else (host_a, host_b)

    result = op(left, right)

    assert isinstance(result, ap.ndarray)
    np.testing.assert_allclose(result.to_numpy(), op(expected_left, expected_right))


@pytest.mark.parametrize(
    ("op", "reverse"),
    [
        (operator.add, False),
        (operator.sub, False),
        (operator.mul, False),
        (operator.truediv, False),
        (operator.add, True),
        (operator.sub, True),
        (operator.mul, True),
        (operator.truediv, True),
    ],
)
def test_scalar_operators_preserve_float32_for_weak_scalars(op, reverse):
    host = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    device = ap.asarray(host)
    left, right = (2, device) if reverse else (device, 2)
    expected_left, expected_right = (2, host) if reverse else (host, 2)

    result = op(left, right)

    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(result.to_numpy(), op(expected_left, expected_right))


def test_unary_operators_match_numpy():
    host = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    device = ap.asarray(host)

    np.testing.assert_array_equal((-device).to_numpy(), -host)
    np.testing.assert_array_equal((+device).to_numpy(), +host)
    np.testing.assert_array_equal(abs(device).to_numpy(), abs(host))


def test_operator_supports_numpy_operand_and_broadcasting():
    host = np.array([[1.0], [2.0]], dtype=np.float32)
    other = np.array([3.0, 4.0], dtype=np.float32)

    result = ap.asarray(host) + other

    np.testing.assert_array_equal(result.to_numpy(), host + other)

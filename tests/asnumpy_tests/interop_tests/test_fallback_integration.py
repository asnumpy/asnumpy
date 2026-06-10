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
from unittest import mock

import numpy as np
import pytest

import asnumpy as ap
from asnumpy._fallback import fallback_to_numpy


@pytest.fixture(autouse=True)
def _reset_fallback_state():
    yield
    ap.auto_fallback(enable=False, return_device=True, warn_on_copy=True)


class TestDecoratorCatchesAndPropagates:

    @staticmethod
    def test_catches_exception_and_dispatches_to_numpy():
        @fallback_to_numpy(numpy_func=np.sin)
        def _failing(x):
            raise RuntimeError("simulated NPU fault")

        ap.auto_fallback(enable=True, return_device=False, warn_on_copy=False)
        result = _failing([0.0, 1.0, 2.0])
        np.testing.assert_allclose(result, np.sin([0.0, 1.0, 2.0]))

    @staticmethod
    def test_disabled_fallback_re_raises_original_exception():
        @fallback_to_numpy(numpy_func=np.sin)
        def _failing(x):
            raise RuntimeError("simulated failure")

        ap.auto_fallback(enable=False)
        with pytest.raises(RuntimeError, match="simulated failure"):
            _failing([0.0])

    @staticmethod
    def test_disabled_fallback_still_returns_on_success():
        @fallback_to_numpy(numpy_func=np.sin)
        def _passthrough(x):
            return 42

        ap.auto_fallback(enable=False)
        assert _passthrough([0.0]) == 42


class TestReturnDevice:

    @staticmethod
    def test_false_returns_raw_numpy_array():
        @fallback_to_numpy(numpy_func=np.sin)
        def _failing(x):
            raise RuntimeError()

        ap.auto_fallback(enable=True, return_device=False, warn_on_copy=False)
        result = _failing([0.0, 1.0])
        assert isinstance(result, np.ndarray)
        assert not hasattr(result, "to_numpy")

    @staticmethod
    def test_true_wraps_result_back_to_npuarray():
        @fallback_to_numpy(numpy_func=np.sin)
        def _failing(x):
            raise RuntimeError()

        ap.auto_fallback(enable=True, return_device=True, warn_on_copy=False)
        result = _failing([0.0, 1.0])
        assert hasattr(result, "to_numpy"), f"Expected NPUArray, got {type(result)}"


class TestWarnCopy:

    @staticmethod
    def test_true_emits_runtime_warning():
        @fallback_to_numpy(numpy_func=np.sin)
        def _failing(x):
            raise RuntimeError()

        ap.auto_fallback(enable=True, warn_on_copy=True)
        with pytest.warns(RuntimeWarning, match="falling back"):
            _failing([0.0, 1.0])

    @staticmethod
    def test_false_no_runtime_warning():
        @fallback_to_numpy(numpy_func=np.cos)
        def _failing(x):
            raise RuntimeError()

        ap.auto_fallback(enable=True, warn_on_copy=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _failing([0.0, 1.0])
            fallback_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(fallback_warnings) == 0, f"Unexpected: {fallback_warnings}"


class TestScalarKwargsPassthrough:

    @staticmethod
    def test_axis_and_keepdims_forwarded():
        @fallback_to_numpy(numpy_func=np.sum)
        def _failing(a, axis=None, keepdims=False, dtype=None):
            raise RuntimeError()

        ap.auto_fallback(enable=True, return_device=False, warn_on_copy=False)
        data = [[1, 2], [3, 4]]
        result = _failing(data, axis=0, keepdims=True)
        expected = np.sum(data, axis=0, keepdims=True)
        np.testing.assert_allclose(result, expected)
        assert result.shape == expected.shape

    @staticmethod
    def test_dtype_is_forwarded():
        @fallback_to_numpy(numpy_func=np.mean)
        def _failing(a, axis=None, keepdims=False, dtype=None):
            raise RuntimeError()

        ap.auto_fallback(enable=True, return_device=False, warn_on_copy=False)
        result = _failing([1, 2, 3], dtype=np.float64)
        expected = np.mean([1, 2, 3], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    @staticmethod
    def test_dtype_none_is_preserved():
        @fallback_to_numpy(numpy_func=np.mean)
        def _failing(a, axis=None, keepdims=False, dtype=None):
            raise RuntimeError()

        ap.auto_fallback(enable=True, return_device=False, warn_on_copy=False)
        result = _failing([1, 2, 3])
        expected = np.mean([1, 2, 3])
        np.testing.assert_allclose(result, expected)


class TestCustomNumpyFunc:

    @staticmethod
    def test_lambda_function():
        @fallback_to_numpy(numpy_func=lambda x: np.maximum(x, 0))
        def _failing(x):
            raise RuntimeError()

        ap.auto_fallback(enable=True, return_device=False, warn_on_copy=False)
        result = _failing([-2.0, 0.0, 2.0])
        np.testing.assert_allclose(result, np.maximum([-2.0, 0.0, 2.0], 0))

    @staticmethod
    def test_namespaced_callable():
        @fallback_to_numpy(numpy_func=np.linalg.det)
        def _failing(a):
            raise RuntimeError()

        ap.auto_fallback(enable=True, return_device=False, warn_on_copy=False)
        result = _failing([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(result, np.linalg.det([[1.0, 2.0], [3.0, 4.0]]))


class TestContextManagerWithDecorator:

    @staticmethod
    def test_context_enables_fallback_for_failing_function():
        @fallback_to_numpy(numpy_func=np.sin)
        def _failing(x):
            raise RuntimeError()

        with pytest.raises(RuntimeError):
            _failing([0.0])

        with ap.numpy_fallback(enable=True, return_device=False, warn_on_copy=False):
            result = _failing([0.0, 1.0])
            np.testing.assert_allclose(result, np.sin([0.0, 1.0]))

        with pytest.raises(RuntimeError):
            _failing([0.0])

    @staticmethod
    def test_context_restores_state_on_internal_exception():
        ap.auto_fallback(enable=False)
        with pytest.raises(ValueError, match="unrelated"):
            with ap.numpy_fallback(enable=True):
                raise ValueError("unrelated error")
        assert ap.get_fallback_state().enabled is False


_FKW = dict(enable=True, return_device=False, warn_on_copy=False)


class TestMathOperatorsWired:

    @staticmethod
    def test_sin_fallback_when_npu_fails():
        with mock.patch("asnumpy.math._sin", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.sin([0.0, 1.0, 2.0])
            np.testing.assert_allclose(result, np.sin([0.0, 1.0, 2.0]))

    @staticmethod
    def test_add_fallback_when_npu_fails():
        with mock.patch("asnumpy.math._add", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.add([1, 2], [3, 4])
            np.testing.assert_allclose(result, np.add([1, 2], [3, 4]))

    @staticmethod
    def test_absolute_fallback_when_npu_fails():
        with mock.patch("asnumpy.math._absolute", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.absolute([-1.5, 0.0, 2.3])
            np.testing.assert_allclose(result, np.absolute([-1.5, 0.0, 2.3]))

    @staticmethod
    def test_gelu_custom_numpy_func_triggered():
        with mock.patch("asnumpy.math._gelu", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.gelu([0.0, 1.0, -1.0])
            x = np.asarray([0.0, 1.0, -1.0])
            expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
            np.testing.assert_allclose(result, expected, rtol=1e-6)

    @staticmethod
    def test_round_uses_numpy_round_not_round_():
        with mock.patch("asnumpy.math._round_", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.round_([1.23, 4.56], decimals=1)
            np.testing.assert_allclose(result, np.round([1.23, 4.56], decimals=1))


class TestLogicOperatorsWired:

    @staticmethod
    def test_logical_and_fallback():
        with mock.patch("asnumpy.logic._logical_and", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.logical_and([True, False], [True, True])
            np.testing.assert_array_equal(result, np.logical_and([True, False], [True, True]))

    @staticmethod
    def test_isfinite_fallback():
        with mock.patch("asnumpy.logic._isfinite", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.isfinite([0.0, np.inf, np.nan])
            np.testing.assert_array_equal(result, np.isfinite([0.0, np.inf, np.nan]))


class TestArrayCreationWired:

    @staticmethod
    def test_zeros_fallback_when_npu_fails():
        with mock.patch("asnumpy.array._zeros", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.zeros((2, 3), dtype=np.float32)
            np.testing.assert_allclose(result, np.zeros((2, 3), dtype=np.float32))

    @staticmethod
    def test_full_fallback_when_npu_fails():
        with mock.patch("asnumpy.array._full", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.full((2, 3), 7.0, dtype=np.float32)
            np.testing.assert_allclose(result, np.full((2, 3), 7.0, dtype=np.float32))


class TestLinalgOperatorsWired:

    @staticmethod
    def test_det_fallback_when_npu_fails():
        from asnumpy import linalg

        with mock.patch("asnumpy.linalg._linalg._det", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = linalg.det([[1.0, 2.0], [3.0, 4.0]])
            np.testing.assert_allclose(result, np.linalg.det([[1.0, 2.0], [3.0, 4.0]]))

    @staticmethod
    def test_inv_fallback_when_npu_fails():
        from asnumpy import linalg

        with mock.patch("asnumpy.linalg._linalg._inv", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = linalg.inv([[1.0, 2.0], [3.0, 4.0]])
            np.testing.assert_allclose(result, np.linalg.inv([[1.0, 2.0], [3.0, 4.0]]))


class TestRandomOperatorsWired:

    @staticmethod
    def test_uniform_fallback_when_npu_fails():
        with mock.patch("asnumpy.random._random._uniform", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.random.uniform(0, 1, size=10)
            assert result.shape == (10,)

    @staticmethod
    def test_exponential_fallback_when_npu_fails():
        with mock.patch(
            "asnumpy.random._random._exponential", side_effect=RuntimeError("NPU fail")
        ):
            ap.auto_fallback(**_FKW)
            result = ap.random.exponential(1.0, size=5)
            assert result.shape == (5,)


class TestTupleReturnOperatorsWired:

    @staticmethod
    def test_modf_fallback_when_npu_fails():
        with mock.patch("asnumpy.math._modf", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            frac, inte = ap.modf([1.5, 2.7, -3.2])
            expected_frac, expected_inte = np.modf([1.5, 2.7, -3.2])
            np.testing.assert_allclose(frac, expected_frac)
            np.testing.assert_allclose(inte, expected_inte)

    @staticmethod
    def test_divmod_fallback_when_npu_fails():
        with mock.patch("asnumpy.math._divmod", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            quot, rem = ap.divmod([10.0, 20.0], [3.0, 7.0])
            expected_quot, expected_rem = np.divmod([10.0, 20.0], [3.0, 7.0])
            np.testing.assert_allclose(quot, expected_quot)
            np.testing.assert_allclose(rem, expected_rem)


class TestSortingStatisticsWired:

    @staticmethod
    def test_sort_fallback_when_npu_fails():
        with mock.patch("asnumpy.sorting._sort", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.sort([3.0, 1.0, 2.0])
            np.testing.assert_allclose(result, np.sort([3.0, 1.0, 2.0]))

    @staticmethod
    def test_mean_fallback_when_npu_fails():
        with mock.patch("asnumpy.statistics._mean", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.mean([1.0, 2.0, 3.0, 4.0])
            np.testing.assert_allclose(result, np.mean([1.0, 2.0, 3.0, 4.0]))


class TestNnOperatorsWired:

    @staticmethod
    def test_softmax_fallback_when_npu_fails():
        with mock.patch("asnumpy.nn._softmax", side_effect=RuntimeError("NPU fail")):
            ap.auto_fallback(**_FKW)
            result = ap.softmax([1.0, 2.0, 3.0])
            x = np.asarray([1.0, 2.0, 3.0])
            x_max = x.max()
            expected = np.exp(x - x_max) / np.sum(np.exp(x - x_max))
            np.testing.assert_allclose(result, expected)


class TestDisabledFallbackTransparent:

    @staticmethod
    def test_sin_with_disabled_fallback_raises_on_npu_failure():
        ap.auto_fallback(enable=False)
        with mock.patch("asnumpy.math._sin", side_effect=RuntimeError("NPU fail")):
            with pytest.raises(RuntimeError, match="NPU fail"):
                ap.sin([0.0, 1.0])

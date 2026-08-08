"""Hardware-free contract tests for ``asnumpy.mean``."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def contract(monkeypatch):
    calls, raw_results = [], []

    class CoreArray:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = tuple(shape), np.dtype(dtype)
            self.ndim = len(self.shape)

    class Array(CoreArray):
        def __init__(self, value, dtype=None):
            if isinstance(value, CoreArray):
                super().__init__(value.shape, value.dtype)
                self.wrapped = value
            else:
                super().__init__(value, dtype)
                self.wrapped = None

    def backend_mean(a, axes, keepdims, compute_dtype, result_dtype, out):
        calls.append((a, axes, keepdims, compute_dtype, result_dtype, out))
        reduced = set(axes)
        shape = tuple(
            1 if keepdims and i in reduced else n
            for i, n in enumerate(a.shape)
            if keepdims or i not in reduced
        )
        raw = CoreArray(shape, out.dtype if out is not None else result_dtype)
        raw_results.append(raw)
        return raw

    def fake_module(name, **attributes):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        return module

    package_dir = Path(__file__).parents[3] / "src" / "asnumpy"
    package = fake_module("asnumpy")
    package.__path__ = [str(package_dir)]
    core = fake_module("asnumpy._core", ndarray=CoreArray)
    core.__path__ = []
    modules = {
        "asnumpy": package,
        "asnumpy._core": core,
        "asnumpy._core.statistics": fake_module("asnumpy._core.statistics", mean=backend_mean),
        "asnumpy._types": fake_module(
            "asnumpy._types", ArrayLike=object, AxisLike=object, DTypeLike=object
        ),
        "asnumpy.utils": fake_module("asnumpy.utils", ndarray=Array),
    }
    for name, fake in modules.items():
        monkeypatch.setitem(sys.modules, name, fake)

    spec = importlib.util.spec_from_file_location(
        "asnumpy.statistics", package_dir / "statistics.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return types.SimpleNamespace(module=module, Array=Array, calls=calls, raw=raw_results)


def test_public_signature(contract):
    parameters = inspect.signature(contract.module.mean).parameters
    assert list(parameters) == ["a", "axis", "dtype", "out", "keepdims"]
    defaults = [parameter.default for parameter in parameters.values()]
    assert defaults == [inspect.Parameter.empty, None, None, None, False]


def test_axis_normalization_and_validation(contract):
    normalize = contract.module._normalize_axes
    assert normalize(None, 3) == (0, 1, 2)
    assert normalize(None, 0) == ()
    assert normalize(-1, 3) == (2,)
    assert normalize(np.int64(-2), 3) == (1,)
    assert normalize((2, -3), 3) == (2, 0)
    assert normalize((), 3) == ()
    for axis in [3, -4, (0, 3)]:
        with pytest.raises(np.exceptions.AxisError):
            normalize(axis, 3)
    with pytest.raises(ValueError):
        normalize((0, -3, 0), 3)
    for axis in [1.5, [0], True, (0, 1.5)]:
        with pytest.raises(TypeError):
            normalize(axis, 3)


def test_keepdims_and_dtype_normalization(contract):
    for value in [False, True, 0, 1, 2, -1, np.int64(0), np.int64(2)]:
        assert contract.module._normalize_keepdims(value) is bool(value)
    for value in [np.bool_(False), np.bool_(True), None, "yes", 1.5]:
        with pytest.raises(TypeError, match="keepdims"):
            contract.module._normalize_keepdims(value)
    for dtype in [np.bool_, np.int32, np.uint64]:
        expected = (np.dtype(np.float64), np.dtype(np.float64))
        assert contract.module._mean_dtypes(np.dtype(dtype), None) == expected
    assert contract.module._mean_dtypes(np.dtype(np.float16), None) == (
        np.dtype(np.float32),
        np.dtype(np.float16),
    )
    for dtype in [np.float32, np.complex64]:
        expected = (np.dtype(dtype), np.dtype(dtype))
        assert contract.module._mean_dtypes(np.dtype(dtype), None) == expected
    for dtype in [np.bool_, np.int16, np.uint32]:
        expected = (np.dtype(np.float64), np.dtype(dtype))
        assert contract.module._mean_dtypes(np.dtype(np.float32), dtype) == expected
    for dtype in [np.float32, "float64", np.dtype("complex64")]:
        expected = (np.dtype(dtype), np.dtype(dtype))
        assert contract.module._mean_dtypes(np.dtype(np.int32), dtype) == expected


def test_reduction_shapes(contract):
    cases = (
        ((2, 3, 4), (0, 2), False, (3,)),
        ((2, 3, 4), (0, 2), True, (1, 3, 1)),
        ((2, 3), (0, 1), False, ()),
        ((2, 3), (), False, (2, 3)),
        ((2, 3), (), True, (2, 3)),
        ((), (), True, ()),
    )
    for shape, axes, keepdims, expected in cases:
        assert contract.module._reduction_shape(shape, axes, keepdims) == expected


def test_mean_wraps_result_and_forwards_all_arguments(contract):
    source = contract.Array((2, 3, 4), np.int32)
    result = contract.module.mean(source, (-1, 0), np.float32, None, True)
    assert contract.calls == [
        (source, (2, 0), True, np.dtype("float32"), np.dtype("float32"), None)
    ]
    assert (result.shape, result.dtype) == ((1, 3, 1), np.dtype("float32"))
    assert isinstance(result, contract.Array) and result.wrapped is contract.raw[0]

    result = contract.module.mean(source)
    assert contract.calls[-1] == (
        source,
        (0, 1, 2),
        False,
        np.dtype("float64"),
        np.dtype("float64"),
        None,
    )
    assert (result.shape, result.dtype) == ((), np.dtype("float64"))


def test_out_contract(contract):
    source = contract.Array((2, 3, 4), np.int32)
    out = contract.Array((3,), np.float32)
    assert contract.module.mean(source, axis=(0, 2), out=out) is out
    assert contract.calls[-1] == (
        source,
        (0, 2),
        False,
        np.dtype("float64"),
        np.dtype("float64"),
        out,
    )

    for bad_out, error in [
        (contract.Array((2,), np.float32), ValueError),
        (object(), TypeError),
        (np.empty((3,), dtype=np.float32), TypeError),
    ]:
        call_count = len(contract.calls)
        with pytest.raises(error):
            contract.module.mean(source, axis=(0, 2), out=bad_out)
        assert len(contract.calls) == call_count

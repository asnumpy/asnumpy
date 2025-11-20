import types

import numpy as np

import asnumpy as ap


def test_dtypes_module_has_int32():
    assert isinstance(ap.dtypes, types.ModuleType)
    dtype_attr = getattr(ap.dtypes, "int32", None)
    assert dtype_attr is not None
    assert dtype_attr == np.dtype("int32")


def test_dtypes_int32_can_be_used_in_array_creation():
    zeros_arr = ap.zeros((2, 2), dtype=ap.dtypes.int32)
    ones_arr = ap.ones((2, 2), dtype=ap.dtypes.int32)
    print("zeros((2,2), dtype=ap.dtypes.int32) ->", zeros_arr)
    print("ones((2,2), dtype=ap.dtypes.int32) ->", ones_arr)
    assert hasattr(zeros_arr, "dtype")
    assert zeros_arr.dtype == ap.dtypes.int32
    assert ones_arr.dtype == ap.dtypes.int32


def _run_test(func):
    try:
        func()
        print(f"[PASS] {func.__name__}")
        return True
    except AssertionError as err:
        print(f"[FAIL] {func.__name__}: {err}")
    except Exception as err:
        print(f"[ERROR] {func.__name__}: {err}")
    return False


if __name__ == "__main__":
    tests = [
        test_dtypes_module_has_int32,
        test_dtypes_int32_can_be_used_in_array_creation,
    ]
    total = len(tests)
    passed = sum(_run_test(func) for func in tests)
    print(f"\nSummary: {passed}/{total} tests passed")
    raise SystemExit(0 if passed == total else 1)

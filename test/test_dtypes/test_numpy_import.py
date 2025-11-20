import numpy as np

import asnumpy as ap


def test_debug_numpy_dtype_str():
    dtype_str = str(ap.dtypes._debug_numpy_dtype_str())
    assert "int32" in dtype_str


def test_debug_numpy_create_array():
    np_arr = ap.dtypes._debug_numpy_create_array()
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.dtype == np.dtype("int32")
    assert np_arr.shape == (3,)


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
        test_debug_numpy_dtype_str,
        test_debug_numpy_create_array,
    ]
    total = len(tests)
    passed = sum(_run_test(func) for func in tests)
    print(f"\nSummary: {passed}/{total} tests passed")
    raise SystemExit(0 if passed == total else 1)


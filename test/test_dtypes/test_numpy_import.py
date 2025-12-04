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

import logging
import numpy as np

import asnumpy as ap

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def test_debug_numpy_dtype_str():
    dtype_str = str(ap.dtypes.test_numpy_dtype_str())
    assert "int32" in dtype_str


def test_debug_numpy_create_array():
    np_arr = ap.dtypes.test_numpy_create_array()
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.dtype == np.dtype("int32")
    assert np_arr.shape == (3,)


def _run_test(func):
    try:
        func()
        logger.info(f"[PASS] {func.__name__}")
        return True
    except AssertionError as err:
        logger.error(f"[FAIL] {func.__name__}: {err}")
    except Exception as err:
        logger.error(f"[ERROR] {func.__name__}: {err}")
    return False


if __name__ == "__main__":
    tests = [
        test_debug_numpy_dtype_str,
        test_debug_numpy_create_array,
    ]
    total = len(tests)
    passed = sum(_run_test(func) for func in tests)
    logger.info(f"\nSummary: {passed}/{total} tests passed")
    raise SystemExit(0 if passed == total else 1)


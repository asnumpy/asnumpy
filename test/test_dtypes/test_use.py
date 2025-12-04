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
import types

import numpy as np

import asnumpy as ap

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def test_dtypes_module_has_int32():
    assert isinstance(ap.dtypes, types.ModuleType)
    dtype_attr = getattr(ap.dtypes, "int32", None)
    assert dtype_attr is not None
    assert dtype_attr == np.dtype("int32")


def test_dtypes_int32_can_be_used_in_array_creation():
    zeros_arr = ap.zeros((2, 2), dtype=ap.dtypes.int32)
    ones_arr = ap.ones((2, 2), dtype=ap.dtypes.int32)
    logger.info(f"zeros((2,2), dtype=ap.dtypes.int32) -> {zeros_arr}")
    logger.info(f"ones((2,2), dtype=ap.dtypes.int32) -> {ones_arr}")
    assert hasattr(zeros_arr, "dtype")
    assert zeros_arr.dtype == ap.dtypes.int32
    assert ones_arr.dtype == ap.dtypes.int32


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
        test_dtypes_module_has_int32,
        test_dtypes_int32_can_be_used_in_array_creation,
    ]
    total = len(tests)
    passed = sum(_run_test(func) for func in tests)
    logger.info(f"\nSummary: {passed}/{total} tests passed")
    raise SystemExit(0 if passed == total else 1)

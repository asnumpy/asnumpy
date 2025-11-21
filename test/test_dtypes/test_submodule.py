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

import importlib
import types

import asnumpy


def test_asnumpy_exposes_dtypes_submodule():
    """确保 asnumpy 顶层可以直接访问 dtypes 子模块。"""
    dtypes_module = getattr(asnumpy, "dtypes", None)
    assert isinstance(dtypes_module, types.ModuleType), "asnumpy.dtypes 应该是一个有效的模块对象"


def test_asnumpy_dtypes_is_importable():
    """确保可以通过常规的模块导入语法使用 asnumpy.dtypes。"""
    imported = importlib.import_module("asnumpy.dtypes")
    assert imported is asnumpy.dtypes, "asnumpy.dtypes 应该与 importlib.import_module 返回的模块一致"


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
        test_asnumpy_exposes_dtypes_submodule,
        test_asnumpy_dtypes_is_importable,
    ]
    total = len(tests)
    passed = sum(_run_test(func) for func in tests)
    print(f"\nSummary: {passed}/{total} tests passed")
    raise SystemExit(0 if passed == total else 1)
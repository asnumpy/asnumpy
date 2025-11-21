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

import numpy as np

import asnumpy as ap


def test_float8_e5m2_is_registered():
    """检查 float8_e5m2 是否已注册"""
    is_registered = ap.dtypes._check_float8_e5m2_registered()
    assert is_registered, "float8_e5m2 应该已注册"
    print(f"[PASS] float8_e5m2 注册状态: {is_registered}")


def test_float8_e5m2_type_num():
    """检查 float8_e5m2 的类型号"""
    type_num = ap.dtypes._get_float8_e5m2_type_num()
    assert type_num != -1, "float8_e5m2 类型号应该有效"
    print(f"[PASS] float8_e5m2 类型号: {type_num}")


def test_float8_e5m2_is_bound():
    """检查 float8_e5m2 是否已绑定到子模块"""
    assert hasattr(ap.dtypes, "float8_e5m2"), "float8_e5m2 应该绑定到 ap.dtypes"
    dtype_obj = ap.dtypes.float8_e5m2
    assert dtype_obj is not None, "float8_e5m2 类型对象不应为空"
    print(f"[PASS] float8_e5m2 已绑定到子模块: {dtype_obj}")


def test_float8_e5m2_can_create_dtype():
    """检查是否可以使用 float8_e5m2 创建 numpy dtype"""
    dtype = np.dtype(ap.dtypes.float8_e5m2)
    assert dtype is not None, "应该能够创建 numpy dtype"
    print(f"[PASS] 成功创建 numpy dtype: {dtype}")


def test_float8_e5m2_can_create_array():
    """检查是否可以使用 float8_e5m2 创建数组"""
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.dtypes.float8_e5m2)
    assert arr.dtype == np.dtype(ap.dtypes.float8_e5m2), "数组 dtype 应该匹配"
    print(f"[PASS] 成功创建数组: {arr}, dtype: {arr.dtype}")


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
        test_float8_e5m2_is_registered,
        test_float8_e5m2_type_num,
        test_float8_e5m2_is_bound,
        test_float8_e5m2_can_create_dtype,
        test_float8_e5m2_can_create_array,
    ]
    total = len(tests)
    passed = sum(_run_test(func) for func in tests)
    print(f"\nSummary: {passed}/{total} tests passed")
    raise SystemExit(0 if passed == total else 1)


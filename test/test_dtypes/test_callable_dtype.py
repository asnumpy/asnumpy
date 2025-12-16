#!/usr/bin/env python3
"""
测试 _CallableDtype 类的三种用法：
1. 调用创建标量: ap.bfloat16(3.14)
2. 直接作为 dtype: dtype=ap.bfloat16
3. 用于结构化数组: np.dtype([('x', ap.bfloat16)])
"""
import numpy as np
import asnumpy as ap


def test_scalar_creation():
    """测试 1: 调用创建标量"""
    print("=" * 60)
    print("测试 1: 调用创建标量")
    print("=" * 60)
    
    # 测试 bfloat16
    try:
        scalar_bf16 = ap.bfloat16(3.14)
        print(f"✓ ap.bfloat16(3.14) = {scalar_bf16}")
        print(f"isinstance(ap.bfloat16, np.dtype): {isinstance(ap.bfloat16, np.dtype)}")
        print(f"  type(scalar_bf16) = {type(scalar_bf16)}")
        assert scalar_bf16 is not None, "标量创建失败"
        print("[PASS] bfloat16 标量创建成功")
    except Exception as e:
        print(f"[FAIL] bfloat16 标量创建失败: {e}")
        raise
    
    # 测试 float8_e5m2
    try:
        scalar_f8 = ap.float8_e5m2(3.14)
        print(f"✓ ap.float8_e5m2(3.14) = {scalar_f8}")
        print(f"  type(scalar_f8) = {type(scalar_f8)}")
        assert scalar_f8 is not None, "标量创建失败"
        print("[PASS] float8_e5m2 标量创建成功")
    except Exception as e:
        print(f"[FAIL] float8_e5m2 标量创建失败: {e}")
        raise
    
    print()


def test_direct_dtype_usage():
    """测试 2: 直接作为 dtype"""
    print("=" * 60)
    print("测试 2: 直接作为 dtype")
    print("=" * 60)
    
    # 测试 bfloat16
    try:
        arr_bf16 = np.array([1.0, 2.0, 3.14], dtype=ap.bfloat16)
        print(f"✓ np.array([1.0, 2.0, 3.14], dtype=ap.bfloat16)")
        print(f"  arr.dtype = {arr_bf16.dtype}")
        print(f"  arr = {arr_bf16}")
        assert arr_bf16.dtype == ap.bfloat16 or arr_bf16.dtype == np.dtype(ap.dtypes.bfloat16), \
            f"dtype 不匹配: {arr_bf16.dtype} vs {ap.bfloat16}"
        print("[PASS] bfloat16 可以直接作为 dtype 使用")
    except Exception as e:
        print(f"[FAIL] bfloat16 作为 dtype 失败: {e}")
        raise
    
    # 测试 float8_e5m2
    try:
        arr_f8 = np.array([1.0, 2.0, 3.14], dtype=ap.float8_e5m2)
        print(f"✓ np.array([1.0, 2.0, 3.14], dtype=ap.float8_e5m2)")
        print(f"  arr.dtype = {arr_f8.dtype}")
        print(f"  arr = {arr_f8}")
        assert arr_f8.dtype == ap.float8_e5m2 or arr_f8.dtype == np.dtype(ap.dtypes.float8_e5m2), \
            f"dtype 不匹配: {arr_f8.dtype} vs {ap.float8_e5m2}"
        print("[PASS] float8_e5m2 可以直接作为 dtype 使用")
    except Exception as e:
        print(f"[FAIL] float8_e5m2 作为 dtype 失败: {e}")
        raise
    
    print()


def test_structured_array():
    """测试 3: 用于结构化数组"""
    print("=" * 60)
    print("测试 3: 用于结构化数组")
    print("=" * 60)
    
    # 测试 bfloat16
    try:
        dtype_struct = np.dtype([('x', ap.bfloat16), ('y', ap.bfloat16)])
        print(f"✓ np.dtype([('x', ap.bfloat16), ('y', ap.bfloat16)])")
        print(f"  dtype_struct = {dtype_struct}")
        print(f"  dtype_struct['x'] = {dtype_struct['x']}")
        print(f"  dtype_struct['y'] = {dtype_struct['y']}")
        
        # 创建结构化数组
        arr_struct = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype_struct)
        print(f"  arr_struct = {arr_struct}")
        print(f"  arr_struct['x'] = {arr_struct['x']}")
        print(f"  arr_struct['y'] = {arr_struct['y']}")
        
        assert dtype_struct['x'] == np.dtype(ap.dtypes.bfloat16) or dtype_struct['x'] == ap.bfloat16, \
            f"结构化数组中的 dtype 不匹配"
        print("[PASS] bfloat16 可以用于结构化数组")
    except Exception as e:
        print(f"[FAIL] bfloat16 用于结构化数组失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 测试 float8_e5m2
    try:
        dtype_struct = np.dtype([('x', ap.float8_e5m2), ('y', ap.float8_e5m2)])
        print(f"✓ np.dtype([('x', ap.float8_e5m2), ('y', ap.float8_e5m2)])")
        print(f"  dtype_struct = {dtype_struct}")
        print(f"  dtype_struct['x'] = {dtype_struct['x']}")
        print(f"  dtype_struct['y'] = {dtype_struct['y']}")
        
        # 创建结构化数组
        arr_struct = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype_struct)
        print(f"  arr_struct = {arr_struct}")
        print(f"  arr_struct['x'] = {arr_struct['x']}")
        print(f"  arr_struct['y'] = {arr_struct['y']}")
        
        assert dtype_struct['x'] == np.dtype(ap.dtypes.float8_e5m2) or dtype_struct['x'] == ap.float8_e5m2, \
            f"结构化数组中的 dtype 不匹配"
        print("[PASS] float8_e5m2 可以用于结构化数组")
    except Exception as e:
        print(f"[FAIL] float8_e5m2 用于结构化数组失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print()


def test_dtype_properties():
    """测试 dtype 属性访问"""
    print("=" * 60)
    print("测试 4: dtype 属性访问")
    print("=" * 60)
    
    try:
        print(f"ap.bfloat16.name = {ap.bfloat16.name}")
        print(f"ap.bfloat16.itemsize = {ap.bfloat16.itemsize}")
        print(f"ap.bfloat16.kind = {ap.bfloat16.kind}")
        print(f"ap.bfloat16.char = {ap.bfloat16.char}")
        print(f"ap.bfloat16.type = {ap.bfloat16.type}")
        
        assert hasattr(ap.bfloat16, 'name'), "缺少 name 属性"
        assert hasattr(ap.bfloat16, 'itemsize'), "缺少 itemsize 属性"
        assert hasattr(ap.bfloat16, 'kind'), "缺少 kind 属性"
        print("[PASS] dtype 属性访问正常")
    except Exception as e:
        print(f"[FAIL] dtype 属性访问失败: {e}")
        raise
    
    print()


def test_dtype_comparison():
    """测试 dtype 比较"""
    print("=" * 60)
    print("测试 5: dtype 比较")
    print("=" * 60)
    
    try:
        dtype_from_wrapper = ap.bfloat16
        dtype_from_numpy = np.dtype(ap.dtypes.bfloat16)
        
        print(f"ap.bfloat16 = {ap.bfloat16}")
        print(f"np.dtype(ap.dtypes.bfloat16) = {dtype_from_numpy}")
        print(f"ap.bfloat16 == np.dtype(ap.dtypes.bfloat16): {dtype_from_wrapper == dtype_from_numpy}")
        
        # 创建数组测试
        arr1 = np.array([1.0], dtype=ap.bfloat16)
        arr2 = np.array([1.0], dtype=np.dtype(ap.dtypes.bfloat16))
        print(f"arr1.dtype == arr2.dtype: {arr1.dtype == arr2.dtype}")
        
        assert arr1.dtype == arr2.dtype, "数组 dtype 应该相等"
        print("[PASS] dtype 比较正常")
    except Exception as e:
        print(f"[FAIL] dtype 比较失败: {e}")
        raise
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试 _CallableDtype 类的三种用法")
    print("=" * 60 + "\n")
    
    try:
        test_scalar_creation()
        test_direct_dtype_usage()
        test_structured_array()
        test_dtype_properties()
        test_dtype_comparison()
        
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)


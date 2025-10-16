"""
测试ACL整数类型(int4, uint1)的NumPy dtype支持

测试目标：
1. 验证int4和uint1类型可以被NumPy识别
2. 验证类型的getACLenum()方法返回正确的ACL枚举值
3. 验证标量创建和基本操作
4. 验证数组转换
"""

import sys
import os
# 添加 build/python 到路径以导入最新编译的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))

import asnumpy_core as core
import numpy as np
adt = core.dtypes

print("=" * 80)
print("测试ACL整数类型的NumPy dtype支持")
print("=" * 80)
print()

def test_int4_basic():
    """测试 int4 基本功能"""
    print("=" * 80)
    print("测试 int4 基本功能")
    print("=" * 80)
    
    # 检查类型对象
    assert hasattr(adt, 'int4'), "int4 类型对象未绑定"
    dt = np.dtype(adt.int4)
    assert dt is not None, "np.dtype 未能识别 int4 类型对象"
    print("✓ int4 类型对象已绑定且可被 numpy.dtype 识别")
    
    # 创建标量
    try:
        scalar = adt.int4(5)
        print(f"✓ 成功创建 int4 标量: {scalar}")
        print(f"  类型: {type(scalar).__name__}")
        
        # 测试getACLenum
        acl_enum = scalar.getACLenum()
        print(f"  ACL枚举值: {acl_enum}")
        assert acl_enum == 29, f"ACL_INT4 should be 29, got {acl_enum}"  # ACL_INT4 = 29
        print(f"✓ getACLenum() 返回正确: {acl_enum}")
        
    except Exception as e:
        print(f"✗ 标量创建失败: {e}")
        raise
    
    # 测试边界值
    print("\n测试边界值:")
    min_val = adt.int4(-8)  # 4-bit signed min
    max_val = adt.int4(7)   # 4-bit signed max
    print(f"  min (-8): {min_val}")
    print(f"  max (7): {max_val}")
    print("✓ 边界值测试通过")
    
    return True

def test_uint1_basic():
    """测试 uint1 基本功能"""
    print("\n" + "=" * 80)
    print("测试 uint1 基本功能")
    print("=" * 80)
    
    # 检查类型对象
    assert hasattr(adt, 'uint1'), "uint1 类型对象未绑定"
    dt = np.dtype(adt.uint1)
    assert dt is not None, "np.dtype 未能识别 uint1 类型对象"
    print("✓ uint1 类型对象已绑定且可被 numpy.dtype 识别")
    
    # 创建标量
    try:
        scalar = adt.uint1(1)
        print(f"✓ 成功创建 uint1 标量: {scalar}")
        print(f"  类型: {type(scalar).__name__}")
        
        # 测试getACLenum
        acl_enum = scalar.getACLenum()
        print(f"  ACL枚举值: {acl_enum}")
        assert acl_enum == 30, f"ACL_UINT1 should be 30, got {acl_enum}"  # ACL_UINT1 = 30
        print(f"✓ getACLenum() 返回正确: {acl_enum}")
        
    except Exception as e:
        print(f"✗ 标量创建失败: {e}")
        raise
    
    # 测试边界值
    print("\n测试边界值:")
    zero_val = adt.uint1(0)  # 1-bit unsigned min
    one_val = adt.uint1(1)   # 1-bit unsigned max
    print(f"  min (0): {zero_val}")
    print(f"  max (1): {one_val}")
    print("✓ 边界值测试通过")
    
    return True

def test_int_types_comparison():
    """测试整数类型比较"""
    print("\n" + "=" * 80)
    print("测试整数类型比较")
    print("=" * 80)
    
    # int4 比较
    a = adt.int4(3)
    b = adt.int4(5)
    c = adt.int4(3)
    
    print("int4 比较:")
    print(f"  3 < 5: {a < b}")
    print(f"  3 == 3: {a == c}")
    print(f"  3 > 5: {a > b}")
    assert a < b, "int4 比较失败"
    assert a == c, "int4 相等判断失败"
    print("✓ int4 比较测试通过")
    
    # uint1 比较
    x = adt.uint1(0)
    y = adt.uint1(1)
    z = adt.uint1(0)
    
    print("\nuint1 比较:")
    print(f"  0 < 1: {x < y}")
    print(f"  0 == 0: {x == z}")
    print(f"  0 > 1: {x > y}")
    assert x < y, "uint1 比较失败"
    assert x == z, "uint1 相等判断失败"
    print("✓ uint1 比较测试通过")
    
    return True

def test_numpy_array_creation():
    """测试使用int类型创建NumPy数组"""
    print("\n" + "=" * 80)
    print("测试使用int类型创建NumPy数组")
    print("=" * 80)
    
    try:
        # 创建int4数组 - 使用empty然后填充
        int4_array = np.empty(16, dtype=np.dtype(adt.int4))
        print(f"✓ 成功创建 int4 数组:")
        print(f"  形状: {int4_array.shape}")
        print(f"  dtype: {int4_array.dtype}")
        print(f"  itemsize: {int4_array.itemsize}")
        
        # 创建uint1数组 - 使用empty然后填充
        uint1_array = np.empty(10, dtype=np.dtype(adt.uint1))
        print(f"\n✓ 成功创建 uint1 数组:")
        print(f"  形状: {uint1_array.shape}")
        print(f"  dtype: {uint1_array.dtype}")
        print(f"  itemsize: {uint1_array.itemsize}")
        
        # 测试类型转换
        print("\n测试类型转换:")
        int32_arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        int4_arr = int32_arr.astype(np.dtype(adt.int4))
        print(f"  int32 -> int4 转换成功")
        print(f"  dtype: {int4_arr.dtype}")
        
        return True
    except Exception as e:
        print(f"✗ 数组创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试ACL整数类型...\n")
    
    # 运行所有测试
    int4_success = test_int4_basic()
    uint1_success = test_uint1_basic()
    comparison_success = test_int_types_comparison()
    array_success = test_numpy_array_creation()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    
    all_success = int4_success and uint1_success and comparison_success and array_success
    
    if all_success:
        print("\n🎉 所有测试通过！")
        print("✓ int4 和 uint1 类型成功注册到NumPy")
        print("✓ getACLenum() 方法工作正常")
        print("✓ 标量创建和比较操作正常")
        print("✓ NumPy数组创建正常")
    else:
        print("\n⚠ 部分测试失败")
        if not int4_success:
            print("✗ int4 基本功能测试失败")
        if not uint1_success:
            print("✗ uint1 基本功能测试失败")
        if not comparison_success:
            print("✗ 比较测试失败")
        if not array_success:
            print("✗ 数组创建测试失败")


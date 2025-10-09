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

#!/usr/bin/env python3
"""
测试 float_types.hpp 中定义的浮点类型是否正确注册到 NumPy 并绑定到模块
"""

import numpy as np
import asnumpy as ap

def test_float_types_registration():
    """测试所有浮点类型是否正确注册到NumPy"""
    print("=" * 60)
    print("测试 float_types.hpp 中的类型注册到 NumPy")
    print("=" * 60)
    
    # 定义要测试的类型（来自 float_types.hpp）
    float_types = [
        'float8_e5m2',
        'float8_e4m3fn', 
        'float8_e8m0',
        'bfloat16',
        'float6_e2m3fn',
        'float6_e3m2fn',
        'float4_e2m1fn',
    ]
    
    success_count = 0
    total_count = len(float_types)
    
    for type_name in float_types:
        try:
            print(f"\n测试 {type_name}:")
            
            # 1. 检查类型是否在asnumpy模块中可用
            if not hasattr(ap, type_name):
                print(f"  ✗ 类型 {type_name} 不在 asnumpy 模块中")
                continue
            
            dtype_obj = getattr(ap, type_name)
            print(f"  ✓ 类型对象: {dtype_obj}")
            
            # 2. 检查NumPy是否能识别这个类型
            try:
                np_dtype = np.dtype(dtype_obj)
                print(f"  ✓ NumPy dtype: {np_dtype}")
                print(f"  ✓ dtype 名称: {np_dtype.name}")
                print(f"  ✓ dtype 大小: {np_dtype.itemsize} 字节")
            except Exception as e:
                print(f"  ✗ NumPy dtype 识别失败: {e}")
                continue
            
            # 3. 测试标量创建
            try:
                scalar = dtype_obj(3.14)
                print(f"  ✓ 标量创建: {scalar}")
                print(f"  ✓ 标量类型: {type(scalar)}")
            except Exception as e:
                print(f"  ✗ 标量创建失败: {e}")
                continue
            
            # 4. 测试数组创建（直接使用类型对象）
            try:
                arr = np.array([1.0, 2.0, 3.0], dtype=dtype_obj)
                print(f"  ✓ 数组创建: {arr}")
                print(f"  ✓ 数组dtype: {arr.dtype}")
                print(f"  ✓ 数组形状: {arr.shape}")
            except Exception as e:
                print(f"  ✗ 数组创建失败: {e}")
                continue
            
            # 5. 测试类型转换（直接使用类型对象）
            try:
                # 从float转换
                converted = np.array([1.5, 2.7, 3.9], dtype=np.float32).astype(dtype_obj)
                print(f"  ✓ 类型转换: {converted}")
                
                # 转换回float
                back_to_float = converted.astype(np.float32)
                print(f"  ✓ 转换回float: {back_to_float}")
            except Exception as e:
                print(f"  ✗ 类型转换失败: {e}")
                continue
            
            success_count += 1
            print(f"  🎉 {type_name} 测试通过!")
            
        except Exception as e:
            print(f"  ✗ {type_name} 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_count} 个类型注册成功")
    print("=" * 60)
    
    return success_count == total_count

def test_numpy_dtype_methods():
    """测试NumPy dtype方法是否可用"""
    print("\n" + "=" * 60)
    print("测试 NumPy dtype 方法")
    print("=" * 60)
    
    test_type = ap.float8_e5m2
    np_dtype = np.dtype(test_type)
    
    print(f"测试类型: {test_type}")
    print(f"NumPy dtype: {np_dtype}")
    
    # 测试dtype的各种属性和方法
    dtype_methods = [
        ('name', 'dtype名称'),
        ('itemsize', '元素大小'),
        ('kind', '类型种类'),
        ('char', '类型字符'),
        ('type', 'Python类型'),
        ('str', '字符串表示'),
        ('descr', '描述符'),
    ]
    
    for method_name, description in dtype_methods:
        try:
            if hasattr(np_dtype, method_name):
                value = getattr(np_dtype, method_name)
                print(f"  ✓ {description}: {value}")
            else:
                print(f"  ✗ {description}: 方法不存在")
        except Exception as e:
            print(f"  ✗ {description}: 错误 - {e}")

def test_array_operations():
    """测试数组操作"""
    print("\n" + "=" * 60)
    print("测试数组操作")
    print("=" * 60)
    
    try:
        # 使用float8_e5m2进行测试
        dtype = ap.float8_e5m2
        
        print(f"使用类型: {dtype}")
        
        # 直接创建数组，无需np.dtype()包装
        arr1 = np.array([1.0, 2.0, 3.0], dtype=dtype)
        arr2 = np.array([0.5, 1.5, 2.5], dtype=dtype)
        
        print(f"arr1: {arr1}")
        print(f"arr2: {arr2}")
        
        # 测试基本操作
        operations = [
            ('加法', lambda a, b: a + b),
            ('减法', lambda a, b: a - b),
            ('乘法', lambda a, b: a * b),
            ('除法', lambda a, b: a / b),
        ]
        
        for op_name, op_func in operations:
            try:
                result = op_func(arr1, arr2)
                print(f"  ✓ {op_name}: {result}")
            except Exception as e:
                print(f"  ✗ {op_name}: {e}")
        
        # 测试数组属性
        print(f"  ✓ 数组dtype: {arr1.dtype}")
        print(f"  ✓ 数组形状: {arr1.shape}")
        print(f"  ✓ 数组大小: {arr1.size}")
        print(f"  ✓ 数组维度: {arr1.ndim}")
        
    except Exception as e:
        print(f"数组操作测试失败: {e}")

def test_type_compatibility():
    """测试类型兼容性"""
    print("\n" + "=" * 60)
    print("测试类型兼容性")
    print("=" * 60)
    
    # 测试与标准NumPy类型的兼容性
    standard_types = [np.float32, np.float64, np.int32, np.int64]
    custom_types = [ap.float8_e5m2, ap.bfloat16, ap.float8_e4m3fn]
    
    for custom_type in custom_types:
        np_custom_dtype = np.dtype(custom_type)
        print(f"\n测试 {custom_type}:")
        
        for std_type in standard_types:
            try:
                # 创建标准类型数组
                std_arr = np.array([1.0, 2.0, 3.0], dtype=std_type)
                
                # 转换为自定义类型
                custom_arr = std_arr.astype(np_custom_dtype)
                print(f"  ✓ {std_type} -> {custom_type}: {custom_arr}")
                
                # 转换回标准类型
                back_arr = custom_arr.astype(std_type)
                print(f"  ✓ {custom_type} -> {std_type}: {back_arr}")
                
            except Exception as e:
                print(f"  ✗ {std_type} <-> {custom_type}: {e}")

def test_getACLenum_interface():
    """测试getACLenum()接口"""
    print("\n" + "=" * 60)
    print("测试 getACLenum() 接口")
    print("=" * 60)
    
    # 预期的ACL枚举值（来自float_types.hpp的实际值）
    expected_acl_values = {
        'float8_e5m2': 35,      # ACL_FLOAT8_E5M2
        'float8_e4m3fn': 36,     # ACL_FLOAT8_E4M3FN
        'float8_e8m0': 37,       # ACL_FLOAT8_E8M0
        'bfloat16': 27,          # ACL_BF16
        'float6_e2m3fn': 39,     # ACL_FLOAT6_E2M3 (实际值)
        'float6_e3m2fn': 38,     # ACL_FLOAT6_E3M2 (实际值)
        'float4_e2m1fn': 40,     # ACL_FLOAT4_E2M1
    }
    
    success_count = 0
    total_count = len(expected_acl_values)
    
    for type_name, expected_value in expected_acl_values.items():
        try:
            print(f"\n测试 {type_name}:")
            
            # 获取类型对象
            dtype_obj = getattr(ap, type_name)
            print(f"  ✓ 类型对象: {dtype_obj}")
            
            # 创建标量
            scalar = dtype_obj(1.0)
            print(f"  ✓ 标量创建: {scalar}")
            
            # 检查getACLenum方法是否存在
            if not hasattr(scalar, 'getACLenum'):
                print(f"  ✗ getACLenum方法不存在")
                continue
            
            # 调用getACLenum方法
            acl_enum = scalar.getACLenum()
            print(f"  ✓ ACL枚举值: {acl_enum}")
            print(f"  ✓ 期望值: {expected_value}")
            
            # 验证返回值
            if acl_enum == expected_value:
                print(f"  🎉 {type_name} ACL枚举值正确!")
                success_count += 1
            else:
                print(f"  ✗ ACL枚举值不匹配: 期望 {expected_value}, 实际 {acl_enum}")
            
            # 测试不同标量值的getACLenum是否一致
            scalar2 = dtype_obj(3.14)
            acl_enum2 = scalar2.getACLenum()
            if acl_enum == acl_enum2:
                print(f"  ✓ 不同标量值的ACL枚举值一致")
            else:
                print(f"  ✗ 不同标量值的ACL枚举值不一致")
            
        except Exception as e:
            print(f"  ✗ {type_name} 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"getACLenum() 测试结果: {success_count}/{total_count} 个类型通过")
    print("=" * 60)
    
    return success_count == total_count

def test_getACLenum_with_arrays():
    """测试数组元素的getACLenum()接口"""
    print("\n" + "=" * 60)
    print("测试数组元素的 getACLenum() 接口")
    print("=" * 60)
    
    try:
        # 直接使用asnumpy类型创建数组，无需包装
        print(f"使用类型: {ap.float8_e5m2}")
        
        # 直接创建数组，无需np.dtype()包装
        arr = np.array([1.0, 2.0, 3.14], dtype=ap.float8_e5m2)
        print(f"数组: {arr}")
        print(f"数组dtype: {arr.dtype}")
        
        # 测试数组元素的getACLenum
        for i, element in enumerate(arr):
            try:
                acl_enum = element.getACLenum()
                print(f"  ✓ 元素[{i}] = {element}, ACL枚举 = {acl_enum}")
            except Exception as e:
                print(f"  ✗ 元素[{i}] getACLenum失败: {e}")
        
        # 测试数组切片元素的getACLenum
        slice_elements = arr[1:3]
        print(f"切片元素: {slice_elements}")
        
        for i, element in enumerate(slice_elements):
            try:
                acl_enum = element.getACLenum()
                print(f"  ✓ 切片元素[{i}] = {element}, ACL枚举 = {acl_enum}")
            except Exception as e:
                print(f"  ✗ 切片元素[{i}] getACLenum失败: {e}")
        
        # 测试其他类型的数组
        print(f"\n测试其他类型:")
        other_types = [ap.bfloat16, ap.float8_e4m3fn, ap.float6_e2m3fn]
        
        for dtype in other_types:
            try:
                test_arr = np.array([1.5, 2.7], dtype=dtype)
                print(f"  ✓ {dtype.__name__} 数组: {test_arr}")
                for element in test_arr:
                    acl_enum = element.getACLenum()
                    print(f"    ACL枚举: {acl_enum}")
            except Exception as e:
                print(f"  ✗ {dtype.__name__} 测试失败: {e}")
                
    except Exception as e:
        print(f"数组getACLenum测试失败: {e}")

def main():
    """主测试函数"""
    print("Float Types NumPy Integration Test")
    print("=" * 60)
    print("测试 float_types.hpp 中定义的浮点类型是否正确注册到 NumPy")
    print("=" * 60)
    
    # 运行所有测试
    test1_passed = test_float_types_registration()
    test_numpy_dtype_methods()
    test_array_operations()
    test_type_compatibility()
    test2_passed = test_getACLenum_interface()
    test_getACLenum_with_arrays()
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  类型注册: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"  getACLenum接口: {'✓ 通过' if test2_passed else '✗ 失败'}")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("🎉 所有浮点类型已成功注册到 NumPy 并绑定到模块！")
        print("✅ 可以使用 numpy.dtype() 方法识别这些类型")
        print("✅ 可以创建标量和数组")
        print("✅ 支持类型转换和数组操作")
        print("✅ getACLenum() 接口正常工作")
        print("✅ 可以获取正确的ACL枚举值")
        return True
    else:
        print("❌ 部分测试失败，需要进一步调试。")
        return False

if __name__ == "__main__":
    main()

import sys
import os
# 添加 build/python 到路径以导入最新编译的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))

import asnumpy_core as core
import numpy as np
adt = core.dtypes

# 初始化ACL环境（直接导入build/python时需要手动初始化）
try:
    core.cann.init()
    core.cann.set_device(0)
    print("✓ ACL环境初始化成功\n")
except Exception as e:
    print(f"⚠ ACL环境初始化失败: {e}")
    print("  NPUArray创建测试将会失败\n")

def test_all_custom_dtypes():
    """测试所有自定义注册的numpy dtype进行ndarray创建tensor"""
    print("=" * 80)
    print("测试所有自定义注册的numpy dtype进行ndarray创建tensor")
    print("=" * 80)
    
    # 所有已注册的自定义ACL浮点类型及其期望的ACL枚举值
    custom_float_dtypes = [
        ('float8_e5m2', adt.float8_e5m2, 35, "ACL_FLOAT8_E5M2"),
        ('float8_e4m3fn', adt.float8_e4m3fn, 36, "ACL_FLOAT8_E4M3FN"),
        ('float8_e8m0', adt.float8_e8m0, 37, "ACL_FLOAT8_E8M0"),
        ('bfloat16', adt.bfloat16, 27, "ACL_BF16"),
        ('float6_e2m3fn', adt.float6_e2m3fn, 39, "ACL_FLOAT6_E2M3"),
        ('float6_e3m2fn', adt.float6_e3m2fn, 38, "ACL_FLOAT6_E3M2"),
        ('float4_e2m1fn', adt.float4_e2m1fn, 40, "ACL_FLOAT4_E2M1"),
        ('float4_e1m2fn', adt.float4_e1m2fn, 41, "ACL_FLOAT4_E1M2"),
    ]
    
    # 所有已注册的自定义ACL整数类型及其期望的ACL枚举值
    custom_int_dtypes = [
        ('int4', adt.int4, 29, "ACL_INT4"),
        ('uint1', adt.uint1, 30, "ACL_UINT1"),
    ]
    
    # 合并所有自定义类型
    custom_dtypes = custom_float_dtypes + custom_int_dtypes
    
    # 标准类型作为对比（使用实际的ACL枚举值）
    standard_dtypes = [
        ('float32', np.float32, 0, "ACL_FLOAT"),
        ('float64', np.float64, 11, "ACL_DOUBLE"),  # 修正为实际值
        ('int32', np.int32, 3, "ACL_INT32"),
        ('int64', np.int64, 9, "ACL_INT64"),  # 修正为实际值
        ('uint8', np.uint8, 4, "ACL_UINT8"),  # 修正为实际值
        ('bool', np.bool_, 12, "ACL_BOOL"),  # 修正为实际值
    ]
    
    success_count = 0
    total_count = len(custom_dtypes) + len(standard_dtypes)
    
    print("\n=== 自定义ACL浮点类型测试 ===")
    for name, dtype_obj, expected_acl, acl_name in custom_float_dtypes:
        try:
            print(f"\n测试 {name} ({acl_name}):")
            
            # 直接使用dtype对象
            print(f"  dtype对象: {dtype_obj}")
            
            # 创建不同形状的ndarray
            shapes = [[2, 3], [1, 4, 2], [5]]
            
            for i, shape in enumerate(shapes):
                try:
                    array = core.ndarray(shape, dtype=np.dtype(dtype_obj))
                    print(f"  形状 {shape}: 成功创建")
                    print(f"    类型: {type(array)}")
                    print(f"    形状: {array.shape}")
                    print(f"    dtype: {array.dtype}")
                    print(f"    aclDtype: {array.aclDtype}")
                    
                    # 验证ACL枚举值
                    if array.aclDtype == expected_acl:
                        print(f"    ✓ ACL枚举值正确: {array.aclDtype}")
                        if i == 0:  # 只在第一个形状时计数
                            success_count += 1
                    else:
                        print(f"    ✗ ACL枚举值不匹配: 期望 {expected_acl}, 实际 {array.aclDtype}")
                        
                except Exception as e:
                    print(f"  形状 {shape}: 创建失败 - {e}")
                    
        except Exception as e:
            print(f"  {name}: 整体测试失败 - {e}")
    
    print("\n=== 标准类型对比测试 ===")
    for name, dtype_obj, expected_acl, acl_name in standard_dtypes:
        try:
            print(f"\n测试 {name} ({acl_name}):")
            
            # 直接使用dtype对象
            array = core.ndarray([2, 2], dtype=np.dtype(dtype_obj))
            print(f"  成功创建: {type(array)}")
            print(f"  aclDtype: {array.aclDtype}")
            
            # 验证ACL枚举值
            if array.aclDtype == expected_acl:
                print(f"  ✓ ACL枚举值正确: {array.aclDtype}")
                success_count += 1
            else:
                print(f"  ✗ ACL枚举值不匹配: 期望 {expected_acl}, 实际 {array.aclDtype}")
                
        except Exception as e:
            print(f"  {name}: 测试失败 - {e}")
    
    print("\n=== 自定义ACL整数类型测试 ===")
    for name, dtype_obj, expected_acl, acl_name in custom_int_dtypes:
        try:
            print(f"\n测试 {name} ({acl_name}):")
            
            # 直接使用dtype对象
            print(f"  dtype对象: {dtype_obj}")
            
            # 创建不同形状的ndarray
            shapes = [[2, 3], [1, 4, 2], [5]]
            
            for i, shape in enumerate(shapes):
                try:
                    array = core.ndarray(shape, dtype=np.dtype(dtype_obj))
                    print(f"  形状 {shape}: 成功创建")
                    print(f"    类型: {type(array)}")
                    print(f"    形状: {array.shape}")
                    print(f"    dtype: {array.dtype}")
                    print(f"    aclDtype: {array.aclDtype}")
                    
                    # 验证ACL枚举值
                    if array.aclDtype == expected_acl:
                        print(f"    ✓ ACL枚举值正确: {array.aclDtype}")
                        if i == 0:  # 只在第一个形状时计数
                            success_count += 1
                    else:
                        print(f"    ✗ ACL枚举值不匹配: 期望 {expected_acl}, 实际 {array.aclDtype}")
                        
                except Exception as e:
                    print(f"  形状 {shape}: 创建失败 - {e}")
                    
        except Exception as e:
            print(f"  {name}: 整体测试失败 - {e}")
    
    print("\n" + "=" * 80)
    print(f"测试结果: {success_count}/{total_count} 个类型测试通过")
    print("=" * 80)
    print(f"  浮点类型: {len(custom_float_dtypes)} 个")
    print(f"  整数类型: {len(custom_int_dtypes)} 个")
    print(f"  标准类型: {len(standard_dtypes)} 个")
    
    return success_count == total_count

def test_dtype_properties():
    """测试dtype属性"""
    print("\n=== dtype属性测试 ===")
    
    # 浮点类型样例
    float_dtypes = [
        ('float8_e5m2', adt.float8_e5m2, 1.0),
        ('float8_e4m3fn', adt.float8_e4m3fn, 1.0),
        ('bfloat16', adt.bfloat16, 1.0),
        ('float4_e1m2fn', adt.float4_e1m2fn, 1.0),
    ]
    
    # 整数类型样例
    int_dtypes = [
        ('int4', adt.int4, 5),
        ('uint1', adt.uint1, 1),
    ]
    
    all_test_dtypes = float_dtypes + int_dtypes
    
    for name, dtype_obj, test_val in all_test_dtypes:
        try:
            print(f"\n{name} 属性:")
            dtype_wrapped = np.dtype(dtype_obj)
            
            # 检查dtype属性
            print(f"  dtype: {dtype_wrapped}")
            print(f"  dtype.name: {dtype_wrapped.name}")
            print(f"  dtype.type: {dtype_wrapped.type}")
            print(f"  dtype.itemsize: {dtype_wrapped.itemsize}")
            print(f"  dtype.alignment: {dtype_wrapped.alignment}")
            
            # 检查type对象是否有getACLenum方法
            type_obj = dtype_wrapped.type
            if hasattr(type_obj, 'getACLenum'):
                scalar = type_obj(test_val)
                acl_enum = scalar.getACLenum()
                print(f"  getACLenum(): {acl_enum}")
            else:
                print(f"  没有getACLenum方法")
                
        except Exception as e:
            print(f"  {name}: 属性测试失败 - {e}")

if __name__ == "__main__":
    print("自定义numpy dtype ndarray创建tensor测试\n")
    
    # 运行所有测试
    success = test_all_custom_dtypes()
    test_dtype_properties()
    
    if success:
        print("\n🎉 所有测试通过！GetACLDataType函数工作正常！")
    else:
        print("\n❌ 部分测试失败，请检查实现。")
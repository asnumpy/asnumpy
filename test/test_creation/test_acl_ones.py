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
AsNumpy 直接 ACL 路径验证测试
验证 AsNumpy 支持 NumPy 不支持的数据类型
"""

import asnumpy as ap
import numpy as np
import struct


def analyze_memory_representation(array, dtype_name):
    """分析数组的内存表示"""
    print(f"  内存分析 - {dtype_name}:")
    try:
        # 获取原始字节数据
        np_array = array.to_numpy()
        print(f"    NumPy数组类型: {np_array.dtype}")
        print(f"    NumPy数组值: \n{np_array}")
        
        # 分析字节表示
        if np_array.size > 0:
            first_element_bytes = np_array.tobytes()[:np_array.itemsize]
            print(f"    第一个元素的字节表示: {[b for b in first_element_bytes]}")
            
            # 尝试不同的解释方式
            if len(first_element_bytes) == 4:  # 4字节
                as_int32 = struct.unpack('i', first_element_bytes)[0]
                as_uint32 = struct.unpack('I', first_element_bytes)[0]
                as_float32 = struct.unpack('f', first_element_bytes)[0]
                print(f"    解释为int32: {as_int32}")
                print(f"    解释为uint32: {as_uint32}")
                print(f"    解释为float32: {as_float32}")
            elif len(first_element_bytes) == 8:  # 8字节
                as_int64 = struct.unpack('q', first_element_bytes)[0]
                as_uint64 = struct.unpack('Q', first_element_bytes)[0]
                as_float64 = struct.unpack('d', first_element_bytes)[0]
                print(f"    解释为int64: {as_int64}")
                print(f"    解释为uint64: {as_uint64}")
                print(f"    解释为float64: {as_float64}")
            elif len(first_element_bytes) == 2:  # 2字节
                as_int16 = struct.unpack('h', first_element_bytes)[0]
                as_uint16 = struct.unpack('H', first_element_bytes)[0]
                print(f"    解释为int16: {as_int16}")
                print(f"    解释为uint16: {as_uint16}")
            elif len(first_element_bytes) == 1:  # 1字节
                as_int8 = struct.unpack('b', first_element_bytes)[0]
                as_uint8 = struct.unpack('B', first_element_bytes)[0]
                print(f"    解释为int8: {as_int8}")
                print(f"    解释为uint8: {as_uint8}")
    except Exception as e:
        print(f"    内存分析失败: {e}")


def print_array_info(array, dtype_name, description=""):
    """打印数组信息的辅助函数"""
    try:
        print(f"   ✓ {dtype_name:12} = {getattr(ap, dtype_name):2} -> 成功")
        print(f"      形状: {array.shape}")
        print(f"      数据类型: {array.dtype}")
        if hasattr(array, 'aclDtype'):
            print(f"      ACL数据类型: {array.aclDtype}")
        
        # 分析内存表示
        analyze_memory_representation(array, dtype_name)
        
        # 验证是否为全1数组
        try:
            np_array = array.to_numpy()
            if np_array.size > 0:
                first_val = np_array.flat[0]
                all_same = np.allclose(np_array, first_val)
                print(f"      所有元素相同: {all_same}")
                print(f"      第一个元素值: {first_val}")
                print(f"      数组和: {np.sum(np_array)}")
                print(f"      期望和: {np_array.size}")
        except Exception as e:
            print(f"      验证失败: {e}")
        
        print()
    except Exception as e:
        print(f"      打印失败: {e}")


def test_all_acl_types():
    """测试所有 ACL 数据类型"""
    print("所有 ACL 数据类型测试:")
    print("=" * 50)
    
    # 直接使用 ap.xxx 格式的 ACL 类型列表
    acl_type_tests = [
        ('dt_undefined', ap.dt_undefined),
        ('float32', ap.float32),
        ('float16', ap.float16),
        ('int8', ap.int8),
        ('int32', ap.int32),
        ('uint8', ap.uint8),
        ('int16', ap.int16),
        ('uint16', ap.uint16),
        ('uint32', ap.uint32),
        ('int64', ap.int64),
        ('uint64', ap.uint64),
        ('float64', ap.float64),
        ('bool', ap.bool),
        ('string', ap.string),
        ('complex64', ap.complex64),
        ('complex128', ap.complex128),
        ('bf16', ap.bf16),
        ('int4', ap.int4),
        ('uint1', ap.uint1),
        ('complex32', ap.complex32),
        ('hifloat8', ap.hifloat8),
        ('float8_e5m2', ap.float8_e5m2),
        ('float8_e4m3fn', ap.float8_e4m3fn),
        ('float8_e8m0', ap.float8_e8m0),
        ('float6_e3m2', ap.float6_e3m2),
        ('float6_e2m3', ap.float6_e2m3),
        ('float4_e2m1', ap.float4_e2m1),
        ('float4_e1m2', ap.float4_e1m2)
    ]
    
    success_count = 0
    total_count = len(acl_type_tests)
    
    for dtype_name, dtype_value in acl_type_tests:
        try:
            # 直接使用 ap.xxx 格式传入 ones 函数
            array = ap.ones((2, 2), dtype=dtype_value)
            print_array_info(array, dtype_name, "ACL类型")
            success_count += 1
        except Exception as e:
            print(f"   ✗ {dtype_name:12} = {dtype_value:2} -> 失败: {e}")
    
    print("=" * 50)
    print(f"测试结果: {success_count}/{total_count} 个类型支持成功")
    print("=" * 50)


def main():
    """主测试函数"""
    print("AsNumpy 直接 ACL 路径验证测试")
    print("=" * 50)
    
    # 直接测试所有 ACL 类型
    test_all_acl_types()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == "__main__":
    main() 
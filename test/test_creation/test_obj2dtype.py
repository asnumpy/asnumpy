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
测试 ObjToDtype 函数的安全性和错误处理
专注于潜在安全漏洞和错误处理完整性
输出结果到文件
"""

import sys
import os
import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 使用固定的输出文件
output_file = "security_test_results.txt"

def log(message):
    """将消息同时输出到控制台和文件"""
    print(message)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

try:
    import asnumpy
    import numpy as np
    log("✓ 成功导入 asnumpy 和 numpy")
except ImportError as e:
    log(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_security_vulnerabilities():
    """测试潜在安全漏洞"""
    log("\n=== 测试潜在安全漏洞 ===")
    
    security_tests = [
        # 注入攻击
        ("SQL注入", "'; DROP TABLE users; --"),
        ("XSS攻击", "<script>alert('xss')</script>"),
        ("命令注入", "$(rm -rf /)"),
        ("路径遍历", "../../../etc/passwd"),
        
        # 格式字符串攻击
        ("格式字符串", "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s"),
        ("格式化攻击", "%n%n%n%n%n%n%n%n%n%n"),
        
        # 缓冲区溢出攻击（减少长度避免大量输出）
        ("缓冲区溢出", "A" * 1000),
        ("超大字符串", "float32" * 100),  # 从100000减少到100
        
        # 控制字符攻击
        ("控制字符", "\x00\x01\x02\x03\x04\x05\x06\x07"),
        ("Unicode攻击", "\u0000\u0001\u0002\u0003"),
        ("换行符攻击", "\n\r\n\r\n\r"),
        ("制表符攻击", "\t\t\t\t\t"),
        
        # 特殊字符攻击
        ("特殊字符", "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
        ("中文字符", "中文类型"),
        ("Unicode特殊字符", "float32\u2022\u2023\u2024"),
    ]
    
    for name, value in security_tests:
        try:
            result = asnumpy.ones((2, 2), dtype=value)
            log(f"  ✗ {name}: 安全漏洞 - 应该失败但成功了")
        except Exception as e:
            log(f"  ✓ {name}: 安全 - {type(e).__name__}: {e}")

def test_error_handling_completeness():
    """测试错误处理完整性"""
    log("\n=== 测试错误处理完整性 ===")
    
    error_tests = [
        # 空值和None
        ("None", None),
        ("空字符串", ""),
        ("空列表", []),
        ("空元组", ()),
        ("空字典", {}),
        
        # 无效类型
        ("不存在的类型", "nonexistent_type"),
        ("无效字符串", "invalid_dtype"),
        ("数字字符串", "12345"),
        ("混合字符串", "float32_int64"),
        
        # 无效对象
        ("函数对象", lambda x: x),
        ("类对象", object()),
        ("异常对象", Exception("test")),
        ("生成器", (x for x in range(5))),
        ("迭代器", iter([1, 2, 3])),
        ("模块对象", sys),
        ("类型对象", type),
        
        # 复杂对象
        ("NumPy数组", np.array([1, 2, 3])),
        ("嵌套列表", [[1, 2], [3, 4]]),
        ("嵌套字典", {"a": {"b": 1}, "c": {"d": 2}}),
        
        # 数值边界
        ("无穷大", float('inf')),
        ("NaN", float('nan')),
        ("复数", 1+2j),
        ("极大数", 1e308),
        ("极小数", 1e-308),
    ]
    
    for name, value in error_tests:
        try:
            result = asnumpy.ones((2, 2), dtype=value)
            log(f"  ✗ {name}: 错误处理不完整 - 应该失败但成功了")
        except Exception as e:
            log(f"  ✓ {name}: 正确处理 - {type(e).__name__}: {e}")

def test_numpy_scalar_acceptance():
    """测试 NumPy 标量是否被正确接受"""
    log("\n=== 测试 NumPy 标量接受 ===")
    
    numpy_scalar_tests = [
        ("np.int32(42)", np.int32(42)),
        ("np.float32(3.14)", np.float32(3.14)),
        ("np.bool_(True)", np.bool_(True)),
        ("np.int64(123)", np.int64(123)),
        ("np.float64(2.718)", np.float64(2.718)),
        ("np.uint8(255)", np.uint8(255)),
    ]
    
    for name, value in numpy_scalar_tests:
        try:
            result = asnumpy.ones((2, 2), dtype=value)
            log(f"  ✓ {name}: 正确接受 - 输出dtype: {result.to_numpy().dtype}")
        except Exception as e:
            log(f"  ✗ {name}: 意外拒绝 - {type(e).__name__}: {e}")

def test_custom_malicious_objects():
    """测试自定义恶意对象"""
    log("\n=== 测试自定义恶意对象 ===")
    
    class MaliciousObject:
        def __init__(self):
            self.counter = 0
        
        def __str__(self):
            # 模拟无限递归（限制递归深度）
            self.counter += 1
            if self.counter > 10:  # 从1000减少到10
                return "float32"  # 避免真正的无限递归
            return str(self)
        
        def __repr__(self):
            return self.__str__()
    
    class AttributeErrorObject:
        def __getattr__(self, name):
            if name == 'name':
                raise RuntimeError("模拟属性访问错误")
            return name
    
    class RecursiveObject:
        def __init__(self):
            self.self = self
    
    malicious_tests = [
        ("递归对象", MaliciousObject()),
        ("属性错误对象", AttributeErrorObject()),
        ("循环引用对象", RecursiveObject()),
    ]
    
    for name, value in malicious_tests:
        try:
            result = asnumpy.ones((2, 2), dtype=value)
            log(f"  ✗ {name}: 恶意对象处理失败 - 应该失败但成功了")
        except Exception as e:
            log(f"  ✓ {name}: 正确处理恶意对象 - {type(e).__name__}: {e}")

def test_error_recovery():
    """测试错误恢复能力"""
    log("\n=== 测试错误恢复能力 ===")
    
    # 先测试错误输入
    try:
        asnumpy.ones((2, 2), dtype="invalid_type")
        log("  ✗ 错误输入应该失败")
    except Exception as e:
        log(f"  ✓ 错误输入正确失败: {type(e).__name__}: {e}")
    
    # 然后测试正确输入，确保系统仍然工作
    try:
        result = asnumpy.ones((2, 2), dtype=np.float32)
        log("  ✓ 正确输入仍然工作")
        log(f"    结果形状: {result.to_numpy().shape}")
        log(f"    结果dtype: {result.to_numpy().dtype}")
    except Exception as e:
        log(f"  ✗ 正确输入失败: {type(e).__name__}: {e}")

def test_performance_under_attack():
    """测试攻击下的性能"""
    log("\n=== 测试攻击下的性能 ===")
    
    import time
    
    # 测试大量恶意输入的处理速度（减少数量）
    malicious_inputs = [
        None, "", "invalid", object(), lambda x: x,
        "'; DROP TABLE users; --", "<script>alert('xss')</script>",
        "A" * 100, "\x00\x01\x02\x03", "!@#$%^&*()"  # 从1000减少到100
    ]
    
    start_time = time.time()
    error_count = 0
    
    for _ in range(10):  # 从100减少到10
        for malicious_input in malicious_inputs:
            try:
                asnumpy.ones((2, 2), dtype=malicious_input)
            except:
                error_count += 1
    
    end_time = time.time()
    
    total_tests = 10 * len(malicious_inputs)
    log(f"  处理 {total_tests} 个恶意输入耗时: {(end_time - start_time) * 1000:.2f} ms")
    log(f"  错误处理率: {error_count}/{total_tests} = {error_count/total_tests*100:.1f}%")
    log(f"  平均每个恶意输入处理时间: {((end_time - start_time) * 1000) / total_tests:.3f} ms")

def test_summary():
    """生成测试总结"""
    log("\n=== 测试总结 ===")
    log(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"输出文件: {output_file}")
    log("总结:")
    log("- ✓ 表示函数安全且正确处理了错误输入")
    log("- ✗ 表示存在安全漏洞或错误处理不完整")
    log("- NumPy 标量应该被正确接受")
    log("- 理想情况下，恶意输入应该被正确拒绝，有效输入应该被接受")

if __name__ == "__main__":
    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"ObjToDtype 安全性和错误处理测试报告\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
    
    log("开始测试 ObjToDtype 函数的安全性和错误处理...")
    
    test_security_vulnerabilities()
    test_error_handling_completeness()
    test_numpy_scalar_acceptance()
    test_custom_malicious_objects()
    test_error_recovery()
    test_performance_under_attack()
    test_summary()
    
    log(f"\n=== 测试完成 ===")
    log(f"详细结果已保存到: {output_file}") 
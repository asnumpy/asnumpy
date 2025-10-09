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

def test_dtypes_is_submodule():
    import sys, types, importlib
    import asnumpy

    m = importlib.import_module('asnumpy.dtypes')
    assert isinstance(m, types.ModuleType)
    # asnumpy 命名空间中暴露了 dtypes
    assert getattr(asnumpy, 'dtypes', None) is m
    # sys.modules 中注册了完整模块名
    assert 'asnumpy.dtypes' in sys.modules

def test_float8_e5m2_basic():
    """测试 float8_e5m2 基本功能"""
    import asnumpy.dtypes
    
    # 仅检查类型对象，并确认其可被 numpy.dtype 识别
    assert hasattr(asnumpy.dtypes, 'float8_e5m2'), "float8_e5m2 类型对象未绑定"
    dt = np.dtype(asnumpy.dtypes.float8_e5m2)
    assert dt is not None, "np.dtype 未能识别 float8_e5m2 类型对象"
    print("✓ float8_e5m2 类型对象已绑定且可被 numpy.dtype 识别")

def test_float8_e5m2_scalar():
    """测试 float8_e5m2 标量创建"""
    import asnumpy.dtypes
    
    try:
        # 创建标量
        scalar = asnumpy.dtypes.float8_e5m2(3.14)
        print(f"✓ 成功创建 float8_e5m2 标量: {scalar}")
        
        # 检查类型
        assert type(scalar).__name__ == 'acl_float8_e5m2'
        print(f"✓ 标量类型正确: {type(scalar).__name__}")
        
    except Exception as e:
        print(f"✗ 标量创建失败: {e}")
        raise

def test_float8_e5m2_array():
    """测试 float8_e5m2 数组操作"""
    import asnumpy.dtypes
    
    try:
        # 创建 float32 数组
        float32_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        print(f"原始 float32 数组: {float32_arr}")
        
        # 转换为 float8_e5m2（直接使用类型对象作为 dtype）
        float8_arr = float32_arr.astype(asnumpy.dtypes.float8_e5m2)
        print(f"✓ 成功转换为 float8_e5m2 数组: {float8_arr}")
        print(f"数组 dtype: {float8_arr.dtype}")
        
        # 检查形状和大小
        assert float8_arr.shape == float32_arr.shape
        assert float8_arr.itemsize == 1  # float8 应该是 1 字节
        print(f"✓ 数组形状: {float8_arr.shape}, 元素大小: {float8_arr.itemsize} 字节")
        
        # 转换回 float32 验证
        recovered = float8_arr.astype(np.float32)
        print(f"转换回 float32: {recovered}")
        
        # 检查精度损失在合理范围内
        max_diff = np.max(np.abs(float32_arr - recovered))
        print(f"最大精度损失: {max_diff}")
        assert max_diff < 1.0, f"精度损失过大: {max_diff}"
        
    except Exception as e:
        print(f"✗ 数组操作失败: {e}")
        raise

def test_numpy_float8_e5m2_alias():
    """测试使用 np.float8_e5m2 作为 dtype（要求别名存在）。"""
    import asnumpy.dtypes
    values = [1.0, 2.0, 3.0, 4.0]
    try:
        assert hasattr(np, 'float8_e5m2'), "np.float8_e5m2 别名不存在"
        float8_e5m2_arr = np.array(values, dtype=np.float8_e5m2)
        print("✓ 使用 np.float8_e5m2 创建数组成功")
        # 校验 dtype 与我们注册的类型一致
        assert float8_e5m2_arr.dtype == np.dtype(asnumpy.dtypes.float8_e5m2)
        print("数组: ", float8_e5m2_arr)
    except Exception as e:
        print(f"✗ 使用 np.float8_e5m2/asnumpy.dtypes.float8_e5m2 失败: {e}")
        raise

def test_float8_e5m2_dtype_properties():
    """测试 float8_e5m2 dtype 属性"""
    import asnumpy.dtypes
    
    # 通过 numpy.dtype 从类型对象获取 dtype
    dtype = np.dtype(asnumpy.dtypes.float8_e5m2)
    
    # 检查基本属性
    assert dtype.itemsize == 1, f"预期元素大小为 1，实际为 {dtype.itemsize}"
    assert dtype.kind == 'f', f"预期种类为 'f'，实际为 '{dtype.kind}'"
    
    print(f"✓ dtype 属性正确:")
    print(f"  - 元素大小: {dtype.itemsize} 字节")
    print(f"  - 种类: '{dtype.kind}' (浮点)")
    print(f"  - 名称: {dtype.name}")

    # 通过类型对象静态方法获取 ACL 枚举常量
    assert hasattr(asnumpy.dtypes.float8_e5m2, 'GetACLDataType'), "缺少 GetACLDataType 方法"
    acl_enum = asnumpy.dtypes.float8_e5m2.GetACLDataType()
    print(f"  - ACL 枚举常量: {acl_enum}")

def run_all_tests():
    """运行所有测试"""
    tests = [
        test_dtypes_is_submodule,
        test_float8_e5m2_basic,
        test_float8_e5m2_scalar,
        test_float8_e5m2_array,
        test_numpy_float8_e5m2_alias,
        test_float8_e5m2_dtype_properties,
    ]
    
    print("运行 float8_e5m2 dtype 测试...")
    print("=" * 50)
    
    for test in tests:
        try:
            print(f"\n运行测试: {test.__name__}")
            test()
            print(f"✓ {test.__name__} 通过")
        except Exception as e:
            print(f"✗ {test.__name__} 失败: {e}")
            raise
    
    print("\n" + "=" * 50)
    print("✓ 所有测试通过!")

if __name__ == "__main__":
    run_all_tests()
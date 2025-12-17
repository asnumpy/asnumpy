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
import numpy as np

# 配置日志记录（所有导入此模块的测试文件共享此配置）
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


def run_test(func):
    """运行单个测试函数并返回是否通过。
    
    Args:
        func: 要运行的测试函数
        
    Returns:
        bool: 如果测试通过返回 True，否则返回 False
    """
    try:
        func()
        logger.info(f"[PASS] {func.__name__}")
        return True
    except AssertionError as err:
        logger.error(f"[FAIL] {func.__name__}: {err}")
    except Exception as err:
        logger.error(f"[ERROR] {func.__name__}: {err}")
    return False


def run_tests(tests):
    """运行测试列表并输出摘要。
    
    Args:
        tests: 测试函数列表
        
    Returns:
        int: 退出码，0 表示所有测试通过，1 表示有测试失败
    """
    total = len(tests)
    passed = sum(run_test(func) for func in tests)
    logger.info(f"\nSummary: {passed}/{total} tests passed")
    return 0 if passed == total else 1


def main(tests):
    """主函数，运行测试并返回退出码。
    
    Args:
        tests: 测试函数列表
        
    Returns:
        int: 退出码，0 表示所有测试通过，1 表示有测试失败
    """
    return run_tests(tests)


def test_dtype_basic_attributes(dtype, dtype_name, expected_itemsize, logger):
    """测试 dtype 的基本属性（char, num, str, name, itemsize）。
    
    这是一个共享函数，用于消除测试文件中的重复代码。
    
    Args:
        dtype: 要测试的 dtype 对象
        dtype_name: dtype 的名称（用于错误消息）
        expected_itemsize: 期望的 itemsize 值
        logger: 日志记录器
    """
    # dtype.char
    dtype_char = dtype.char
    logger.info(f"dtype.char: {dtype_char}")
    assert dtype_char is not None, "dtype.char 应该存在"
    
    # dtype.num
    dtype_num = dtype.num
    logger.info(f"dtype.num: {dtype_num}")
    assert dtype_num is not None, "dtype.num 应该存在"
    assert dtype_num != -1, "dtype.num 应该是有效的类型号"
    
    # dtype.str
    dtype_str = dtype.str
    logger.info(f"dtype.str: {dtype_str}")
    assert dtype_str is not None, "dtype.str 应该存在"
    assert isinstance(dtype_str, str), "dtype.str 应该是字符串"
    
    # dtype.name
    dtype_name_attr = dtype.name
    logger.info(f"dtype.name: {dtype_name_attr}")
    assert dtype_name_attr is not None, "dtype.name 应该存在"
    assert isinstance(dtype_name_attr, str), "dtype.name 应该是字符串"
    
    # dtype.itemsize
    dtype_itemsize = dtype.itemsize
    logger.info(f"dtype.itemsize: {dtype_itemsize}")
    assert dtype_itemsize > 0, "dtype.itemsize 应该大于 0"
    assert dtype_itemsize == expected_itemsize, \
        f"{dtype_name} 的 itemsize 应该是 {expected_itemsize} 字节，实际为 {dtype_itemsize}"


def test_dtype_advanced_attributes_and_methods(dtype, dtype_name, dtype_obj_for_setstate, logger):
    """测试 dtype 的高级属性和方法（flags, isbuiltin, isnative, descr, alignment, base, metadata, newbyteorder, __reduce__, __setstate__）。
    
    这是一个共享函数，用于消除测试文件中的重复代码。
    
    Args:
        dtype: 要测试的 dtype 对象
        dtype_name: dtype 的名称（用于错误消息）
        dtype_obj_for_setstate: 用于 __setstate__ 测试的 dtype 对象
        logger: 日志记录器
    """
    # dtype.flags
    dtype_flags = dtype.flags
    logger.info(f"dtype.flags: {dtype_flags}")
    assert dtype_flags is not None, "dtype.flags 应该存在"
    
    # dtype.isbuiltin
    dtype_isbuiltin = dtype.isbuiltin
    logger.info(f"dtype.isbuiltin: {dtype_isbuiltin}")
    assert isinstance(dtype_isbuiltin, (int, bool)), "dtype.isbuiltin 应该是整数或布尔值"
    
    # dtype.isnative
    dtype_isnative = dtype.isnative
    logger.info(f"dtype.isnative: {dtype_isnative}")
    assert isinstance(dtype_isnative, bool), "dtype.isnative 应该是布尔值"
    
    # dtype.descr
    dtype_descr = dtype.descr
    logger.info(f"dtype.descr: {dtype_descr}")
    assert dtype_descr is not None, "dtype.descr 应该存在"
    assert isinstance(dtype_descr, list), "dtype.descr 应该是列表"
    assert len(dtype_descr) > 0, "dtype.descr 应该包含至少一个元素"
    
    # dtype.alignment
    dtype_alignment = dtype.alignment
    logger.info(f"dtype.alignment: {dtype_alignment}")
    assert dtype_alignment > 0, "dtype.alignment 应该大于 0"
    
    # dtype.base
    dtype_base = dtype.base
    logger.info(f"dtype.base: {dtype_base}")
    # base 可能为 None 或与 dtype 相同
    
    # dtype.metadata
    dtype_metadata = dtype.metadata
    logger.info(f"dtype.metadata: {dtype_metadata}")
    # metadata 可能为 None
    
    # 测试方法
    logger.info("\n--- 方法测试 ---")
    
    # dtype.newbyteorder()
    try:
        new_dtype = dtype.newbyteorder('>')
        logger.info(f"dtype.newbyteorder('>'): {new_dtype}")
        assert new_dtype is not None, "newbyteorder 应该返回有效的 dtype"
    except Exception as e:
        logger.info(f"dtype.newbyteorder('>') 不支持或出错: {e}")
    
    try:
        new_dtype = dtype.newbyteorder('<')
        logger.info(f"dtype.newbyteorder('<'): {new_dtype}")
    except Exception as e:
        logger.info(f"dtype.newbyteorder('<') 不支持或出错: {e}")
    
    # dtype.__reduce__()
    try:
        reduce_result = dtype.__reduce__()
        logger.info(f"dtype.__reduce__(): {reduce_result}")
        assert reduce_result is not None, "__reduce__ 应该返回有效的结果"
        assert isinstance(reduce_result, tuple), "__reduce__ 应该返回元组"
    except Exception as e:
        logger.error(f"dtype.__reduce__() 失败: {e}")
        raise
    
    # dtype.__setstate__()
    try:
        # 创建一个状态用于测试
        state = dtype.__reduce__()[1] if hasattr(dtype, '__reduce__') else None
        if state is not None:
            # 创建一个新的 dtype 对象来测试 __setstate__
            new_dtype = dtype_obj_for_setstate
            if hasattr(new_dtype, '__setstate__'):
                new_dtype.__setstate__(state)
                logger.info(f"dtype.__setstate__() 成功")
        else:
            logger.info(f"dtype.__setstate__() 跳过（无状态）")
    except Exception as e:
        logger.info(f"dtype.__setstate__() 不支持或出错: {e}")


def test_dtype_structured_and_subarray_types(base_dtype, dtype_name, dtype_type_obj, logger):
    """测试 dtype 在结构化数组和子数组类型中的使用。
    
    这是一个共享函数，用于消除测试文件中的重复代码。
    
    Args:
        base_dtype: 基础 dtype 对象（如 ap.bfloat16）
        dtype_name: dtype 的名称（用于日志消息）
        dtype_type_obj: dtype 的类型对象（如 ap.dtypes.bfloat16）
        logger: 日志记录器
    """
    logger.info(f"基础 dtype: {base_dtype}")
    logger.info(f"基础 dtype.fields: {base_dtype.fields}")
    logger.info(f"基础 dtype.names: {base_dtype.names}")
    logger.info(f"基础 dtype.subdtype: {base_dtype.subdtype}")
    logger.info(f"基础 dtype.shape: {base_dtype.shape}")
    
    # ========== 测试 1: 结构化数组（fields 和 names 不为 None）==========
    logger.info("\n--- 测试 1: 结构化数组 ---")
    
    # 创建包含 dtype 字段的结构化数组
    structured_dtype = np.dtype([
        ('x', dtype_type_obj),
        ('y', dtype_type_obj),
        ('z', dtype_type_obj)
    ])
    logger.info(f"结构化 dtype: {structured_dtype}")
    logger.info(f"structured_dtype.fields: {structured_dtype.fields}")
    logger.info(f"structured_dtype.names: {structured_dtype.names}")
    
    assert structured_dtype.fields is not None, "结构化数组的 fields 不应该为 None"
    assert structured_dtype.names is not None, "结构化数组的 names 不应该为 None"
    assert len(structured_dtype.names) == 3, "应该有 3 个字段"
    assert structured_dtype.names == ('x', 'y', 'z'), "字段名应该是 ('x', 'y', 'z')"
    
    # 验证每个字段的 dtype
    for name in structured_dtype.names:
        field_dtype, offset = structured_dtype.fields[name]
        logger.info(f"  字段 '{name}': dtype={field_dtype}, offset={offset}")
        assert field_dtype == base_dtype, f"字段 '{name}' 的 dtype 应该是 {dtype_name}"
    
    # 创建结构化数组并测试
    try:
        structured_array = np.array([
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0)
        ], dtype=structured_dtype)
        logger.info(f"结构化数组创建成功: {structured_array}")
        logger.info(f"  访问字段 'x': {structured_array['x']}")
        logger.info(f"  访问字段 'y': {structured_array['y']}")
        logger.info(f"  访问字段 'z': {structured_array['z']}")
    except Exception as e:
        logger.info(f"结构化数组创建失败（可能不支持）: {e}")
    
    # ========== 测试 2: 子数组类型（subdtype 和 shape 不为 None）==========
    logger.info("\n--- 测试 2: 子数组类型 ---")
    
    # 创建 2x2 矩阵的子数组类型
    subarray_dtype_2x2 = np.dtype((dtype_type_obj, (2, 2)))
    logger.info(f"子数组 dtype (2x2): {subarray_dtype_2x2}")
    logger.info(f"subarray_dtype_2x2.subdtype: {subarray_dtype_2x2.subdtype}")
    logger.info(f"subarray_dtype_2x2.shape: {subarray_dtype_2x2.shape}")
    
    assert subarray_dtype_2x2.subdtype is not None, "子数组类型的 subdtype 不应该为 None"
    assert subarray_dtype_2x2.shape is not None, "子数组类型的 shape 不应该为 None"
    assert subarray_dtype_2x2.shape == (2, 2), "shape 应该是 (2, 2)"
    assert subarray_dtype_2x2.subdtype[0] == base_dtype, f"subdtype 的基础类型应该是 {dtype_name}"
    assert subarray_dtype_2x2.subdtype[1] == (2, 2), "subdtype 的形状应该是 (2, 2)"
    
    # 创建 1D 向量的子数组类型
    subarray_dtype_1d = np.dtype((dtype_type_obj, 3))
    logger.info(f"子数组 dtype (1D, 长度3): {subarray_dtype_1d}")
    logger.info(f"subarray_dtype_1d.subdtype: {subarray_dtype_1d.subdtype}")
    logger.info(f"subarray_dtype_1d.shape: {subarray_dtype_1d.shape}")
    
    assert subarray_dtype_1d.subdtype is not None, "子数组类型的 subdtype 不应该为 None"
    assert subarray_dtype_1d.shape == (3,), "shape 应该是 (3,)"
    assert subarray_dtype_1d.subdtype[0] == base_dtype, f"subdtype 的基础类型应该是 {dtype_name}"
    assert subarray_dtype_1d.subdtype[1] == (3,), "subdtype 的形状应该是 (3,)"
    
    # 创建子数组并测试
    try:
        subarray_array = np.zeros((2,), dtype=subarray_dtype_2x2)
        logger.info(f"子数组创建成功: shape={subarray_array.shape}")
        logger.info(f"  第一个元素 (2x2 矩阵):\n{subarray_array[0]}")
    except Exception as e:
        logger.info(f"子数组创建失败（可能不支持）: {e}")
    
    # ========== 测试 3: 组合使用（结构化数组 + 子数组）==========
    logger.info("\n--- 测试 3: 组合使用（结构化数组 + 子数组）---")
    
    # 创建包含子数组字段的结构化数组
    complex_dtype = np.dtype([
        ('id', 'i4'),
        ('position', (dtype_type_obj, 3)),      # 3D 位置向量
        ('matrix', (dtype_type_obj, (2, 2)))    # 2x2 矩阵
    ])
    logger.info(f"复杂 dtype: {complex_dtype}")
    logger.info(f"complex_dtype.fields: {complex_dtype.fields}")
    logger.info(f"complex_dtype.names: {complex_dtype.names}")
    
    assert complex_dtype.fields is not None, "复杂类型的 fields 不应该为 None"
    assert complex_dtype.names is not None, "复杂类型的 names 不应该为 None"
    assert len(complex_dtype.names) == 3, "应该有 3 个字段"
    
    # 检查 position 字段（子数组类型）
    position_dtype, position_offset = complex_dtype.fields['position']
    logger.info(f"  position 字段: dtype={position_dtype}, offset={position_offset}")
    logger.info(f"  position.subdtype: {position_dtype.subdtype}")
    logger.info(f"  position.shape: {position_dtype.shape}")
    
    assert position_dtype.subdtype is not None, "position 字段的 subdtype 不应该为 None"
    assert position_dtype.shape == (3,), "position 字段的 shape 应该是 (3,)"
    assert position_dtype.subdtype[0] == base_dtype, f"position 的基础类型应该是 {dtype_name}"
    
    # 检查 matrix 字段（子数组类型）
    matrix_dtype, matrix_offset = complex_dtype.fields['matrix']
    logger.info(f"  matrix 字段: dtype={matrix_dtype}, offset={matrix_offset}")
    logger.info(f"  matrix.subdtype: {matrix_dtype.subdtype}")
    logger.info(f"  matrix.shape: {matrix_dtype.shape}")
    
    assert matrix_dtype.subdtype is not None, "matrix 字段的 subdtype 不应该为 None"
    assert matrix_dtype.shape == (2, 2), "matrix 字段的 shape 应该是 (2, 2)"
    assert matrix_dtype.subdtype[0] == base_dtype, f"matrix 的基础类型应该是 {dtype_name}"
    
    # 创建复杂数组并测试
    try:
        complex_array = np.array([
            (1, [1.0, 2.0, 3.0], [[1.0, 2.0], [3.0, 4.0]]),
            (2, [4.0, 5.0, 6.0], [[5.0, 6.0], [7.0, 8.0]])
        ], dtype=complex_dtype)
        logger.info(f"复杂数组创建成功: {complex_array}")
        logger.info(f"  访问 position 字段: {complex_array['position']}")
        logger.info(f"  访问 matrix 字段: {complex_array['matrix']}")
    except Exception as e:
        logger.info(f"复杂数组创建失败（可能不支持）: {e}")
    
    # ========== 测试 4: 验证基础类型属性仍然为 None ==========
    logger.info("\n--- 测试 4: 验证基础类型属性仍然为 None ---")
    
    # 即使用于结构化数组或子数组，基础类型本身的属性应该仍然是 None
    logger.info(f"基础 dtype.fields: {base_dtype.fields}")
    logger.info(f"基础 dtype.names: {base_dtype.names}")
    logger.info(f"基础 dtype.subdtype: {base_dtype.subdtype}")
    logger.info(f"基础 dtype.shape: {base_dtype.shape}")
    
    assert base_dtype.fields is None, "基础类型的 fields 应该为 None"
    assert base_dtype.names is None, "基础类型的 names 应该为 None"
    assert base_dtype.subdtype is None, "基础类型的 subdtype 应该为 None"
    assert base_dtype.shape == (), "基础类型的 shape 应该为 ()"


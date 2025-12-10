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

import sys

import numpy as np

from test_utils import main, logger
import asnumpy as ap


def test_float8_e5m2_is_registered():
    """检查 float8_e5m2 是否已注册"""
    is_registered = ap.dtypes.check_float8_e5m2_registered()
    assert is_registered, "float8_e5m2 应该已注册"
    logger.info(f"[PASS] float8_e5m2 注册状态: {is_registered}")


def test_float8_e5m2_type_num():
    """检查 float8_e5m2 的类型号"""
    type_num = ap.dtypes.get_float8_e5m2_type_num()
    assert type_num != -1, "float8_e5m2 类型号应该有效"
    logger.info(f"[PASS] float8_e5m2 类型号: {type_num}")


def test_float8_e5m2_is_bound():
    """检查 float8_e5m2 是否已绑定到子模块"""
    assert hasattr(ap.dtypes, "float8_e5m2"), "float8_e5m2 应该绑定到 ap.dtypes"
    dtype_obj = ap.dtypes.float8_e5m2
    assert dtype_obj is not None, "float8_e5m2 类型对象不应为空"
    logger.info(f"[PASS] float8_e5m2 已绑定到子模块: {dtype_obj}")


def test_float8_e5m2_top_level_access():
    """检查 float8_e5m2 是否可以通过 ap.float8_e5m2 访问"""
    assert hasattr(ap, "float8_e5m2"), "float8_e5m2 应该可以通过 ap.float8_e5m2 访问"
    dtype_obj = ap.float8_e5m2
    assert dtype_obj is not None, "ap.float8_e5m2 类型对象不应为空"
    # 验证 ap.float8_e5m2 和 ap.dtypes.float8_e5m2 是同一个对象
    assert ap.float8_e5m2 is ap.dtypes.float8_e5m2, "ap.float8_e5m2 应该与 ap.dtypes.float8_e5m2 是同一个对象"
    logger.info(f"[PASS] float8_e5m2 可通过 ap.float8_e5m2 访问: {dtype_obj}")


def test_float8_e5m2_as_numpy_dtype():
    """检查 ap.float8_e5m2 是否可以直接作为 numpy dtype 使用（不需要 np.dtype()）"""
    # 测试直接使用 ap.float8_e5m2 创建数组
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.float8_e5m2)
    assert arr.dtype == np.dtype(ap.float8_e5m2), "数组 dtype 应该匹配"
    assert arr.dtype == np.dtype(ap.dtypes.float8_e5m2), "ap.float8_e5m2 和 ap.dtypes.float8_e5m2 应该产生相同的 dtype"
    
    # 测试直接使用导入的 float8_e5m2 变量名（通过 NumPy C API 注册的类型应该可以直接使用）
    from asnumpy import float8_e5m2
    dtype_direct = np.dtype(float8_e5m2)
    assert dtype_direct is not None, "应该能够使用 np.dtype(float8_e5m2) 创建 dtype"
    assert dtype_direct == arr.dtype, "np.dtype(float8_e5m2) 应该与通过 ap.float8_e5m2 创建的 dtype 相同"
    logger.info(f"[PASS] ap.float8_e5m2 可以直接作为 numpy dtype 使用: {arr.dtype}")
    logger.info(f"[PASS] np.dtype(float8_e5m2) 可以直接使用: {dtype_direct}")


def test_float8_e5m2_can_create_dtype():
    """检查是否可以使用 float8_e5m2 创建 numpy dtype"""
    dtype = np.dtype(ap.float8_e5m2)
    assert dtype is not None, "应该能够创建 numpy dtype"
    logger.info(f"[PASS] 成功创建 numpy dtype: {dtype}")


def test_float8_e5m2_can_create_array():
    """检查是否可以使用 float8_e5m2 创建数组"""
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.float8_e5m2)
    assert arr.dtype == np.dtype(ap.float8_e5m2), "数组 dtype 应该匹配"
    logger.info(f"[PASS] 成功创建数组: {arr}, dtype: {arr.dtype}")


def test_float8_e5m2_scalar_creation():
    """测试 float8_e5m2 标量创建"""
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    # 测试创建标量
    scalar = ap.float8_e5m2(3.14)
    assert scalar is not None, "应该能够创建标量对象"
    
    # 验证 ACL 枚举值
    assert hasattr(scalar, 'getACLenum'), "标量应该有 getACLenum 方法"
    acl_enum = scalar.getACLenum()
    assert acl_enum == expected_acl_value, \
        f"ACL枚举值应该为 {expected_acl_value} (ACL_FLOAT8_E5M2)，实际为 {acl_enum}"
    
    logger.info(f"[PASS] 成功创建标量，ACL枚举值: {acl_enum} (期望: {expected_acl_value})")


def test_float8_e5m2_get_acl_enum():
    """检查 float8_e5m2 的 getACLenum() 接口"""
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    scalar = ap.float8_e5m2(0)
    assert scalar is not None, "应该能够创建标量对象"
    assert hasattr(scalar, 'getACLenum'), "getACLenum 方法应该存在"
    
    acl_enum = scalar.getACLenum()
    assert acl_enum == expected_acl_value, f"ACL枚举值应该为 {expected_acl_value}，实际为 {acl_enum}"
    
    # 测试不同标量值的 getACLenum 是否一致
    scalar2 = ap.float8_e5m2(3.14)
    acl_enum2 = scalar2.getACLenum()
    assert acl_enum == acl_enum2, "不同标量值的ACL枚举值应该一致"
    
    logger.info(f"[PASS] float8_e5m2 getACLenum() = {acl_enum} (期望: {expected_acl_value})")


def test_float8_e5m2_get_acl_data_type():
    """测试 GetACLDataType 函数是否能正确识别 float8_e5m2 并返回正确的 ACL 类型"""
    # 创建 float8_e5m2 类型的数组
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.float8_e5m2)
    assert arr.dtype == np.dtype(ap.float8_e5m2), "数组 dtype 应该匹配"
    
    # 使用 ndarray.from_numpy 创建 NPUArray，这会调用 GetACLDataType
    try:
        npu_arr = ap.ndarray.from_numpy(arr)
        expected_acl_value = 35  # ACL_FLOAT8_E5M2
        
        # 检查 aclDtype 是否正确
        assert npu_arr.aclDtype == expected_acl_value, \
            f"ACL类型应该为 {expected_acl_value}，实际为 {npu_arr.aclDtype}"
        
        logger.info(f"[PASS] GetACLDataType 正确识别 float8_e5m2: ACL类型 = {npu_arr.aclDtype}")
        logger.info(f"[PASS] NPUArray 创建成功: shape={npu_arr.shape}, dtype={npu_arr.dtype}")
        
    except Exception as e:
        logger.error(f"[FAIL] GetACLDataType 测试失败: {e}")
        raise


def test_float8_e5m2_static_get_acl_enum():
    """测试静态方法 getACLenum 是否可以直接调用（C++层面）"""
    # 这个测试验证 Python 层面的接口
    # C++ 层面的静态方法调用需要通过 GetACLDataType 间接测试
    
    # 创建多个不同值的数组，验证都能正确识别
    test_values = [
        [0.0],
        [1.0, 2.0],
        [3.14, 2.71, 1.41],
    ]
    
    for values in test_values:
        arr = np.array(values, dtype=ap.float8_e5m2)
        npu_arr = ap.ndarray.from_numpy(arr)
        
        # 所有数组的 ACL 类型应该都是 ACL_FLOAT8_E5M2 (35)
        assert npu_arr.aclDtype == 35, \
            f"所有 float8_e5m2 数组的 ACL 类型应该为 35，实际为 {npu_arr.aclDtype}"
    
    logger.info(f"[PASS] 静态方法 getACLenum 通过 GetACLDataType 间接测试成功")


def _verify_array_properties(arr, expected_shape, expected_acl_value, dtype, operator_name):
    """辅助函数：验证数组属性"""
    assert arr.aclDtype == expected_acl_value, \
        f"{operator_name} 创建的数组 ACL 类型应该为 {expected_acl_value}，实际为 {arr.aclDtype}"
    assert list(arr.shape) == list(expected_shape), \
        f"{operator_name} 创建的数组形状应该为 {expected_shape}，实际为 {arr.shape}"
    cpu_arr = arr.to_numpy()
    assert cpu_arr.dtype == dtype, \
        f"{operator_name} 转换后的 numpy 数组 dtype 应该为 float8_e5m2，实际为 {cpu_arr.dtype}"
    logger.info(f"[PASS] {operator_name} 创建成功: shape={arr.shape}, aclDtype={arr.aclDtype}")
    logger.info(f"[PASS] 转换到 numpy: dtype={cpu_arr.dtype}, shape={cpu_arr.shape}")


def test_float8_e5m2_ones_operator():
    """测试 ones 算子"""
    dtype = np.dtype(ap.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试 ones 算子:")
    ones_arr = ap.ones(shape=(3, 4), dtype=dtype)
    _verify_array_properties(ones_arr, (3, 4), expected_acl_value, dtype, "ones")


def test_float8_e5m2_zeros_operator():
    """测试 zeros 算子"""
    dtype = np.dtype(ap.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试 zeros 算子:")
    zeros_arr = ap.zeros(shape=(2, 3), dtype=dtype)
    _verify_array_properties(zeros_arr, (2, 3), expected_acl_value, dtype, "zeros")


def test_float8_e5m2_full_operator():
    """测试 full 算子"""
    dtype = np.dtype(ap.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试 full 算子:")
    full_value = 2.5
    full_arr = ap.full(shape=(2, 2), value=full_value, dtype=dtype)
    _verify_array_properties(full_arr, (2, 2), expected_acl_value, dtype, "full")
    logger.info(f"[PASS] full 值: {full_value}")


def test_float8_e5m2_npuarray_constructor():
    """测试 NPUArray 构造函数"""
    dtype = np.dtype(ap.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试 NPUArray 构造函数:")
    npu_arr = ap.ndarray(shape=(2, 3), dtype=dtype)
    _verify_array_properties(npu_arr, (2, 3), expected_acl_value, dtype, "NPUArray构造函数")


def test_float8_e5m2_empty_operator():
    """测试 empty 算子和不同形状"""
    dtype = np.dtype(ap.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试不同形状:")
    test_shapes = [
        (5,),
        (2, 3),
        (1, 2, 3),
    ]
    for shape in test_shapes:
        test_arr = ap.empty(shape=shape, dtype=dtype)
        _verify_array_properties(test_arr, shape, expected_acl_value, dtype, f"empty(shape={shape})")


def test_float8_e5m2_array_creation_operators():
    """测试使用 float8_e5m2 类型调用 asnumpy 封装的算子（ones, zeros, full）"""
    test_float8_e5m2_ones_operator()
    test_float8_e5m2_npuarray_constructor()
    test_float8_e5m2_zeros_operator()
    test_float8_e5m2_full_operator()
    test_float8_e5m2_empty_operator()
    logger.info(f"\n[PASS] float8_e5m2 数组创建算子测试成功（ones, zeros, full）")


def test_float8_e5m2_numpy_dtype_attributes():
    """测试 float8_e5m2 的 NumPy dtype 属性和方法"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 float8_e5m2 NumPy dtype 接口")
    logger.info("=" * 60)
    
    # 获取 dtype 对象
    dtype = np.dtype(ap.float8_e5m2)
    assert dtype is not None, "应该能够创建 dtype 对象"
    logger.info(f"dtype 对象: {dtype}")
    
    # 测试基本属性
    logger.info("\n--- 基本属性测试 ---")
    
    # dtype.type
    dtype_type = dtype.type
    logger.info(f"dtype.type: {dtype_type}")
    assert dtype_type is not None, "dtype.type 应该存在"
    
    # dtype.kind
    dtype_kind = dtype.kind
    logger.info(f"dtype.kind: {dtype_kind}")
    assert dtype_kind is not None, "dtype.kind 应该存在"
    assert dtype_kind == 'f', f"float8_e5m2 的 kind 应该是 'f'，实际为 {dtype_kind}"
    
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
    dtype_name = dtype.name
    logger.info(f"dtype.name: {dtype_name}")
    assert dtype_name is not None, "dtype.name 应该存在"
    assert isinstance(dtype_name, str), "dtype.name 应该是字符串"
    
    # dtype.itemsize
    dtype_itemsize = dtype.itemsize
    logger.info(f"dtype.itemsize: {dtype_itemsize}")
    assert dtype_itemsize > 0, "dtype.itemsize 应该大于 0"
    assert dtype_itemsize == 1, f"float8_e5m2 的 itemsize 应该是 1 字节，实际为 {dtype_itemsize}"
    
    # dtype.byteorder
    dtype_byteorder = dtype.byteorder
    logger.info(f"dtype.byteorder: {dtype_byteorder}")
    assert dtype_byteorder is not None, "dtype.byteorder 应该存在"
    assert dtype_byteorder in ['<', '>', '=', '|'], f"byteorder 应该是有效的字符，实际为 {dtype_byteorder}"
    
    # dtype.fields
    dtype_fields = dtype.fields
    logger.info(f"dtype.fields: {dtype_fields}")
    # fields 可能为 None（对于非结构化类型）
    
    # dtype.names
    dtype_names = dtype.names
    logger.info(f"dtype.names: {dtype_names}")
    # names 可能为 None（对于非结构化类型）
    
    # dtype.subdtype
    dtype_subdtype = dtype.subdtype
    logger.info(f"dtype.subdtype: {dtype_subdtype}")
    # subdtype 可能为 None（对于非子数组类型）
    
    # dtype.shape
    dtype_shape = dtype.shape
    logger.info(f"dtype.shape: {dtype_shape}")
    # shape 可能为 ()（对于标量类型）
    
    # dtype.hasobject
    dtype_hasobject = dtype.hasobject
    logger.info(f"dtype.hasobject: {dtype_hasobject}")
    assert isinstance(dtype_hasobject, bool), "dtype.hasobject 应该是布尔值"
    assert not dtype_hasobject, "float8_e5m2 不应该包含对象引用"
    
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
            new_dtype = np.dtype(ap.float8_e5m2)
            if hasattr(new_dtype, '__setstate__'):
                new_dtype.__setstate__(state)
                logger.info(f"dtype.__setstate__() 成功")
        else:
            logger.info(f"dtype.__setstate__() 跳过（无状态）")
    except Exception as e:
        logger.info(f"dtype.__setstate__() 不支持或出错: {e}")
    
    # dtype.__class_getitem__()
    try:
        # 这是一个类方法，用于类型提示
        if hasattr(type(dtype), '__class_getitem__'):
            result = type(dtype).__class_getitem__(dtype)
            logger.info(f"dtype.__class_getitem__(): {result}")
        else:
            logger.info(f"dtype.__class_getitem__() 不存在（可能不需要）")
    except Exception as e:
        logger.info(f"dtype.__class_getitem__() 不支持或出错: {e}")
    
    # 比较操作符测试
    logger.info("\n--- 比较操作符测试 ---")
    
    # 创建另一个相同类型的 dtype 用于比较
    dtype2 = np.dtype(ap.float8_e5m2)
    other_dtype = np.dtype(np.float32)
    
    # dtype.__ge__() (>=)
    try:
        ge_result = dtype.__ge__(dtype2)
        logger.info(f"dtype.__ge__(dtype2): {ge_result}")
        assert isinstance(ge_result, bool), "__ge__ 应该返回布尔值"
    except Exception as e:
        logger.info(f"dtype.__ge__() 不支持或出错: {e}")
    
    try:
        ge_result = dtype.__ge__(other_dtype)
        logger.info(f"dtype.__ge__(other_dtype): {ge_result}")
    except Exception as e:
        logger.info(f"dtype.__ge__(other_dtype) 不支持或出错: {e}")
    
    # dtype.__gt__() (>)
    try:
        gt_result = dtype.__gt__(dtype2)
        logger.info(f"dtype.__gt__(dtype2): {gt_result}")
        assert isinstance(gt_result, bool), "__gt__ 应该返回布尔值"
    except Exception as e:
        logger.info(f"dtype.__gt__() 不支持或出错: {e}")
    
    # dtype.__le__() (<=)
    try:
        le_result = dtype.__le__(dtype2)
        logger.info(f"dtype.__le__(dtype2): {le_result}")
        assert isinstance(le_result, bool), "__le__ 应该返回布尔值"
    except Exception as e:
        logger.info(f"dtype.__le__() 不支持或出错: {e}")
    
    # dtype.__lt__() (<)
    try:
        lt_result = dtype.__lt__(dtype2)
        logger.info(f"dtype.__lt__(dtype2): {lt_result}")
        assert isinstance(lt_result, bool), "__lt__ 应该返回布尔值"
    except Exception as e:
        logger.info(f"dtype.__lt__() 不支持或出错: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("[PASS] float8_e5m2 NumPy dtype 接口测试完成")
    logger.info("=" * 60)


def test_float8_e5m2_structured_and_subarray_types():
    """测试 float8_e5m2 在结构化数组和子数组类型中的使用"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 float8_e5m2 结构化数组和子数组类型")
    logger.info("=" * 60)
    
    base_dtype = np.dtype(ap.float8_e5m2)
    logger.info(f"基础 dtype: {base_dtype}")
    logger.info(f"基础 dtype.fields: {base_dtype.fields}")
    logger.info(f"基础 dtype.names: {base_dtype.names}")
    logger.info(f"基础 dtype.subdtype: {base_dtype.subdtype}")
    logger.info(f"基础 dtype.shape: {base_dtype.shape}")
    
    # ========== 测试 1: 结构化数组（fields 和 names 不为 None）==========
    logger.info("\n--- 测试 1: 结构化数组 ---")
    
    # 创建包含 float8_e5m2 字段的结构化数组
    structured_dtype = np.dtype([
        ('x', ap.float8_e5m2),
        ('y', ap.float8_e5m2),
        ('z', ap.float8_e5m2)
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
        assert field_dtype == base_dtype, f"字段 '{name}' 的 dtype 应该是 float8_e5m2"
    
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
    subarray_dtype_2x2 = np.dtype((ap.float8_e5m2, (2, 2)))
    logger.info(f"子数组 dtype (2x2): {subarray_dtype_2x2}")
    logger.info(f"subarray_dtype_2x2.subdtype: {subarray_dtype_2x2.subdtype}")
    logger.info(f"subarray_dtype_2x2.shape: {subarray_dtype_2x2.shape}")
    
    assert subarray_dtype_2x2.subdtype is not None, "子数组类型的 subdtype 不应该为 None"
    assert subarray_dtype_2x2.shape is not None, "子数组类型的 shape 不应该为 None"
    assert subarray_dtype_2x2.shape == (2, 2), "shape 应该是 (2, 2)"
    assert subarray_dtype_2x2.subdtype[0] == base_dtype, "subdtype 的基础类型应该是 float8_e5m2"
    assert subarray_dtype_2x2.subdtype[1] == (2, 2), "subdtype 的形状应该是 (2, 2)"
    
    # 创建 1D 向量的子数组类型
    subarray_dtype_1d = np.dtype((ap.float8_e5m2, 3))
    logger.info(f"子数组 dtype (1D, 长度3): {subarray_dtype_1d}")
    logger.info(f"subarray_dtype_1d.subdtype: {subarray_dtype_1d.subdtype}")
    logger.info(f"subarray_dtype_1d.shape: {subarray_dtype_1d.shape}")
    
    assert subarray_dtype_1d.subdtype is not None, "子数组类型的 subdtype 不应该为 None"
    assert subarray_dtype_1d.shape == (3,), "shape 应该是 (3,)"
    assert subarray_dtype_1d.subdtype[0] == base_dtype, "subdtype 的基础类型应该是 float8_e5m2"
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
        ('position', (ap.float8_e5m2, 3)),      # 3D 位置向量
        ('matrix', (ap.float8_e5m2, (2, 2)))    # 2x2 矩阵
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
    assert position_dtype.subdtype[0] == base_dtype, "position 的基础类型应该是 float8_e5m2"
    
    # 检查 matrix 字段（子数组类型）
    matrix_dtype, matrix_offset = complex_dtype.fields['matrix']
    logger.info(f"  matrix 字段: dtype={matrix_dtype}, offset={matrix_offset}")
    logger.info(f"  matrix.subdtype: {matrix_dtype.subdtype}")
    logger.info(f"  matrix.shape: {matrix_dtype.shape}")
    
    assert matrix_dtype.subdtype is not None, "matrix 字段的 subdtype 不应该为 None"
    assert matrix_dtype.shape == (2, 2), "matrix 字段的 shape 应该是 (2, 2)"
    assert matrix_dtype.subdtype[0] == base_dtype, "matrix 的基础类型应该是 float8_e5m2"
    
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
    
    logger.info("\n" + "=" * 60)
    logger.info("[PASS] float8_e5m2 结构化数组和子数组类型测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    tests = [
        test_float8_e5m2_is_registered,
        test_float8_e5m2_type_num,
        test_float8_e5m2_is_bound,
        test_float8_e5m2_top_level_access,
        test_float8_e5m2_as_numpy_dtype,
        test_float8_e5m2_can_create_dtype,
        test_float8_e5m2_can_create_array,
        test_float8_e5m2_scalar_creation,
        test_float8_e5m2_get_acl_enum,
        test_float8_e5m2_get_acl_data_type,  
        test_float8_e5m2_static_get_acl_enum,
        test_float8_e5m2_array_creation_operators,
        test_float8_e5m2_numpy_dtype_attributes,
        test_float8_e5m2_structured_and_subarray_types,
    ]
    sys.exit(main(tests))


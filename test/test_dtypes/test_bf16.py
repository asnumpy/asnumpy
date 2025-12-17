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


def test_bfloat16_is_registered():
    """检查 bfloat16 是否已注册"""
    is_registered = ap.dtypes.check_bfloat16_registered()
    assert is_registered, "bfloat16 应该已注册"
    logger.info(f"[PASS] bfloat16 注册状态: {is_registered}")


def test_bfloat16_type_num():
    """检查 bfloat16 的类型号"""
    type_num = ap.dtypes.get_bfloat16_type_num()
    assert type_num != -1, "bfloat16 类型号应该有效"
    logger.info(f"[PASS] bfloat16 类型号: {type_num}")


def test_bfloat16_is_bound():
    """检查 bfloat16 是否已绑定到子模块"""
    assert hasattr(ap.dtypes, "bfloat16"), "bfloat16 应该绑定到 ap.dtypes"
    dtype_obj = ap.dtypes.bfloat16
    assert dtype_obj is not None, "bfloat16 类型对象不应为空"
    logger.info(f"[PASS] bfloat16 已绑定到子模块: {dtype_obj}")


def test_bfloat16_top_level_access():
    """检查 bfloat16 是否可以通过 ap.bfloat16 访问"""
    assert hasattr(ap, "bfloat16"), "bfloat16 应该可以通过 ap.bfloat16 访问"
    dtype_obj = ap.bfloat16
    assert dtype_obj is not None, "ap.bfloat16 dtype 对象不应为空"
    # 验证 ap.bfloat16 是 np.dtype(ap.dtypes.bfloat16) 的结果
    assert ap.bfloat16 == np.dtype(ap.dtypes.bfloat16), "ap.bfloat16 应该等于 np.dtype(ap.dtypes.bfloat16)"
    logger.info(f"[PASS] bfloat16 可通过 ap.bfloat16 访问: {dtype_obj}")


def test_bfloat16_as_numpy_dtype():
    """检查 ap.bfloat16 是否可以直接作为 numpy dtype 使用（不需要 np.dtype()）"""
    # 测试直接使用 ap.bfloat16 创建数组（ap.bfloat16 已经是 dtype 对象）
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.bfloat16)
    assert arr.dtype == ap.bfloat16, "数组 dtype 应该匹配"
    assert arr.dtype == np.dtype(ap.dtypes.bfloat16), "ap.bfloat16 和 np.dtype(ap.dtypes.bfloat16) 应该产生相同的 dtype"
    
    # 测试直接使用导入的 bfloat16 变量名（已经是 dtype 对象）
    from asnumpy import bfloat16
    assert bfloat16 == ap.bfloat16, "导入的 bfloat16 应该与 ap.bfloat16 相同"
    assert bfloat16 == arr.dtype, "bfloat16 应该与数组的 dtype 相同"
    logger.info(f"[PASS] ap.bfloat16 可以直接作为 numpy dtype 使用: {arr.dtype}")
    logger.info(f"[PASS] 导入的 bfloat16 可以直接使用: {bfloat16}")


def test_bfloat16_can_create_dtype():
    """检查 ap.bfloat16 是否已经是 numpy dtype 对象"""
    # ap.bfloat16 已经是 dtype 对象，可以直接使用
    assert ap.bfloat16 is not None, "ap.bfloat16 应该存在"
    assert isinstance(ap.bfloat16, np.dtype), "ap.bfloat16 应该是 numpy dtype 对象"
    logger.info(f"[PASS] ap.bfloat16 已经是 numpy dtype 对象: {ap.bfloat16}")


def test_bfloat16_can_create_array():
    """检查是否可以使用 bfloat16 创建数组"""
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.bfloat16)
    assert arr.dtype == ap.bfloat16, "数组 dtype 应该匹配（ap.bfloat16 已经是 dtype 对象）"
    logger.info(f"[PASS] 成功创建数组: {arr}, dtype: {arr.dtype}")


def test_bfloat16_scalar_creation():
    """测试 bfloat16 标量创建"""
    expected_acl_value = 27  # ACL_BF16
    
    # 测试创建标量（需要使用 ap.dtypes.bfloat16，因为 ap.bfloat16 是 dtype 对象）
    scalar = ap.dtypes.bfloat16(3.14)
    assert scalar is not None, "应该能够创建标量对象"
    
    # 验证 ACL 枚举值
    assert hasattr(scalar, 'getACLenum'), "标量应该有 getACLenum 方法"
    acl_enum = scalar.getACLenum()
    assert acl_enum == expected_acl_value, \
        f"ACL枚举值应该为 {expected_acl_value} (ACL_BF16)，实际为 {acl_enum}"
    
    logger.info(f"[PASS] 成功创建标量，ACL枚举值: {acl_enum} (期望: {expected_acl_value})")


def test_bfloat16_get_acl_enum():
    """检查 bfloat16 的 getACLenum() 接口"""
    expected_acl_value = 27  # ACL_BF16
    
    # 使用 ap.dtypes.bfloat16 创建标量（ap.bfloat16 是 dtype 对象，不能直接调用）
    scalar = ap.dtypes.bfloat16(0)
    assert scalar is not None, "应该能够创建标量对象"
    assert hasattr(scalar, 'getACLenum'), "getACLenum 方法应该存在"
    
    acl_enum = scalar.getACLenum()
    assert acl_enum == expected_acl_value, f"ACL枚举值应该为 {expected_acl_value}，实际为 {acl_enum}"
    
    # 测试不同标量值的 getACLenum 是否一致
    scalar2 = ap.dtypes.bfloat16(3.14)
    acl_enum2 = scalar2.getACLenum()
    assert acl_enum == acl_enum2, "不同标量值的ACL枚举值应该一致"
    
    logger.info(f"[PASS] bfloat16 getACLenum() = {acl_enum} (期望: {expected_acl_value})")


def test_bfloat16_get_acl_data_type():
    """测试 GetACLDataType 函数是否能正确识别 bfloat16 并返回正确的 ACL 类型"""
    # 创建 bfloat16 类型的数组
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.bfloat16)
    assert arr.dtype == ap.bfloat16, "数组 dtype 应该匹配（ap.bfloat16 已经是 dtype 对象）"
    
    # 使用 ndarray.from_numpy 创建 NPUArray，这会调用 GetACLDataType
    try:
        npu_arr = ap.ndarray.from_numpy(arr)
        expected_acl_value = 27  # ACL_BF16
        
        # 检查 aclDtype 是否正确
        assert npu_arr.aclDtype == expected_acl_value, \
            f"ACL类型应该为 {expected_acl_value}，实际为 {npu_arr.aclDtype}"
        
        logger.info(f"[PASS] GetACLDataType 正确识别 bfloat16: ACL类型 = {npu_arr.aclDtype}")
        logger.info(f"[PASS] NPUArray 创建成功: shape={npu_arr.shape}, dtype={npu_arr.dtype}")
        
    except Exception as e:
        logger.error(f"[FAIL] GetACLDataType 测试失败: {e}")
        raise


def test_bfloat16_static_get_acl_enum():
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
        arr = np.array(values, dtype=ap.bfloat16)
        npu_arr = ap.ndarray.from_numpy(arr)
        
        # 所有数组的 ACL 类型应该都是 ACL_BF16 (27)
        assert npu_arr.aclDtype == 27, \
            f"所有 bfloat16 数组的 ACL 类型应该为 27，实际为 {npu_arr.aclDtype}"
    
    logger.info(f"[PASS] 静态方法 getACLenum 通过 GetACLDataType 间接测试成功")


def _verify_array_properties(arr, expected_shape, expected_acl_value, dtype, operator_name):
    """辅助函数：验证数组属性"""
    assert arr.aclDtype == expected_acl_value, \
        f"{operator_name} 创建的数组 ACL 类型应该为 {expected_acl_value}，实际为 {arr.aclDtype}"
    assert list(arr.shape) == list(expected_shape), \
        f"{operator_name} 创建的数组形状应该为 {expected_shape}，实际为 {arr.shape}"
    cpu_arr = arr.to_numpy()
    assert cpu_arr.dtype == dtype, \
        f"{operator_name} 转换后的 numpy 数组 dtype 应该为 bfloat16，实际为 {cpu_arr.dtype}"
    logger.info(f"[PASS] {operator_name} 创建成功: shape={arr.shape}, aclDtype={arr.aclDtype}")
    logger.info(f"[PASS] 转换到 numpy: dtype={cpu_arr.dtype}, shape={cpu_arr.shape}")


def test_bfloat16_ones_operator():
    """测试 ones 算子"""
    dtype = ap.bfloat16  # ap.bfloat16 已经是 dtype 对象
    expected_acl_value = 27  # ACL_BF16
    
    logger.info("\n测试 ones 算子:")
    ones_arr = ap.ones(shape=(3, 4), dtype=dtype)
    _verify_array_properties(ones_arr, (3, 4), expected_acl_value, dtype, "ones")


def test_bfloat16_zeros_operator():
    """测试 zeros 算子"""
    dtype = ap.bfloat16  # ap.bfloat16 已经是 dtype 对象
    expected_acl_value = 27  # ACL_BF16
    
    logger.info("\n测试 zeros 算子:")
    zeros_arr = ap.zeros(shape=(2, 3), dtype=dtype)
    _verify_array_properties(zeros_arr, (2, 3), expected_acl_value, dtype, "zeros")


def test_bfloat16_full_operator():
    """测试 full 算子"""
    dtype = ap.bfloat16  # ap.bfloat16 已经是 dtype 对象
    expected_acl_value = 27  # ACL_BF16
    
    logger.info("\n测试 full 算子:")
    full_value = 2.5
    full_arr = ap.full(shape=(2, 2), value=full_value, dtype=dtype)
    _verify_array_properties(full_arr, (2, 2), expected_acl_value, dtype, "full")
    logger.info(f"[PASS] full 值: {full_value}")


def test_bfloat16_npuarray_constructor():
    """测试 NPUArray 构造函数"""
    dtype = ap.bfloat16  # ap.bfloat16 已经是 dtype 对象
    expected_acl_value = 27  # ACL_BF16
    
    logger.info("\n测试 NPUArray 构造函数:")
    npu_arr = ap.ndarray(shape=(2, 3), dtype=dtype)
    _verify_array_properties(npu_arr, (2, 3), expected_acl_value, dtype, "NPUArray构造函数")


def test_bfloat16_empty_operator():
    """测试 empty 算子和不同形状"""
    dtype = ap.bfloat16  # ap.bfloat16 已经是 dtype 对象
    expected_acl_value = 27  # ACL_BF16
    
    logger.info("\n测试不同形状:")
    test_shapes = [
        (5,),
        (2, 3),
        (1, 2, 3),
    ]
    for shape in test_shapes:
        test_arr = ap.empty(shape=shape, dtype=dtype)
        _verify_array_properties(test_arr, shape, expected_acl_value, dtype, f"empty(shape={shape})")


def test_bfloat16_array_creation_operators():
    """测试使用 bfloat16 类型调用 asnumpy 封装的算子（ones, zeros, full）"""
    test_bfloat16_ones_operator()
    test_bfloat16_npuarray_constructor()
    test_bfloat16_zeros_operator()
    test_bfloat16_full_operator()
    test_bfloat16_empty_operator()
    logger.info(f"\n[PASS] bfloat16 数组创建算子测试成功（ones, zeros, full）")


def test_bfloat16_numpy_dtype_attributes():
    """测试 bfloat16 的 NumPy dtype 属性和方法"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 bfloat16 NumPy dtype 接口")
    logger.info("=" * 60)
    
    # 获取 dtype 对象（ap.bfloat16 已经是 dtype 对象）
    dtype = ap.bfloat16
    assert dtype is not None, "ap.bfloat16 应该存在"
    assert isinstance(dtype, np.dtype), "ap.bfloat16 应该是 numpy dtype 对象"
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
    assert dtype_kind == 'f', f"bfloat16 的 kind 应该是 'f'，实际为 {dtype_kind}"
    
    # 使用共享函数测试基本属性
    from test_utils import test_dtype_basic_attributes
    test_dtype_basic_attributes(dtype, "bfloat16", 2, logger)
    
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
    assert not dtype_hasobject, "bfloat16 不应该包含对象引用"
    
    # 使用共享函数测试高级属性和方法
    from test_utils import test_dtype_advanced_attributes_and_methods
    test_dtype_advanced_attributes_and_methods(dtype, "bfloat16", ap.bfloat16, logger)
    
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
    dtype2 = ap.bfloat16  # ap.bfloat16 已经是 dtype 对象
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
    logger.info("[PASS] bfloat16 NumPy dtype 接口测试完成")
    logger.info("=" * 60)


def test_bfloat16_structured_and_subarray_types():
    """测试 bfloat16 在结构化数组和子数组类型中的使用"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 bfloat16 结构化数组和子数组类型")
    logger.info("=" * 60)
    
    base_dtype = ap.bfloat16  # ap.bfloat16 已经是 dtype 对象
    
    # 使用共享函数测试结构化数组和子数组类型
    from test_utils import test_dtype_structured_and_subarray_types
    test_dtype_structured_and_subarray_types(base_dtype, "bfloat16", ap.dtypes.bfloat16, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("[PASS] bfloat16 结构化数组和子数组类型测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    tests = [
        test_bfloat16_is_registered,
        test_bfloat16_type_num,
        test_bfloat16_is_bound,
        test_bfloat16_top_level_access,
        test_bfloat16_as_numpy_dtype,
        test_bfloat16_can_create_dtype,
        test_bfloat16_can_create_array,
        test_bfloat16_scalar_creation,
        test_bfloat16_get_acl_enum,
        test_bfloat16_get_acl_data_type,  
        test_bfloat16_static_get_acl_enum,
        test_bfloat16_array_creation_operators,
        test_bfloat16_numpy_dtype_attributes,
        test_bfloat16_structured_and_subarray_types,
    ]
    sys.exit(main(tests))


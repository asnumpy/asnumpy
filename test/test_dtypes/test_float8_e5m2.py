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
import sys

import numpy as np

from test_utils import main
import asnumpy as ap

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


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


def test_float8_e5m2_can_create_dtype():
    """检查是否可以使用 float8_e5m2 创建 numpy dtype"""
    dtype = np.dtype(ap.dtypes.float8_e5m2)
    assert dtype is not None, "应该能够创建 numpy dtype"
    logger.info(f"[PASS] 成功创建 numpy dtype: {dtype}")


def test_float8_e5m2_can_create_array():
    """检查是否可以使用 float8_e5m2 创建数组"""
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.dtypes.float8_e5m2)
    assert arr.dtype == np.dtype(ap.dtypes.float8_e5m2), "数组 dtype 应该匹配"
    logger.info(f"[PASS] 成功创建数组: {arr}, dtype: {arr.dtype}")


def test_float8_e5m2_get_acl_enum():
    """检查 float8_e5m2 的 getACLenum() 接口"""
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    scalar = ap.dtypes.float8_e5m2(0)
    assert scalar is not None, "应该能够创建标量对象"
    assert hasattr(scalar, 'getACLenum'), "getACLenum 方法应该存在"
    
    acl_enum = scalar.getACLenum()
    assert acl_enum == expected_acl_value, f"ACL枚举值应该为 {expected_acl_value}，实际为 {acl_enum}"
    
    # 测试不同标量值的 getACLenum 是否一致
    scalar2 = ap.dtypes.float8_e5m2(3.14)
    acl_enum2 = scalar2.getACLenum()
    assert acl_enum == acl_enum2, "不同标量值的ACL枚举值应该一致"
    
    logger.info(f"[PASS] float8_e5m2 getACLenum() = {acl_enum} (期望: {expected_acl_value})")


def test_float8_e5m2_get_acl_data_type():
    """测试 GetACLDataType 函数是否能正确识别 float8_e5m2 并返回正确的 ACL 类型"""
    # 创建 float8_e5m2 类型的数组
    arr = np.array([1.0, 2.0, 3.14], dtype=ap.dtypes.float8_e5m2)
    assert arr.dtype == np.dtype(ap.dtypes.float8_e5m2), "数组 dtype 应该匹配"
    
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
        arr = np.array(values, dtype=ap.dtypes.float8_e5m2)
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
    dtype = np.dtype(ap.dtypes.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试 ones 算子:")
    ones_arr = ap.ones(shape=(3, 4), dtype=dtype)
    _verify_array_properties(ones_arr, (3, 4), expected_acl_value, dtype, "ones")


def test_float8_e5m2_zeros_operator():
    """测试 zeros 算子"""
    dtype = np.dtype(ap.dtypes.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试 zeros 算子:")
    zeros_arr = ap.zeros(shape=(2, 3), dtype=dtype)
    _verify_array_properties(zeros_arr, (2, 3), expected_acl_value, dtype, "zeros")


def test_float8_e5m2_full_operator():
    """测试 full 算子"""
    dtype = np.dtype(ap.dtypes.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试 full 算子:")
    full_value = 2.5
    full_arr = ap.full(shape=(2, 2), value=full_value, dtype=dtype)
    _verify_array_properties(full_arr, (2, 2), expected_acl_value, dtype, "full")
    logger.info(f"[PASS] full 值: {full_value}")


def test_float8_e5m2_npuarray_constructor():
    """测试 NPUArray 构造函数"""
    dtype = np.dtype(ap.dtypes.float8_e5m2)
    expected_acl_value = 35  # ACL_FLOAT8_E5M2
    
    logger.info("\n测试 NPUArray 构造函数:")
    npu_arr = ap.ndarray(shape=(2, 3), dtype=dtype)
    _verify_array_properties(npu_arr, (2, 3), expected_acl_value, dtype, "NPUArray构造函数")


def test_float8_e5m2_empty_operator():
    """测试 empty 算子和不同形状"""
    dtype = np.dtype(ap.dtypes.float8_e5m2)
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


if __name__ == "__main__":
    tests = [
        test_float8_e5m2_is_registered,
        test_float8_e5m2_type_num,
        test_float8_e5m2_is_bound,
        test_float8_e5m2_can_create_dtype,
        test_float8_e5m2_can_create_array,
        test_float8_e5m2_get_acl_enum,
        test_float8_e5m2_get_acl_data_type,  
        test_float8_e5m2_static_get_acl_enum,
        test_float8_e5m2_array_creation_operators,
    ]
    sys.exit(main(tests))


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

from loguru import logger
import numpy as np

import asnumpy as ap



def test_dtypes_is_submodule():
    import importlib
    import sys
    import types

    m = importlib.import_module("asnumpy.dtypes")
    assert isinstance(m, types.ModuleType)
    # asnumpy 命名空间中暴露了 dtypes
    assert getattr(ap, "dtypes", None) is m
    # sys.modules 中注册了完整模块名
    assert "asnumpy.dtypes" in sys.modules


def test_ap_float8_e5m2_is_top_level_alias():
    """测试 ap.float8_e5m2 顶层直出与 asnumpy.dtypes 一致"""
    import importlib

    assert hasattr(ap, "float8_e5m2"), "ap.float8_e5m2 未暴露（顶层直出失败）"
    dtypes_mod = importlib.import_module("asnumpy.dtypes")
    assert ap.float8_e5m2 is dtypes_mod.float8_e5m2
    logger.info("✓ ap.float8_e5m2 顶层直出与 asnumpy.dtypes.float8_e5m2 一致")


def test_float8_e5m2_basic():
    """测试 float8_e5m2 基本功能"""
    # 仅检查类型对象，并确认其可被 numpy.dtype 识别
    assert hasattr(ap, "float8_e5m2"), "float8_e5m2 类型对象未绑定到 ap"
    dt = np.dtype(ap.float8_e5m2)
    assert dt is not None, "np.dtype 未能识别 float8_e5m2 类型对象"
    logger.info("✓ float8_e5m2 类型对象已绑定且可被 numpy.dtype 识别")


def test_float8_e5m2_scalar():
    """测试 float8_e5m2 标量创建"""
    try:
        # 创建标量
        scalar = ap.float8_e5m2(3.14)
        logger.info("✓ 成功创建 float8_e5m2 标量: {}", scalar)

        # 检查类型（与 TypeDescriptor<T>::kQualifiedTypeName 一致）
        scalar_type = type(scalar)
        full_name = f"{scalar_type.__module__}.{scalar_type.__name__}"
        assert full_name == "asnumpy.dtypes.float8_e5m2", f"标量类型名不匹配: {full_name}"
        logger.info("✓ 标量类型正确: {}", full_name)

    except Exception as e:
        logger.error("✗ 标量创建失败: {}", e)
        raise


def test_float8_e5m2_array():
    """测试 float8_e5m2 数组操作"""
    try:
        # 创建 float32 数组
        float32_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        logger.info("原始 float32 数组: {}", float32_arr)

        # 转换为 float8_e5m2（直接使用类型对象作为 dtype）
        float8_arr = float32_arr.astype(ap.float8_e5m2)
        logger.info("✓ 成功转换为 float8_e5m2 数组: {}", float8_arr)
        logger.info("数组 dtype: {}", float8_arr.dtype)

        # 检查形状和大小
        assert float8_arr.shape == float32_arr.shape
        assert float8_arr.itemsize == 1  # float8 应该是 1 字节
        logger.info("✓ 数组形状: {}, 元素大小: {} 字节", float8_arr.shape, float8_arr.itemsize)

        # 转换回 float32 验证
        recovered = float8_arr.astype(np.float32)
        logger.info("转换回 float32: {}", recovered)

        # 检查精度损失在合理范围内
        max_diff = np.max(np.abs(float32_arr - recovered))
        logger.info("最大精度损失: {}", max_diff)
        assert max_diff < 1.0, f"精度损失过大: {max_diff}"

    except Exception as e:
        logger.error("✗ 数组操作失败: {}", e)
        raise


def test_numpy_float8_e5m2_alias():
    """测试使用 np.float8_e5m2 作为 dtype（要求别名存在）。"""
    values = [1.0, 2.0, 3.0, 4.0]
    try:
        if not hasattr(np, "float8_e5m2"):
            logger.warning("跳过：当前 NumPy 未暴露 np.float8_e5m2 别名")
            return
        float8_e5m2_arr = np.array(values, dtype=np.float8_e5m2)
        logger.info("✓ 使用 np.float8_e5m2 创建数组成功")
        # 校验 dtype 与我们注册的类型一致
        assert float8_e5m2_arr.dtype == np.dtype(ap.float8_e5m2)
        logger.info("数组: {}", float8_e5m2_arr)
    except Exception as e:
        logger.error("✗ 使用 np.float8_e5m2/asnumpy.dtypes.float8_e5m2 失败: {}", e)
        raise


def test_float8_e5m2_dtype_properties():
    """测试 float8_e5m2 dtype 属性"""
    # 通过 numpy.dtype 从类型对象获取 dtype
    dtype = np.dtype(ap.float8_e5m2)

    # 检查基本属性
    assert dtype.itemsize == 1, f"预期元素大小为 1，实际为 {dtype.itemsize}"
    assert dtype.kind == "f", f"预期种类为 'f'，实际为 '{dtype.kind}'"

    logger.info("✓ dtype 属性正确:")
    logger.info("  - 元素大小: {} 字节", dtype.itemsize)
    logger.info("  - 种类: '{}' (浮点)", dtype.kind)
    logger.info("  - 名称: {}", dtype.name)

    # 通过标量实例方法获取 ACL 枚举常量（由 C++ 侧 scalar_methods 注册）
    scalar = ap.float8_e5m2(1.0)
    assert hasattr(scalar, "getACLenum"), "缺少 getACLenum 方法"
    acl_enum = scalar.getACLenum()
    logger.info("  - ACL 枚举常量: {}", acl_enum)


def run_all_tests():
    """运行所有测试"""
    tests = [
        test_dtypes_is_submodule,
        test_ap_float8_e5m2_is_top_level_alias,
        test_float8_e5m2_basic,
        test_float8_e5m2_scalar,
        test_float8_e5m2_array,
        test_numpy_float8_e5m2_alias,
        test_float8_e5m2_dtype_properties,
    ]

    logger.info("运行 float8_e5m2 dtype 测试...")
    logger.info("=" * 50)

    for test in tests:
        try:
            logger.info("运行测试: {}", test.__name__)
            test()
            logger.info("✓ {} 通过", test.__name__)
        except Exception as e:
            logger.error("✗ {} 失败: {}", test.__name__, e)
            raise

    logger.info("=" * 50)
    logger.info("✓ 所有测试通过!")


if __name__ == "__main__":
    run_all_tests()

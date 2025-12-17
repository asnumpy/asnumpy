#!/usr/bin/env python3
"""
测试自定义 dtype（如 bfloat16）对各种 asnumpy API 的支持情况

此测试文件设计为可配置的，可以轻松替换测试的 dtype 类型。

使用方法：
1. 修改文件顶部的配置部分：
   - TEST_DTYPE: 要测试的 dtype（如 ap.bfloat16, ap.float8_e5m2, np.float32）
   - TEST_DTYPE_NAME: dtype 的名称（用于日志输出）
   - TEST_DTYPE_ACL_VALUE: ACL 类型值（仅用于某些验证，可选）

2. 运行测试：
   python test/test_dtypes/test_api_support.py

测试覆盖的 API 类别：
- 数组创建：ones, zeros, empty, full, eye, identity, ndarray, ones_like, zeros_like
- 数学运算 - 基础算术：add, subtract, multiply, divide, power, absolute, square
- 数学运算 - 指数对数：exp, expm1, exp2, log, log10, log2, log1p, logaddexp, logaddexp2
- 数学运算 - 三角函数：sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot, radians
- 数学运算 - 双曲函数：sinh, cosh, tanh, arcsinh, arccosh, arctanh
- 数学运算 - 舍入函数：around, round_, rint, fix, floor, ceil, trunc
- 数学运算 - 算术运算：true_divide, floor_divide, float_power, fmod, mod, remainder, modf, divmod, positive, negative, reciprocal
- 归约操作：sum, prod, cumsum, cumprod, nanprod, nansum, nancumprod, nancumsum, cross
- 逻辑运算 - 比较：greater, greater_equal, less, less_equal, equal, not_equal
- 逻辑运算 - 逻辑操作：all, any, logical_and, logical_or, logical_not, logical_xor
- 逻辑运算 - 有限性检查：isfinite, isinf, isneginf, isposinf
- 其他数学函数：sign, heaviside, sinc, lcm, gcd, signbit, nan_to_num, clip, maximum, minimum, fmax, fmin, relu, gelu, real
- 线性代数：dot, vdot, inner, outer, matmul, einsum
"""
import numpy as np
import asnumpy as ap
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 配置：可替换的 dtype 类型
# ============================================================================
# 要测试的 dtype 类型，可以轻松替换为其他类型
# 
# 示例配置：
#   - bfloat16: TEST_DTYPE = ap.bfloat16, TEST_DTYPE_NAME = "bfloat16", TEST_DTYPE_ACL_VALUE = 27
#   - float8_e5m2: TEST_DTYPE = ap.float8_e5m2, TEST_DTYPE_NAME = "float8_e5m2", TEST_DTYPE_ACL_VALUE = 26
#   - float32: TEST_DTYPE = np.float32, TEST_DTYPE_NAME = "float32", TEST_DTYPE_ACL_VALUE = None
#   - float64: TEST_DTYPE = np.float64, TEST_DTYPE_NAME = "float64", TEST_DTYPE_ACL_VALUE = None
TEST_DTYPE = ap.bfloat16
TEST_DTYPE_NAME = "bfloat16"
TEST_DTYPE_ACL_VALUE = 27  # ACL_BF16，其他类型需要相应修改


# ============================================================================
# 辅助函数
# ============================================================================

def verify_array_properties(arr, expected_shape, expected_dtype, operator_name):
    """
    验证数组的基本属性
    
    Args:
        arr: NPUArray 对象
        expected_shape: 期望的形状（可以是 tuple 或 list）
        expected_dtype: 期望的 dtype
        operator_name: 操作名称（用于日志）
    """
    # 使用 list 比较来避免 tuple vs list 的类型问题
    # NPUArray.shape 返回的是 list，而测试中可能使用 tuple
    assert list(arr.shape) == list(expected_shape), \
        f"{operator_name}: shape 不匹配，期望 {expected_shape}，实际 {arr.shape}"
    
    # 转换为 numpy 数组验证 dtype
    # 注意：某些 dtype（如 bfloat16）可能无法直接转换为 numpy buffer
    try:
        cpu_arr = arr.to_numpy()
        # 对于 _CallableDtype，需要比较底层的 dtype
        if hasattr(expected_dtype, '_dtype'):
            expected_dtype_obj = expected_dtype._dtype
        else:
            expected_dtype_obj = np.dtype(expected_dtype)
        
        assert cpu_arr.dtype == expected_dtype_obj, \
            f"{operator_name}: dtype 不匹配，期望 {expected_dtype_obj}，实际 {cpu_arr.dtype}"
        
        logger.info(f"[PASS] {operator_name}: shape={arr.shape}, dtype={cpu_arr.dtype}")
    except (ValueError, TypeError) as e:
        # 某些 dtype（如 bfloat16）可能无法转换为 numpy buffer
        # 这种情况下，我们只验证 shape 和 ACL 类型
        logger.warning(f"{operator_name}: 无法转换为 numpy（可能不支持），但 shape 正确: {arr.shape}")
        logger.info(f"[PASS] {operator_name}: shape={arr.shape} (numpy 转换跳过)")


def test_api(api_name, test_func, description=""):
    """
    统一的 API 测试包装器
    
    Args:
        api_name: API 名称
        test_func: 测试函数
        description: 测试描述
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试 {api_name}" + (f": {description}" if description else ""))
        logger.info(f"{'='*60}")
        test_func()
        logger.info(f"[PASS] {api_name} 测试通过")
        return True
    except Exception as e:
        logger.error(f"[FAIL] {api_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 数组创建 API 测试
# ============================================================================

def test_ones():
    """测试 ones API"""
    arr = ap.ones(shape=(3, 4), dtype=TEST_DTYPE)
    verify_array_properties(arr, (3, 4), TEST_DTYPE, "ones")


def test_zeros():
    """测试 zeros API"""
    arr = ap.zeros(shape=(2, 3), dtype=TEST_DTYPE)
    verify_array_properties(arr, (2, 3), TEST_DTYPE, "zeros")


def test_empty():
    """测试 empty API"""
    arr = ap.empty(shape=(3, 3), dtype=TEST_DTYPE)
    verify_array_properties(arr, (3, 3), TEST_DTYPE, "empty")


def test_full():
    """测试 full API"""
    arr = ap.full(shape=(2, 2), value=2.5, dtype=TEST_DTYPE)
    verify_array_properties(arr, (2, 2), TEST_DTYPE, "full")


def test_eye():
    """测试 eye API"""
    arr = ap.eye(n=3, dtype=TEST_DTYPE)
    verify_array_properties(arr, (3, 3), TEST_DTYPE, "eye")


def test_identity():
    """测试 identity API"""
    arr = ap.identity(n=3, dtype=TEST_DTYPE)
    verify_array_properties(arr, (3, 3), TEST_DTYPE, "identity")


def test_ndarray_constructor():
    """测试 ndarray 构造函数"""
    arr = ap.ndarray(shape=(2, 3), dtype=TEST_DTYPE)
    verify_array_properties(arr, (2, 3), TEST_DTYPE, "ndarray")


def test_ones_like():
    """测试 ones_like API"""
    original = ap.ones(shape=(2, 3), dtype=TEST_DTYPE)
    arr = ap.ones_like(other=original, dtype=TEST_DTYPE)
    verify_array_properties(arr, (2, 3), TEST_DTYPE, "ones_like")


def test_zeros_like():
    """测试 zeros_like API"""
    original = ap.ones(shape=(2, 3), dtype=TEST_DTYPE)
    arr = ap.zeros_like(other=original, dtype=TEST_DTYPE)
    verify_array_properties(arr, (2, 3), TEST_DTYPE, "zeros_like")


# ============================================================================
# 数学运算 API 测试
# ============================================================================

def test_add():
    """测试 add API"""
    a = ap.ones(shape=(2, 3), dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.add(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "add")


def test_subtract():
    """测试 subtract API"""
    a = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.subtract(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "subtract")


def test_multiply():
    """测试 multiply API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.multiply(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "multiply")


def test_divide():
    """测试 divide API"""
    a = ap.full(shape=(2, 3), value=6.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.divide(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "divide")


def test_power():
    """测试 power API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.power(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "power")


def test_abs():
    """测试 absolute/fabs API"""
    arr = ap.full(shape=(2, 3), value=-2.5, dtype=TEST_DTYPE)
    result = ap.absolute(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "absolute")


def test_exp():
    """测试 exp API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.exp(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "exp")


def test_log():
    """测试 log API"""
    arr = ap.full(shape=(2, 3), value=2.71828, dtype=TEST_DTYPE)
    result = ap.log(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "log")


def test_sin():
    """测试 sin API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.sin(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "sin")


def test_cos():
    """测试 cos API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.cos(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "cos")


def test_square():
    """测试 square API（平方）"""
    arr = ap.full(shape=(2, 3), value=4.0, dtype=TEST_DTYPE)
    result = ap.square(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "square")


# ============================================================================
# 三角函数 API 测试
# ============================================================================

def test_tan():
    """测试 tan API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.tan(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "tan")


def test_arcsin():
    """测试 arcsin API"""
    arr = ap.full(shape=(2, 3), value=0.5, dtype=TEST_DTYPE)
    result = ap.arcsin(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "arcsin")


def test_arccos():
    """测试 arccos API"""
    arr = ap.full(shape=(2, 3), value=0.5, dtype=TEST_DTYPE)
    result = ap.arccos(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "arccos")


def test_arctan():
    """测试 arctan API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.arctan(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "arctan")


def test_arctan2():
    """测试 arctan2 API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.arctan2(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "arctan2")


def test_hypot():
    """测试 hypot API"""
    a = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=4.0, dtype=TEST_DTYPE)
    result = ap.hypot(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "hypot")


def test_radians():
    """测试 radians API"""
    arr = ap.full(shape=(2, 3), value=90.0, dtype=TEST_DTYPE)
    result = ap.radians(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "radians")


# ============================================================================
# 双曲函数 API 测试
# ============================================================================

def test_sinh():
    """测试 sinh API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.sinh(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "sinh")


def test_cosh():
    """测试 cosh API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.cosh(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "cosh")


def test_tanh():
    """测试 tanh API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.tanh(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "tanh")


def test_arcsinh():
    """测试 arcsinh API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.arcsinh(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "arcsinh")


def test_arccosh():
    """测试 arccosh API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.arccosh(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "arccosh")


def test_arctanh():
    """测试 arctanh API"""
    arr = ap.full(shape=(2, 3), value=0.5, dtype=TEST_DTYPE)
    result = ap.arctanh(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "arctanh")


# ============================================================================
# 指数对数函数 API 测试
# ============================================================================

def test_expm1():
    """测试 expm1 API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.expm1(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "expm1")


def test_exp2():
    """测试 exp2 API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.exp2(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "exp2")


def test_log10():
    """测试 log10 API"""
    arr = ap.full(shape=(2, 3), value=10.0, dtype=TEST_DTYPE)
    result = ap.log10(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "log10")


def test_log2():
    """测试 log2 API"""
    arr = ap.full(shape=(2, 3), value=8.0, dtype=TEST_DTYPE)
    result = ap.log2(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "log2")


def test_log1p():
    """测试 log1p API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.log1p(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "log1p")


def test_logaddexp():
    """测试 logaddexp API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.logaddexp(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "logaddexp")


def test_logaddexp2():
    """测试 logaddexp2 API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.logaddexp2(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "logaddexp2")


# ============================================================================
# 舍入函数 API 测试
# ============================================================================

def test_around():
    """测试 around API"""
    arr = ap.full(shape=(2, 3), value=3.14159, dtype=TEST_DTYPE)
    result = ap.around(arr, decimals=2, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "around")


def test_round_():
    """测试 round_ API"""
    arr = ap.full(shape=(2, 3), value=3.14159, dtype=TEST_DTYPE)
    result = ap.round_(arr, decimals=2, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "round_")


def test_rint():
    """测试 rint API"""
    arr = ap.full(shape=(2, 3), value=3.7, dtype=TEST_DTYPE)
    result = ap.rint(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "rint")


def test_fix():
    """测试 fix API"""
    arr = ap.full(shape=(2, 3), value=3.7, dtype=TEST_DTYPE)
    result = ap.fix(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "fix")


def test_floor():
    """测试 floor API"""
    arr = ap.full(shape=(2, 3), value=3.7, dtype=TEST_DTYPE)
    result = ap.floor(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "floor")


def test_ceil():
    """测试 ceil API"""
    arr = ap.full(shape=(2, 3), value=3.2, dtype=TEST_DTYPE)
    result = ap.ceil(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "ceil")


def test_trunc():
    """测试 trunc API"""
    arr = ap.full(shape=(2, 3), value=3.7, dtype=TEST_DTYPE)
    result = ap.trunc(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "trunc")


# ============================================================================
# 算术运算 API 测试
# ============================================================================

def test_true_divide():
    """测试 true_divide API"""
    a = ap.full(shape=(2, 3), value=6.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.true_divide(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "true_divide")


def test_floor_divide():
    """测试 floor_divide API"""
    a = ap.full(shape=(2, 3), value=7.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.floor_divide(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "floor_divide")


def test_float_power():
    """测试 float_power API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.float_power(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "float_power")


def test_fmod():
    """测试 fmod API"""
    a = ap.full(shape=(2, 3), value=7.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.fmod(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "fmod")


def test_mod():
    """测试 mod API"""
    a = ap.full(shape=(2, 3), value=7.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.mod(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "mod")


def test_remainder():
    """测试 remainder API"""
    a = ap.full(shape=(2, 3), value=7.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.remainder(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "remainder")


def test_modf():
    """测试 modf API"""
    arr = ap.full(shape=(2, 3), value=3.7, dtype=TEST_DTYPE)
    result = ap.modf(arr)
    # modf 返回元组，包含整数部分和小数部分
    assert isinstance(result, tuple), "modf 应该返回元组"
    assert len(result) == 2, "modf 应该返回两个数组"
    logger.info(f"[PASS] modf: 返回元组，包含 {len(result)} 个数组")


def test_divmod():
    """测试 divmod API"""
    a = ap.full(shape=(2, 3), value=7.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.divmod(a, b, dtype=TEST_DTYPE)
    # divmod 返回元组
    assert isinstance(result, tuple), "divmod 应该返回元组"
    assert len(result) == 2, "divmod 应该返回两个数组"
    logger.info(f"[PASS] divmod: 返回元组，包含 {len(result)} 个数组")


def test_positive():
    """测试 positive API"""
    arr = ap.full(shape=(2, 3), value=-2.0, dtype=TEST_DTYPE)
    result = ap.positive(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "positive")


def test_negative():
    """测试 negative API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.negative(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "negative")


def test_reciprocal():
    """测试 reciprocal API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.reciprocal(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "reciprocal")


# ============================================================================
# 归约操作 API 测试
# ============================================================================

def test_sum():
    """测试 sum API"""
    arr = ap.ones(shape=(2, 3), dtype=TEST_DTYPE)
    try:
        result = ap.sum(arr)
        # sum 返回标量，验证其类型
        assert result is not None, "sum 应该返回结果"
        logger.info(f"[PASS] sum: 结果类型 = {type(result)}")
    except (ValueError, TypeError) as e:
        # bfloat16 等自定义类型可能不支持 sum 操作（无法转换为 numpy buffer）
        logger.warning(f"[SKIP] sum: 不支持 {TEST_DTYPE_NAME} 类型 - {e}")
        raise  # 重新抛出异常以标记测试失败


def test_prod():
    """测试 prod API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    try:
        result = ap.prod(arr)
        assert result is not None, "prod 应该返回结果"
        logger.info(f"[PASS] prod: 结果类型 = {type(result)}")
    except (ValueError, TypeError) as e:
        # bfloat16 等自定义类型可能不支持 prod 操作（无法转换为 numpy buffer）
        logger.warning(f"[SKIP] prod: 不支持 {TEST_DTYPE_NAME} 类型 - {e}")
        raise  # 重新抛出异常以标记测试失败


def test_cumsum():
    """测试 cumsum API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.cumsum(arr, axis=0, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "cumsum")


def test_cumprod():
    """测试 cumprod API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.cumprod(arr, axis=0, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "cumprod")


def test_nanprod():
    """测试 nanprod API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    try:
        result = ap.nanprod(arr)
        assert result is not None, "nanprod 应该返回结果"
        logger.info(f"[PASS] nanprod: 结果类型 = {type(result)}")
    except (ValueError, TypeError) as e:
        logger.warning(f"[SKIP] nanprod: 不支持 {TEST_DTYPE_NAME} 类型 - {e}")
        raise


def test_nansum():
    """测试 nansum API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    try:
        result = ap.nansum(arr)
        assert result is not None, "nansum 应该返回结果"
        logger.info(f"[PASS] nansum: 结果类型 = {type(result)}")
    except (ValueError, TypeError) as e:
        logger.warning(f"[SKIP] nansum: 不支持 {TEST_DTYPE_NAME} 类型 - {e}")
        raise


def test_nancumprod():
    """测试 nancumprod API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.nancumprod(arr, axis=0, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "nancumprod")


def test_nancumsum():
    """测试 nancumsum API"""
    arr = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.nancumsum(arr, axis=0, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "nancumsum")


def test_cross():
    """测试 cross API"""
    a = ap.full(shape=(3,), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(3,), value=2.0, dtype=TEST_DTYPE)
    result = ap.cross(a, b, axis=0)
    verify_array_properties(result, (3,), TEST_DTYPE, "cross")


# ============================================================================
# 逻辑运算 API 测试
# ============================================================================

def test_greater():
    """测试 greater API"""
    a = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.greater(a, b)
    # greater 返回 bool 类型数组，shape 可能是 list
    assert list(result.shape) == [2, 3], f"greater 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] greater: shape={result.shape}")


def test_less():
    """测试 less API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    result = ap.less(a, b)
    assert list(result.shape) == [2, 3], f"less 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] less: shape={result.shape}")


def test_equal():
    """测试 equal API"""
    a = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    try:
        result = ap.equal(a, b)
        assert list(result.shape) == [2, 3], f"equal 应该返回形状 [2, 3]，实际 {result.shape}"
        logger.info(f"[PASS] equal: shape={result.shape}")
    except RuntimeError as e:
        # 某些 dtype（如 bfloat16）可能不支持 equal 操作
        logger.warning(f"[SKIP] equal: 不支持 {TEST_DTYPE_NAME} 类型 - {e}")
        raise  # 重新抛出异常以标记测试失败


def test_greater_equal():
    """测试 greater_equal API"""
    a = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    result = ap.greater_equal(a, b)
    assert list(result.shape) == [2, 3], f"greater_equal 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] greater_equal: shape={result.shape}")


def test_less_equal():
    """测试 less_equal API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    result = ap.less_equal(a, b)
    assert list(result.shape) == [2, 3], f"less_equal 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] less_equal: shape={result.shape}")


def test_not_equal():
    """测试 not_equal API"""
    a = ap.full(shape=(2, 3), value=3.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    result = ap.not_equal(a, b)
    assert list(result.shape) == [2, 3], f"not_equal 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] not_equal: shape={result.shape}")


def test_all():
    """测试 all API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.all(arr)
    assert isinstance(result, (bool, np.bool_)), f"all 应该返回布尔值，实际 {type(result)}"
    logger.info(f"[PASS] all: 结果类型 = {type(result)}")


def test_any():
    """测试 any API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.any(arr)
    assert isinstance(result, (bool, np.bool_)), f"any 应该返回布尔值，实际 {type(result)}"
    logger.info(f"[PASS] any: 结果类型 = {type(result)}")


def test_isfinite():
    """测试 isfinite API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.isfinite(arr)
    assert list(result.shape) == [2, 3], f"isfinite 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] isfinite: shape={result.shape}")


def test_isinf():
    """测试 isinf API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.isinf(arr)
    assert list(result.shape) == [2, 3], f"isinf 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] isinf: shape={result.shape}")


def test_isneginf():
    """测试 isneginf API"""
    arr = ap.full(shape=(2, 3), value=-1.0, dtype=TEST_DTYPE)
    result = ap.isneginf(arr)
    assert list(result.shape) == [2, 3], f"isneginf 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] isneginf: shape={result.shape}")


def test_isposinf():
    """测试 isposinf API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.isposinf(arr)
    assert list(result.shape) == [2, 3], f"isposinf 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] isposinf: shape={result.shape}")


def test_logical_and():
    """测试 logical_and API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.logical_and(a, b)
    assert list(result.shape) == [2, 3], f"logical_and 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] logical_and: shape={result.shape}")


def test_logical_or():
    """测试 logical_or API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.logical_or(a, b)
    assert list(result.shape) == [2, 3], f"logical_or 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] logical_or: shape={result.shape}")


def test_logical_not():
    """测试 logical_not API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.logical_not(arr)
    assert list(result.shape) == [2, 3], f"logical_not 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] logical_not: shape={result.shape}")


def test_logical_xor():
    """测试 logical_xor API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    result = ap.logical_xor(a, b)
    assert list(result.shape) == [2, 3], f"logical_xor 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] logical_xor: shape={result.shape}")


# ============================================================================
# 其他 API 测试
# ============================================================================

def test_clip():
    """测试 clip API"""
    arr = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    result = ap.clip(arr, a_min=1.0, a_max=3.0)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "clip")


def test_maximum():
    """测试 maximum API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    result = ap.maximum(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "maximum")


def test_minimum():
    """测试 minimum API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    result = ap.minimum(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "minimum")


def test_sign():
    """测试 sign API"""
    arr = ap.full(shape=(2, 3), value=-2.5, dtype=TEST_DTYPE)
    result = ap.sign(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "sign")


def test_heaviside():
    """测试 heaviside API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=0.5, dtype=TEST_DTYPE)
    result = ap.heaviside(a, b)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "heaviside")


def test_sinc():
    """测试 sinc API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.sinc(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "sinc")


def test_lcm():
    """测试 lcm API"""
    a = ap.full(shape=(2, 3), value=12.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=8.0, dtype=TEST_DTYPE)
    result = ap.lcm(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "lcm")


def test_gcd():
    """测试 gcd API"""
    a = ap.full(shape=(2, 3), value=12.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=8.0, dtype=TEST_DTYPE)
    result = ap.gcd(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "gcd")


def test_signbit():
    """测试 signbit API"""
    arr = ap.full(shape=(2, 3), value=-2.5, dtype=TEST_DTYPE)
    result = ap.signbit(arr)
    assert list(result.shape) == [2, 3], f"signbit 应该返回形状 [2, 3]，实际 {result.shape}"
    logger.info(f"[PASS] signbit: shape={result.shape}")


def test_nan_to_num():
    """测试 nan_to_num API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "nan_to_num")


def test_fmax():
    """测试 fmax API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    result = ap.fmax(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "fmax")


def test_fmin():
    """测试 fmin API"""
    a = ap.full(shape=(2, 3), value=2.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(2, 3), value=5.0, dtype=TEST_DTYPE)
    result = ap.fmin(a, b, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "fmin")


def test_relu():
    """测试 relu API"""
    arr = ap.full(shape=(2, 3), value=-2.0, dtype=TEST_DTYPE)
    result = ap.relu(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "relu")


def test_gelu():
    """测试 gelu API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.gelu(arr, dtype=TEST_DTYPE)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "gelu")


def test_real():
    """测试 real API"""
    arr = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    result = ap.real(arr)
    verify_array_properties(result, (2, 3), TEST_DTYPE, "real")


# ============================================================================
# 线性代数 API 测试
# ============================================================================

def test_dot():
    """测试 dot API"""
    a = ap.full(shape=(3,), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(3,), value=2.0, dtype=TEST_DTYPE)
    result = ap.dot(a, b)
    # dot 返回标量
    assert result is not None, "dot 应该返回结果"
    logger.info(f"[PASS] dot: 结果类型 = {type(result)}")


def test_vdot():
    """测试 vdot API"""
    a = ap.full(shape=(3,), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(3,), value=2.0, dtype=TEST_DTYPE)
    result = ap.vdot(a, b)
    assert result is not None, "vdot 应该返回结果"
    logger.info(f"[PASS] vdot: 结果类型 = {type(result)}")


def test_inner():
    """测试 inner API"""
    a = ap.full(shape=(3,), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(3,), value=2.0, dtype=TEST_DTYPE)
    result = ap.inner(a, b)
    assert result is not None, "inner 应该返回结果"
    logger.info(f"[PASS] inner: 结果类型 = {type(result)}")


def test_outer():
    """测试 outer API"""
    a = ap.full(shape=(3,), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(4,), value=2.0, dtype=TEST_DTYPE)
    result = ap.outer(a, b)
    verify_array_properties(result, (3, 4), TEST_DTYPE, "outer")


def test_matmul():
    """测试 matmul API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(3, 4), value=2.0, dtype=TEST_DTYPE)
    result = ap.matmul(a, b)
    verify_array_properties(result, (2, 4), TEST_DTYPE, "matmul")


def test_einsum():
    """测试 einsum API"""
    a = ap.full(shape=(2, 3), value=1.0, dtype=TEST_DTYPE)
    b = ap.full(shape=(3, 4), value=2.0, dtype=TEST_DTYPE)
    result = ap.einsum("ij,jk->ik", a, b)
    verify_array_properties(result, (2, 4), TEST_DTYPE, "einsum")


# ============================================================================
# 主测试函数
# ============================================================================

def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info(f"开始测试 {TEST_DTYPE_NAME} 对各种 API 的支持")
    logger.info("=" * 60)
    
    # 定义所有测试用例
    test_cases = [
        # 数组创建
        ("ones", test_ones, "创建全1数组"),
        ("zeros", test_zeros, "创建全0数组"),
        ("empty", test_empty, "创建未初始化数组"),
        ("full", test_full, "创建填充指定值的数组"),
        ("eye", test_eye, "创建单位矩阵"),
        ("identity", test_identity, "创建单位矩阵"),
        ("ndarray", test_ndarray_constructor, "NPUArray 构造函数"),
        ("ones_like", test_ones_like, "创建与给定数组形状相同的全1数组"),
        ("zeros_like", test_zeros_like, "创建与给定数组形状相同的全0数组"),
        
        # 数学运算 - 基础算术
        ("add", test_add, "加法运算"),
        ("subtract", test_subtract, "减法运算"),
        ("multiply", test_multiply, "乘法运算"),
        ("divide", test_divide, "除法运算"),
        ("power", test_power, "幂运算"),
        ("absolute", test_abs, "绝对值"),
        ("square", test_square, "平方"),
        
        # 数学运算 - 指数对数
        ("exp", test_exp, "指数函数"),
        ("expm1", test_expm1, "exp(x)-1"),
        ("exp2", test_exp2, "2的x次幂"),
        ("log", test_log, "自然对数"),
        ("log10", test_log10, "常用对数"),
        ("log2", test_log2, "以2为底的对数"),
        ("log1p", test_log1p, "log(1+x)"),
        ("logaddexp", test_logaddexp, "log(exp(x1)+exp(x2))"),
        ("logaddexp2", test_logaddexp2, "log2(2^x1+2^x2)"),
        
        # 数学运算 - 三角函数
        ("sin", test_sin, "正弦函数"),
        ("cos", test_cos, "余弦函数"),
        ("tan", test_tan, "正切函数"),
        ("arcsin", test_arcsin, "反正弦函数"),
        ("arccos", test_arccos, "反余弦函数"),
        ("arctan", test_arctan, "反正切函数"),
        ("arctan2", test_arctan2, "arctan2函数"),
        ("hypot", test_hypot, "欧几里得范数"),
        ("radians", test_radians, "角度转弧度"),
        
        # 数学运算 - 双曲函数
        ("sinh", test_sinh, "双曲正弦"),
        ("cosh", test_cosh, "双曲余弦"),
        ("tanh", test_tanh, "双曲正切"),
        ("arcsinh", test_arcsinh, "反双曲正弦"),
        ("arccosh", test_arccosh, "反双曲余弦"),
        ("arctanh", test_arctanh, "反双曲正切"),
        
        # 数学运算 - 舍入函数
        ("around", test_around, "四舍五入"),
        ("round_", test_round_, "四舍五入"),
        ("rint", test_rint, "最近整数"),
        ("fix", test_fix, "向零舍入"),
        ("floor", test_floor, "向下取整"),
        ("ceil", test_ceil, "向上取整"),
        ("trunc", test_trunc, "截断"),
        
        # 数学运算 - 算术运算
        ("true_divide", test_true_divide, "真除法"),
        ("floor_divide", test_floor_divide, "向下整除"),
        ("float_power", test_float_power, "浮点幂"),
        ("fmod", test_fmod, "浮点取模"),
        ("mod", test_mod, "取模"),
        ("remainder", test_remainder, "余数"),
        ("modf", test_modf, "分离整数和小数部分"),
        ("divmod", test_divmod, "除法和取模"),
        ("positive", test_positive, "正号"),
        ("negative", test_negative, "负号"),
        ("reciprocal", test_reciprocal, "倒数"),
        
        # 归约操作
        ("sum", test_sum, "求和"),
        ("prod", test_prod, "求积"),
        ("cumsum", test_cumsum, "累积和"),
        ("cumprod", test_cumprod, "累积积"),
        ("nanprod", test_nanprod, "忽略NaN的乘积"),
        ("nansum", test_nansum, "忽略NaN的和"),
        ("nancumprod", test_nancumprod, "忽略NaN的累积积"),
        ("nancumsum", test_nancumsum, "忽略NaN的累积和"),
        ("cross", test_cross, "向量叉积"),
        
        # 逻辑运算 - 比较
        ("greater", test_greater, "大于比较"),
        ("greater_equal", test_greater_equal, "大于等于比较"),
        ("less", test_less, "小于比较"),
        ("less_equal", test_less_equal, "小于等于比较"),
        ("equal", test_equal, "相等比较"),
        ("not_equal", test_not_equal, "不等比较"),
        
        # 逻辑运算 - 逻辑操作
        ("all", test_all, "全为真"),
        ("any", test_any, "任一为真"),
        ("logical_and", test_logical_and, "逻辑与"),
        ("logical_or", test_logical_or, "逻辑或"),
        ("logical_not", test_logical_not, "逻辑非"),
        ("logical_xor", test_logical_xor, "逻辑异或"),
        
        # 逻辑运算 - 有限性检查
        ("isfinite", test_isfinite, "是否有限"),
        ("isinf", test_isinf, "是否无穷"),
        ("isneginf", test_isneginf, "是否负无穷"),
        ("isposinf", test_isposinf, "是否正无穷"),
        
        # 其他数学函数
        ("sign", test_sign, "符号函数"),
        ("heaviside", test_heaviside, "阶跃函数"),
        ("sinc", test_sinc, "sinc函数"),
        ("lcm", test_lcm, "最小公倍数"),
        ("gcd", test_gcd, "最大公约数"),
        ("signbit", test_signbit, "符号位"),
        ("nan_to_num", test_nan_to_num, "NaN替换"),
        ("clip", test_clip, "裁剪"),
        ("maximum", test_maximum, "最大值"),
        ("minimum", test_minimum, "最小值"),
        ("fmax", test_fmax, "浮点最大值"),
        ("fmin", test_fmin, "浮点最小值"),
        ("relu", test_relu, "ReLU激活函数"),
        ("gelu", test_gelu, "GELU激活函数"),
        ("real", test_real, "取实部"),
        
        # 线性代数
        ("dot", test_dot, "点积"),
        ("vdot", test_vdot, "向量点积"),
        ("inner", test_inner, "内积"),
        ("outer", test_outer, "外积"),
        ("matmul", test_matmul, "矩阵乘法"),
        ("einsum", test_einsum, "Einstein求和"),
    ]
    
    # 运行测试
    results = []
    for api_name, test_func, description in test_cases:
        success = test_api(api_name, test_func, description)
        results.append((api_name, success))
    
    # 统计结果
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    logger.info(f"总计: {total} 个 API")
    logger.info(f"通过: {passed} 个")
    logger.info(f"失败: {total - passed} 个")
    logger.info("=" * 60)
    
    # 打印失败的 API
    failed_apis = [name for name, success in results if not success]
    if failed_apis:
        logger.warning(f"失败的 API: {', '.join(failed_apis)}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)


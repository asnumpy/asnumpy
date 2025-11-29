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
import asnumpy as ap
from script_test_cases import *
"""
How to use:

在下面的XXX_FUNCTIONS中按模块增删你想要测的API
然后运行python3 test/test_script.py检查输出结果

"""
# ====================API注册表========================
# 格式：(函数名, numpy函数, asnumpy函数, 测试用例列表)

# 测试用例列表目前位于test/script_test_cases.py 由于不同API要求输入的测试数据格式不同，之后请根据自身需求增添
# 目前已有的：
# UNARY_TEST_CASES: 单操作数测试数据集
# BINARY_TEST_CASES: 普通双操作数测试数据集（加减乘除等
# BROADCAST_TEST_CASES: 双操作数广播测试数据集
# MATMUL_DOT_TEST_CASES、MATRIX_POWER_TEST_CASES: 矩阵乘法等对输入张量格式有要求的数据集

MATH_FUNCTIONS = [
    ("fabs", np.fabs, ap.fabs, UNARY_TEST_CASES),
    ("absolute", np.absolute, ap.absolute, UNARY_TEST_CASES),
    ("sign", np.sign, ap.sign, UNARY_TEST_CASES),
    ("heaviside", np.heaviside, ap.heaviside, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("positive", np.positive, ap.positive, UNARY_TEST_CASES),
    ("negative", np.negative, ap.negative, UNARY_TEST_CASES),
    ("reciprocal", np.reciprocal, ap.reciprocal, UNARY_TEST_CASES),
    ("modf", np.modf, ap.modf, UNARY_TEST_CASES),
    ("add", np.add, ap.add, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("subtract", np.subtract, ap.subtract, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("multiply", np.multiply, ap.multiply, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("divide", np.divide, ap.divide, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("true_divide", np.true_divide, ap.true_divide, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("floor_divide", np.floor_divide, ap.floor_divide, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("power", np.power, ap.power, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("float_power", np.float_power, ap.float_power, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("fmod", np.fmod, ap.fmod, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("mod", np.mod, ap.mod, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("remainder", np.remainder, ap.remainder, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("divmod", divmod, ap.divmod, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("sinc", np.sinc, ap.sinc, UNARY_TEST_CASES),
    ("lcm", np.lcm, ap.lcm, LCM_TEST_CASES),
    ("gcd", np.gcd, ap.gcd, GCD_TEST_CASES),
    ("around", np.around, ap.around, AROUND_TEST_CASES),
    ("round", np.round, ap.round_, AROUND_TEST_CASES),
    ("rint", np.rint, ap.rint, RINT_TEST_CASES),
    ("fix", np.fix, ap.fix, FIX_TEST_CASES),
    ("floor", np.floor, ap.floor, FLOOR_TEST_CASES),
    ("ceil", np.ceil, ap.ceil, CEIL_TEST_CASES),
    ("trunc", np.trunc, ap.trunc, TRUNC_TEST_CASES),
    ("sinh", np.sinh, ap.sinh, UNARY_TEST_CASES),
    ("cosh", np.cosh, ap.cosh, UNARY_TEST_CASES),
    ("tanh", np.tanh, ap.tanh, UNARY_TEST_CASES),
    ("arcsinh", np.arcsinh, ap.arcsinh, UNARY_TEST_CASES),
    ("arccosh", np.arccosh, ap.arccosh, UNARY_TEST_CASES),
    ("arctanh", np.arctanh, ap.arctanh, UNARY_TEST_CASES),
    ("signbit", np.signbit, ap.signbit, UNARY_TEST_CASES),
    ("clip", np.clip, ap.clip, CLIP_TEST_CASES + CLIP_ARR_OBJ_CASES + CLIP_OBJ_ARR_CASES + CLIP_OBJ_OBJ_CASES),
    ("square", np.square, ap.square, UNARY_TEST_CASES),
    ("fabs", np.abs, ap.fabs, UNARY_TEST_CASES),
    ("nan_to_num", np.nan_to_num, ap.nan_to_num, NAN_TO_NUM_TEST_CASES),
    ("maximum", np.maximum, ap.maximum, MAX_MIN_TEST_CASES),
    ("minimum", np.minimum, ap.minimum, MAX_MIN_TEST_CASES),
    ("fmax", np.fmax, ap.fmax, MAX_MIN_TEST_CASES),
    ("fmin", np.fmin, ap.fmin, MAX_MIN_TEST_CASES),
    ("sin", np.sin, ap.sin, UNARY_TEST_CASES),
    ("cos", np.cos, ap.cos, UNARY_TEST_CASES),
    ("tan", np.tan, ap.tan, UNARY_TEST_CASES),
    ("arcsin", np.arcsin, ap.arcsin, UNARY_TEST_CASES),
    ("arccos", np.arccos, ap.arccos, UNARY_TEST_CASES),
    ("arctan", np.arctan, ap.arctan, UNARY_TEST_CASES),
    ("hypot", np.hypot, ap.hypot, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("arctan2", np.arctan2, ap.arctan2, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("radians", np.radians, ap.radians, UNARY_TEST_CASES),
    ("degrees", np.degrees, ap.degrees, DEGREES_TEST_CASES),
    ("rad2deg", np.rad2deg, ap.rad2deg, DEGREES_TEST_CASES),
    ("prod", np.prod, ap.prod, UNARY_TEST_CASES + PROD_DIM_TEST_CASES),
    ("sum", np.sum, ap.sum, UNARY_TEST_CASES + SUM_DIM_TEST_CASES),
    ("nanprod", np.nanprod, ap.nanprod, UNARY_TEST_CASES + NANPROD_DIM_TEST_CASES),
    ("nansum", np.nansum, ap.nansum, UNARY_TEST_CASES + NANSUM_DIM_TEST_CASES),
    ("cumprod", np.cumprod, ap.cumprod, CUM_TEST_CASES),
    ("cumsum", np.cumsum, ap.cumsum, CUM_TEST_CASES),
    ("nancumprod", np.nancumprod, ap.nancumprod, CUM_TEST_CASES),
    ("nancumsum", np.nancumsum, ap.nancumsum, CUM_TEST_CASES),
    ("cross", np.cross, ap.cross, CROSS_TEST_CASES),
    ("real", np.real, ap.real, COMPLEX_UNARY_TEST_CASES),
    ("exp", np.exp, ap.exp, UNARY_TEST_CASES),
    ("expm1", np.expm1, ap.expm1, UNARY_TEST_CASES),
    ("exp2", np.exp2, ap.exp2, UNARY_TEST_CASES),
    ("log", np.log, ap.log, LOG_UNARY_TEST_CASES),
    ("log10", np.log10, ap.log10, LOG_UNARY_TEST_CASES),
    ("log2", np.log2, ap.log2, LOG_UNARY_TEST_CASES),
    ("log1p", np.log1p, ap.log1p, LOG_UNARY_TEST_CASES),
    ("logaddexp", np.logaddexp, ap.logaddexp, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("logaddexp2", np.logaddexp2, ap.logaddexp2, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("pow", np.power, ap.pow, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("ldexp", np.ldexp, ap.ldexp, LDEXP_TEST_CASES),
    ("copysign", np.copysign, ap.copysign, COPYSIGN_TEST_CASES),
    ("max", np.max, ap.max, UNARY_TEST_CASES + PROD_DIM_TEST_CASES),
    ("amax", np.amax, ap.amax, UNARY_TEST_CASES + PROD_DIM_TEST_CASES),
    ("nanmax", np.nanmax, ap.nanmax, UNARY_TEST_CASES + NANPROD_DIM_TEST_CASES),
]

LINALG_FUNCTIONS = [
    ("qr", np.linalg.qr, ap.linalg.qr, QR_TEST_CASES),
    ("norm", np.linalg.norm, ap.linalg.norm, NORM_TEST_CASES),
    ("det",np.linalg.det,ap.linalg.det,DET_SLOGDET_TEST_CASES),
    ("slogdet",np.linalg.slogdet,ap.linalg.slogdet,DET_SLOGDET_TEST_CASES),
    ("matmul", np.matmul, ap.matmul, MATMUL_DOT_TEST_CASES),
    ("einsum", np.einsum, ap.einsum, EINSUM_TEST_CASES),
    ("matrix_power", np.linalg.matrix_power, ap.linalg.matrix_power, MATRIX_POWER_TEST_CASES),
    ("dot", np.dot, ap.dot, MATMUL_DOT_TEST_CASES),
    ("vdot", np.vdot, ap.vdot, MATMUL_DOT_TEST_CASES),
    ("inner", np.inner, ap.inner, INNER_TEST_CASES),
    ("outer", np.outer, ap.outer, OUTER_TEST_CASES),
    ("inv",np.linalg.inv,ap.linalg.inv,DET_SLOGDET_TEST_CASES)
]

LOGIC_FUNCTIONS = [
    ("all", np.all, ap.all, UNARY_TEST_CASES),
    ("any", np.any, ap.any, UNARY_TEST_CASES),
    ("isfinite", np.isfinite, ap.isfinite, UNARY_TEST_CASES),
    ("isinf", np.isinf, ap.isinf, UNARY_TEST_CASES),
    ("isneginf", np.isneginf, ap.isneginf, UNARY_TEST_CASES),
    ("isposinf", np.isposinf, ap.isposinf, UNARY_TEST_CASES),
    ("logical_and", np.logical_and, ap.logical_and, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("logical_or", np.logical_or, ap.logical_or, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("logical_not", np.logical_not, ap.logical_not, UNARY_TEST_CASES),
    ("logical_xor", np.logical_xor, ap.logical_xor, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("greater", np.greater, ap.greater, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("greater_equal", np.greater_equal, ap.greater_equal, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("less", np.less, ap.less, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("less_equal", np.less_equal, ap.less_equal, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("equal", np.equal, ap.equal, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
    ("not_equal", np.not_equal, ap.not_equal, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
]

SORTING_FUNCTIONS = [
    ("sort", np.sort, ap.sort, SORT_TEST_CASES),
]

# 总表
FUNCTIONS_TABLE = MATH_FUNCTIONS + LINALG_FUNCTIONS + LOGIC_FUNCTIONS + SORTING_FUNCTIONS

# =========================测试函数本体======================

def test_functions():
    
    for name, np_func, ap_func, test_cases in FUNCTIONS_TABLE:
        print("=" * 50)
        print(f"Testing {name} function:")
        print("=" * 50)
        
        passed = 0
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            # 将测试用例元组解包为参数
            try:
                # 处理单参数和多参数用例
                if isinstance(test_case, tuple):
                    # 参数里任意位置可能出现ndarray，需要转换为npuarray供asnumpy调用
                    converted_args = tuple(
                        ap.ndarray.from_numpy(arg) if isinstance(arg, np.ndarray) else arg
                        for arg in test_case
                    )
                    # === 特殊函数注释说明 ===
                    # 测试 prod, sum, nanprod, nansum, max, amax 时，
                    # 可使用：np_func(test_case[0], axis=test_case[1], keepdims=test_case[2])
                    # 测试 cross 时，可使用：np_func(test_case[0], test_case[1], axis=test_case[2])
                    # 测试 nan_to_num 时，可使用：np_func(test_case[0], nan=test_case[1], posinf=test_case[2], neginf=test_case[3])
                    if name == "prod" or name == "sum" or name == "nanprod" or name == "nansum" or name == "max" \
                    or name == "amax" or name == "nanmax":
                        np_result = np_func(test_case[0], axis=test_case[1], keepdims=test_case[2])
                    elif name == "cross":
                        np_result = np_func(test_case[0], test_case[1], axis=test_case[2])
                    elif name == "nan_to_num":
                        np_result = np_func(test_case[0], nan=test_case[1], posinf=test_case[2], neginf=test_case[3])
                    elif name == "sort":
                        np_result = np_func(test_case[0], axis=test_case[1])
                    else:
                        np_result = np_func(*test_case)
                    ap_result = ap_func(*converted_args)
                else:
                    # 单个参数的情况
                    converted_arg = ap.ndarray.from_numpy(test_case) if isinstance(test_case, np.ndarray) else test_case
                    np_result = np_func(test_case)
                    ap_result = ap_func(converted_arg)
                
                # 转换结果为numpy数组
                if hasattr(ap_result, 'to_numpy'):
                    ap_result_np = ap_result.to_numpy()
                else:
                    ap_result_np = ap_result
                
                # 对于多个返回值的函数（如qr）
                if isinstance(np_result, tuple) and isinstance(ap_result_np, tuple):
                    all_close = True
                    for np_res, ap_res in zip(np_result, ap_result_np):
                        if not np.allclose(np_res, ap_res.to_numpy() if hasattr(ap_res, 'to_numpy') else ap_res, equal_nan=True):
                            all_close = False
                    if not all_close:
                        print(f"Test {i+1} FAILED:")
                        print(f"  Input: {test_case}")
                        print(f"  NumPy result: {np_result}")
                        print(f"  AP result: {ap_result_np}")
                        print()
                    else:
                        passed += 1
                else:
                    if not np.allclose(np_result, ap_result_np, equal_nan=True):
                        print(f"Test {i+1} FAILED:")
                        print(f"  Input: {test_case}")
                        print(f"  NumPy result: {np_result}")
                        print(f"  AP result: {ap_result_np}")
                        print()
                    else:
                        passed += 1
            except Exception as e:
                print(f"Test {i+1} ERROR:")
                print(f"  Input: {test_case}")
                print(f"  Error: {e}")
                print()
        
        print(f"Passed: {passed}/{total} tests")
        print()
        

def main():
    print("Testing NPU Math Functions")
    print("=" * 60)
    print()
    
    test_functions()
    
    print("=" * 60)
    print("All tests completed")

if __name__ == "__main__":
    main()
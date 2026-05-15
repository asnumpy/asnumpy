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

"""算子注册与封装系统

提供装饰器 @register_op 自动处理 dtype 转换、日志记录和错误处理，
大幅减少 Python 算子包装层的样板代码。

设计原则:
- KISS: 装饰器仅做三件事 — dtype 转换、日志、异常处理
- DRY:  将重复的 `@logger.catch` + `_convert_dtype` + `ndarray(...)` 模式抽象为装饰器
- YAGNI: 不引入算子图、延迟执行等复杂概念；仅做声明式包装

使用示例:
    from asnumpy._registry import register_op
    from ._core.math import sin as _sin

    @register_op("sin", module="math")
    def sin(x, dtype=None):
        return _sin(x, _convert_dtype(dtype))
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from .utils import ndarray

# 算子注册表（运行时查询）
_OP_REGISTRY: dict[str, dict[str, Any]] = {}


def register_op(
    name: str | None = None,
    *,
    module: str = "core",
    category: str = "general",
    doc: str | None = None,
):
    """注册 NPU 算子装饰器。

    自动为算子包装函数注入:
    - 结构化日志 (logger.catch + 自动 debug 日志)
    - 统一的异常处理
    - 运行时算子注册（供 list_ops / get_op_info 查询）

    注意: dtype 转换和 ndarray 包装由算子函数自身处理，
    装饰器保持轻量，遵循 KISS 原则。

    Args:
        name: 算子名称（默认使用函数名）
        module: 所属模块 (math, linalg, logic, statistics, random, nn)
        category: 算子分类 (arithmetic, trigonometric, reduction, ...)
        doc: 可选的额外文档说明

    Returns:
        装饰后的函数

    Examples:
        @register_op("sin", module="math", category="trigonometric")
        def sin(x, dtype=None):
            return ndarray(_core.sin(x, _convert_dtype(dtype)))

        @register_op("mean_scalar", module="statistics")
        def mean_all(a):
            return _core.mean(a)
    """
    def decorator(func: Callable) -> Callable:
        op_name = name if name is not None else func.__name__

        @functools.wraps(func)
        @logger.catch(reraise=True)
        def wrapper(*args, **kwargs):
            logger.debug(
                "[{module}.{op}] called with args={args}, kwargs={kwargs}",
                module=module,
                op=op_name,
                args=_safe_repr(args),
                kwargs=_safe_repr(kwargs),
            )
            result = func(*args, **kwargs)
            logger.debug("[{module}.{op}] completed", module=module, op=op_name)
            return result

        # 保留元数据供运行时查询
        wrapper._op_name = op_name
        wrapper._op_module = module
        wrapper._op_category = category
        wrapper._op_doc = doc

        _OP_REGISTRY[op_name] = {
            "module": module,
            "category": category,
            "doc": doc or inspect.getdoc(func) or "",
            "callable": func,
        }
        return wrapper

    return decorator


def wrap_result(result, dtype=None) -> ndarray:
    """将 C++ 返回的 ndarray 包装为 Python ndarray。

    对于返回标量的函数不应调用此函数。
    """
    if isinstance(result, ndarray):
        return result
    return ndarray(result)


def list_ops(module: str | None = None, category: str | None = None) -> list[dict]:
    """列出已注册的算子。

    Args:
        module: 按模块过滤 (None 表示全部)
        category: 按分类过滤 (None 表示全部)

    Returns:
        算子信息列表
    """
    results = []
    for op_name, info in _OP_REGISTRY.items():
        if module is not None and info["module"] != module:
            continue
        if category is not None and info["category"] != category:
            continue
        results.append({"name": op_name, **info})
    return results


def get_op_info(name: str) -> dict | None:
    """获取指定算子的注册信息。"""
    return _OP_REGISTRY.get(name)


def _safe_repr(obj) -> str:
    """安全地表示对象，避免大数组打印。"""
    if isinstance(obj, (np.ndarray, ndarray)):
        shape = getattr(obj, "shape", "?")
        dtype = getattr(obj, "dtype", "?")
        return f"<array shape={shape} dtype={dtype}>"
    if isinstance(obj, tuple):
        return f"({', '.join(_safe_repr(x) for x in obj)},)"
    if isinstance(obj, list):
        return f"[{', '.join(_safe_repr(x) for x in obj)}]"
    s = repr(obj)
    if len(s) > 200:
        return s[:200] + "..."
    return s

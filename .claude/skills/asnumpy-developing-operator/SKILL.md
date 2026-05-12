---
name: asnumpy-developing-operator
description: 当需要为 asnumpy 项目添加新的算子或函数时使用。触发词包括"添加 xx 算子"、"实现 xx 函数"、"新增 xx 运算"、"开发 xx"，或涉及跨 asnumpy C++/pybind11/Python/测试层的工作。
---

# AsNumpy 算子开发指南

## 概述

AsNumpy 是基于华为昇腾 NPU 的 NumPy 兼容计算库，采用三层架构：
Python 前端 (`asnumpy/*.py`) → pybind11 绑定层 (`python/bind_*.cpp`) → C++ CANN 后端 (`src/` + `include/`)。

新增算子遵循严格的 **7 步流水线**。本指南为每步提供模板与检查清单。

## 第 0 步：CANN API 调研（写代码前必做）

**在实现 C++ 代码之前，必须先查询 CANN 官方文档确认 ACLNN 算子是否存在。**

算子可能：
- CANN 已内置 → 直接封装
- OpenBOAT 提供 → 引入 OpenBOAT
- CANN 仅有基础算子 → 组合实现（如 `sinc(x) = sin(pi*x)/(pi*x)` at x≠0）

### 查询策略

1. **访问算子库首页**（替换版本号为最新）：
   - `https://www.hiascend.com/document/detail/zh/canncommercial/{version}/API/aolapi/operatorlist_00001.html`
   - 版本号映射：`900` = CANN 9.0.0，`850` = CANN 8.5.0，`800` = CANN 8.0.0
   - 先试 `900`，失败则回退到 `850`，再 `800`

2. **搜索引擎检索具体算子**：
   ```
   aclnn{算子名} site:hiascend.com
   ```

3. **查询结果分类处理**：
   - **找到内置 aclnnXxx**：记录头文件名（`aclnnop/aclnn_xxx.h`）、函数签名（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）、支持的 dtype 列表
   - **未找到**：判断 OpenBOAT 是否提供，或可否用已有算子组合实现

### CANN 关键头文件

| 分类 | 头文件 | 链接库 |
|------|--------|--------|
| 数学 | `aclnnop/aclnn_ops_math.h` | `libopapi_math.so` |
| 神经网络 | `aclnnop/aclnn_ops_nn.h` | `libopapi_nn.so` |
| 单个算子 | `aclnnop/aclnn_sin.h` 等 | — |
| 公共 | `aclnn/aclnn_base.h`、`aclnn/acl_meta.h` | `libnnopbase.so` |

## 算子分类决策

```
dot digraph op_flow {
    "新算子" [shape=doublecircle];
    "几个输入？" [shape=diamond];
    "有可选dtype参数？" [shape=diamond];
    "使用 EXECUTE_UNARY_OP 宏" [shape=box];
    "使用 EXECUTE_BINARY_OP 宏" [shape=box];
    "手动5步流程" [shape=box];
    "需要广播？" [shape=diamond];
    "使用 ExecuteBinaryOp 模板" [shape=box];
    "使用 ExecuteUnaryOp 模板" [shape=box];

    "新算子" -> "几个输入？";
    "几个输入？" -> "有可选dtype参数？" [label="1个"];
    "几个输入？" -> "需要广播？" [label="2个"];
    "几个输入？" -> "手动5步流程" [label="3个以上/复杂"];
    "有可选dtype参数？" -> "使用 EXECUTE_UNARY_OP 宏" [label="是"];
    "有可选dtype参数？" -> "使用 ExecuteUnaryOp 模板" [label="否"];
    "需要广播？" -> "使用 EXECUTE_BINARY_OP 宏" [label="是"];
    "需要广播？" -> "使用 ExecuteBinaryOp 模板" [label="否"];
}
```

## 第 1 步：C++ 头文件声明

**位置**：`include/asnumpy/<模块名>/<分组名>.hpp`

模板：
```cpp
/**
 * @brief [一行描述，与 NumPy 文档一致]。
 *
 * [可选：描述数学公式或边界行为。]
 *
 * @param x 输入数组 [形状要求、dtype 约束]。
 * @param dtype 可选输出 dtype，不指定则使用输入 dtype。
 * @return NPUArray [输出描述]。
 * @throws std::runtime_error ACL 算子执行或内存分配失败时抛出。
 */
NPUArray FuncName(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);
```

检查清单：
- [ ] Apache 2.0 许可证头块（从现有文件复制）
- [ ] 头文件保护（`#pragma once` 或 header guard，与目标文件保持一致）
- [ ] 函数名与 NumPy 对应（C++ 用 PascalCase，Python 用 snake_case）
- [ ] 可选 dtype 参数用 `std::optional<py::dtype>`
- [ ] 路径符合 `include/asnumpy/<模块>/<分组>.hpp` 规范

## 第 2 步：C++ 实现

**位置**：`src/<模块名>/<分组名>.cpp`

### 模式 A：标准一元算子（最常见，默认选择）

```cpp
NPUArray FuncName(const NPUArray& x, std::optional<py::dtype> dtype) {
    aclDataType aclType = ACL_DOUBLE;  // 默认回退类型
    if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE ||
        x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128) {
        aclType = x.aclDtype;
    }
    ACL_DTYPE_WARN(x.aclDtype, aclType, __func__);
    py::dtype out_dtype = dtype.has_value() ? *dtype : NPUArray::GetPyDtype(aclType);
    return EXECUTE_UNARY_OP(
        x,
        out_dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* wsSize, aclOpExecutor** exec) {
            return aclnnXxxGetWorkspaceSize(in, out, wsSize, exec);
        },
        [](void* ws, uint64_t wsSize, aclOpExecutor* exec, void* stream) {
            return aclnnXxx(ws, wsSize, exec, nullptr);
        },
        "OpName",
        "aclnnXxx"
    );
}
```

### 模式 B：标准二元算子

与模式 A 结构相同，但有两个输入，输出 shape 通过 `GetBroadcastShape(x1, x2)` 计算，使用 `EXECUTE_BINARY_OP` 宏。

### 模式 C：手动 5 步流程（用于复杂算子或无需 dtype 参数的算子）

```cpp
NPUArray OpName(const NPUArray& x) {
    auto out = NPUArray(x.shape, x.dtype);

    // 1. 获取 workspace 大小和执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnXxxGetWorkspaceSize(x.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnXxxGetWorkspaceSize");

    // 2. RAII 管理 workspace
    asnumpy::AclWorkspace workspace(workspaceSize);

    // 3. 执行算子
    error = aclnnXxx(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnXxx");

    // 4. 同步设备
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclnnXxx: aclrtSynchronizeDevice");

    // 5. workspace 由 RAII 自动释放
    return out;
}
```

### 必需的 include

```cpp
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/acl_executor.hpp>   // EXECUTE_UNARY_OP, EXECUTE_BINARY_OP
#include <asnumpy/utils/status_handler.hpp> // ACLNN_CHECK, ACL_RT_CHECK, LOG_*
#include <aclnnop/aclnn_xxx.h>              // 具体算子头文件
```

### Dtype 处理规则

- **浮点算子**（sin、cos、exp）：支持 `ACL_FLOAT, ACL_FLOAT16, ACL_DOUBLE`，默认 fallback 为 `ACL_DOUBLE`
- **整数算子**（add、multiply）：接受所有类型，默认输出 = 输入 dtype
- **布尔算子**（logical_*）：支持 `ACL_BOOL`
- 发生自动转换时必须调用 `ACL_DTYPE_WARN(input.aclDtype, resolvedType, __func__)`

检查清单：
- [ ] 许可证头
- [ ] 正确的 `aclType` 回退逻辑
- [ ] 使用 `AclWorkspace` RAII（无需手动 free）
- [ ] 三个 ACL 调用均检查错误：GetWorkspaceSize, Execute, Synchronize

## 第 3 步：Pybind11 绑定

### 3a. 在分组绑定函数中添加

**位置**：`python/bind_<模块>.cpp`

```cpp
// 在已有的 namespace asnumpy { ... } 块内：
void bind_<分组>(py::module_& <模块>) {
    <模块>.def("funcname", &FuncName, py::arg("x"), py::arg("dtype") = py::none());
}
```

重载函数绑定：
```cpp
<模块>.def("funcname",
    py::overload_cast<const NPUArray&, const NPUArray&, std::optional<py::dtype>>(&FuncName),
    py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
```

### 3b. 在 bind_<模块>.cpp 调度函数中注册

声明并调用：
```cpp
void bind_<分组>(py::module_& <模块>);  // 前向声明
// ...
void bind_<模块>(py::module_& <模块>) {
    bind_<分组>(<模块>);  // 调用
}
```

### 3c. 在 asnumpy.cpp 中注册（仅新建模块时需要）

**位置**：`python/asnumpy.cpp`

仅在创建全新模块时需要，向已有 math/logic 等模块追加算子不需此步。

检查清单：
- [ ] 绑定函数先声明后使用
- [ ] 所有可选 dtype 参数带 `py::arg("dtype") = py::none()`
- [ ] Python 名称用 `snake_case`，C++ 名称用 `PascalCase`

## 第 4 步：Python 包装层

**位置**：`asnumpy/<模块>.py`

### 4a. 导入 C++ 函数

```python
from .lib.asnumpy_core.<模块> import (
    funcname as _funcname,
)
from .utils import ndarray, _convert_dtype
```

### 4b. 编写包装函数

```python
def funcname(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_funcname(x, _convert_dtype(dtype)))
```

多返回值函数示例：
```python
def modf(x: ArrayLike) -> tuple:
    frac, inte = _modf(x)
    return [ndarray(frac), ndarray(inte)]
```

### 命名约定

| C++ 函数 | 绑定名称 | Python 包装 |
|----------|---------|-----------|
| `Sin` | `"sin"` | `sin()` |
| `Arctan2` | `"arctan2"` | `arctan2()` |
| `Round_` | `"round_"` | `round_()` |

检查清单：
- [ ] 所有参数和返回值有类型注解
- [ ] 所有 dtype 参数调用 `_convert_dtype(dtype)`
- [ ] 返回值用 `ndarray()` 包裹（多返回值用列表）
- [ ] 函数 docstring 可选（NumPy 文档是参考标准）

## 第 5 步：导出到主命名空间

**位置**：`asnumpy/__init__.py`

需要修改两处：

### 5a. TYPE_CHECKING 导入块（约第 25 行）
```python
if TYPE_CHECKING:
    from .<模块> import (
        # ... 已有导入 ...
        funcname,
    )
```

### 5b. _LAZY_MAPPING 字典条目（约第 180 行）
```python
_LAZY_MAPPING = {
    # ... 已有条目 ...
    "funcname": ".<模块>",
}
```

### 5c. `__all__` 由 `_EAGER_EXPORTS + list(_LAZY_MAPPING.keys())` 自动生成，无需手动修改。

检查清单：
- [ ] `TYPE_CHECKING` 块内添加导入
- [ ] `_LAZY_MAPPING` 添加条目（字母顺序非必须但建议）

## 第 6 步：测试

**位置**：`tests/asnumpy_tests/<模块>_tests/test_<分组>.py`

### 测试模板

```python
import numpy
import pytest
from asnumpy import testing


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# 基础功能测试
@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_funcname_basic(xp, dtype):
    numpy.random.seed(42)
    data = numpy.random.uniform(low=-3.0, high=3.0, size=(10, 10)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.funcname(a)


# 边界值测试
@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_funcname_edge(xp, dtype):
    data = [0.0, 1.0, -1.0]
    a = _create_array(xp, data, dtype)
    return xp.funcname(a)
```

### 装饰器速查

| 装饰器 | 用途 |
|--------|------|
| `@testing.for_float_dtypes(no_float16=True)` | 参数化 float32、float64 |
| `@testing.for_int_dtypes()` | 参数化 int8–int64、uint8–uint32 |
| `@testing.for_all_dtypes()` | 所有 dtype（默认排除 float16/uint32/uint64） |
| `@testing.for_dtypes([numpy.float32, numpy.int32])` | 指定 dtype 列表 |
| `@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)` | 与 NumPy 结果做 allclose 比较 |
| `@testing.numpy_asnumpy_array_equal()` | 与 NumPy 结果做精确比较 |
| `@testing.for_orders(['C', 'F'])` | 参数化内存顺序 |

### 应覆盖的测试类别

1. **基础功能**：随机数据，验证与 NumPy 输出一致
2. **边界值**：零、负一、特殊值（inf、nan）
3. **Dtype 覆盖**：浮点类型（必须）、整数类型（如支持）、布尔（如适用）
4. **形状变体**：标量、1D、2D、3D+
5. **已知缺陷**：用 `@pytest.mark.xfail(reason="...")` 标注

### xfail 模式

```python
@pytest.mark.xfail(reason="Bug: aclnnXxx 不支持 Int32，asnumpy 缺少自动类型转换")
@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_funcname_int(xp, dtype):
    data = [1, 2, 3]
    a = _create_array(xp, data, dtype)
    return xp.funcname(a)
```

检查清单：
- [ ] 使用 `_create_array` 辅助函数（或复制到测试文件中）
- [ ] 浮点算子至少一个 `numpy_asnumpy_allclose` 测试
- [ ] 至少一个边界值测试
- [ ] 使用 `numpy.random.seed(42)` 保证可重复
- [ ] 已知不支持的类型用 `xfail` 标注

## 第 7 步：编译与验证

```bash
# 编译（项目根目录执行）
pip install -e .

# 运行新增测试
pytest tests/asnumpy_tests/<模块>_tests/test_<分组>.py -v

# 运行整个模块测试，检查是否有回归
pytest tests/asnumpy_tests/<模块>_tests/ -v
```

编译失败排查：
- 检查 C++ 源码是否包含了所有 ACLNN 头文件
- 检查 dtype 转换逻辑
- 确认绑定函数已声明并调用

## 文件级规范速查

| 关注点 | 规范 |
|--------|------|
| 许可证头 | 每个 C++/Python 文件必须有 Apache 2.0 块，与已有文件风格一致 |
| C++ 命名空间 | 所有算子函数放在 `asnumpy::` 下 |
| 日志 | 使用 `spdlog` + `LOG_DEBUG/INFO/WARN` 宏 |
| 错误信息格式 | `[文件名](函数名) API名 error = <错误码> - <详细信息>` |
| Python 导入 | C++ 绑定以 `_funcname` 形式导入，包装为类型注解函数 |
| Python 类型 | 使用 `_types.py` 的别名：`ArrayLike`、`DTypeLike`、`AxisOptional` |

## 常见错误

1. **忘记 `_convert_dtype()`** — Python 层 `dtype=None` 与 C++ `nullopt` 不是一回事；必须通过 `_convert_dtype(dtype)` 转换
2. **不用 `AclWorkspace` RAII** — 用手动 `aclrtMalloc`/`aclrtFree` 而非 `asnumpy::AclWorkspace`，异常路径上会泄漏内存
3. **aclType 回退逻辑错误** — 整数类型错误地回退到 `ACL_DOUBLE`；必须参照同文件已有模式
4. **绑定函数注册遗漏** — 添加了 `bind_xxx` 函数但忘记在 `bind_<模块>()` 中调用
5. **Python 名称与 C++ 绑定不一致** — C++ 绑定 `math.def("sin", ...)` 但 Python 尝试 `from .lib.asnumpy_core.math import sine`
6. **测试中误调 `.to_numpy()`** — 使用 `@testing.numpy_asnumpy_allclose` 的测试应直接返回 NPUArray，不要调用 `.to_numpy()`

---
name: asnumpy-development
description: Use when developing asnumpy APIs, operators, bindings, Python frontend, CANN backend, or tests. 当用户需要新增、修改、删除、排查 asnumpy 接口、后端实现、CANN/NPU 算子逻辑、pybind11 绑定、Python 前端或测试时使用。
---

# AsNumpy 开发

## 概述

使用本技能指导 asnumpy 的后端 CANN/C++、pybind11 绑定、Python 前端和测试开发。核心规则是：在改代码前先理解 NumPy 语义和 CANN 算子覆盖能力，再按可追踪的层级顺序实施变更。

本技能不覆盖纯文档任务。

## 何时使用

以下 asnumpy 任务应使用本技能：
- 新增、修改、删除或排查 API/算子。
- 面向 `NPUArray` 的 NumPy 兼容行为。
- CANN `aclnn*` / ACL 算子选择，或指定版本算子能力检查。
- C++ 后端、pybind11 绑定、Python 前端、导出入口或 pytest 测试。

如果只是修改 README 或普通文档，且不伴随 API 开发，不使用本技能。

## 入口分流

先从用户请求中推断以下信息。只有缺失信息会阻塞推进时才追问。

| 需要确认 | 示例 |
|---|---|
| 操作类型 | 新增、修改、删除、排错 |
| 目标对象 | NumPy 相似 API 名称、已有 asnumpy 符号、失败测试 |
| 涉及范围 | 后端、绑定、前端、测试或全流程 |
| CANN 版本 | 默认最新；用户指定版本时以指定版本为准 |
| 兼容基线 | 默认以 NumPy 行为为基线 |

如果范围不清楚，先按全流程分析，并说明实际受影响的层级。

## 项目结构速查

开发时优先按目标模块定位同名或相近文件，遵循现有分层：

| 层级 | 主要位置 | 关注点 |
|---|---|---|
| Python 前端 | `src/asnumpy/*.py`、`src/asnumpy/linalg/`、`src/asnumpy/random/` | 用户 API、参数归一化、`ndarray` 包装、导出入口 |
| pybind11 绑定 | `bindings/python/bind_*.cpp`、`bindings/python/module.cpp` | 子模块注册、函数名、重载、默认参数、异常边界 |
| C++ 声明 | `include/asnumpy/<module>/*.hpp` | 对外声明、命名、参数类型、模块边界 |
| C++ 后端 | `csrc/<module>/*.cpp` | CANN 调用、shape/dtype、workspace、同步、资源释放 |
| 公共工具 | `include/asnumpy/utils/`、`csrc/utils/` | `NPUArray`、广播、ACL 资源、状态检查、算子宏 |
| 测试 | `tests/asnumpy_tests/<module>_tests/`、`src/asnumpy/testing/` | NumPy/asnumpy 对照、dtype 参数化、xfail 限制 |

新增模块级 API 时，检查是否需要同步：模块 CMakeLists、绑定注册、`src/asnumpy/__init__.py`、模块 `__all__` 或导入入口。

## 项目实现模式

优先复用项目已有模式，不要另起一套抽象：

- 简单一元/二元 CANN 算子优先参考 `include/asnumpy/utils/npu_ops_macros.hpp`、`include/asnumpy/utils/acl_executor.hpp` 中的既有执行封装。
- 特殊 shape、返回多值、dtype 转换或多算子组合，参考 `csrc/linalg/`、`csrc/logic/`、`csrc/statistics/` 中的手写实现。
- CANN 调用通常遵循 `aclnnXxxGetWorkspaceSize(...)` → RAII workspace → `aclnnXxx(...)` → 同步 → `ACLNN_CHECK/ACL_RT_CHECK`。
- 后端日志和错误处理复用现有体系：优先用 `LOG_DEBUG`、`LOG_INFO`、`LOG_WARN`、`ACL_DTYPE_WARN`、`ACLNN_CHECK`、`ACL_RT_CHECK`；不要临时引入 `printf` 或新的日志机制。
- 新增 C++ 源文件、模块或绑定文件时，同步检查对应 `CMakeLists.txt`、模块 target、依赖链接和 `_core` 扩展源文件列表。
- 绑定层按模块放入 `bind_<module>.cpp`，使用显式 `py::arg(...)`；重载函数用 `py::overload_cast`；异常翻译保持集中在 `_core` 模块入口，不在各绑定文件分散处理。
- Python 前端通常从 `_core.<module>` 导入绑定函数，做参数归一化后返回 `ndarray(...)`；dtype、shape、size 等转换优先复用 `src/asnumpy/utils.py` 的既有工具。
- 顶层 `asnumpy.xxx` 入口依赖 `src/asnumpy/__init__.py` 的懒加载映射；新增或删除公开 API 时必须同步检查映射、`TYPE_CHECKING` 导入和模块导出。
- 测试优先使用 `asnumpy.testing` 的 NumPy/asnumpy 对照装饰器，例如 `numpy_asnumpy_allclose`、`numpy_asnumpy_array_equal` 和 dtype 参数化工具。
- 已知 CANN/asnumpy 差异不要静默跳过；用明确测试、限制说明或 `pytest.mark.xfail(..., strict=True)` 标注原因。

## 实现前必须完成的调研

对于 NumPy 相似 API，改代码前必须先产出以下分析：

1. **NumPy 语义**：签名、参数、默认值、广播、dtype 行为、shape 规则、标量处理、错误行为、边界场景和不支持特性。
2. **CANN 算子调研**：默认联网检索官方 CANN 最新文档；如果用户指定版本，则检索指定版本。检查 `aclnn*` 名称、签名、workspace/executor 流程、dtype 支持、shape/广播支持、内存/layout 约束、同步行为和版本说明。
3. **覆盖矩阵**：

| 需求点 | NumPy 行为 | CANN 支持情况 | asnumpy 决策 |
|---|---|---|---|
| 参数或边界场景 | 期望行为 | 支持/部分支持/缺失 | 实现、限制、fallback 或拒绝 |

必须明确差异：不支持的 dtype、缺失的 `out`、部分广播、标量限制、layout 约束、精度差异，或需要 Python 侧校验的行为。

## 默认全流程顺序

除非任务明显是局部修改，否则按以下顺序推进：

1. **后端**：新增/修改/删除 C++ 声明和实现；封装 CANN 调用；处理 workspace、executor、stream、错误、dtype/shape 检查和资源释放。
2. **绑定**：暴露或移除 pybind11 函数；确保 Python 可见类型、所有权、异常和名称与后端匹配。
3. **前端**：新增/修改/删除 Python wrapper、参数归一化、`NPUArray` 分发、NumPy 兼容错误和包导出。
4. **测试**：新增或更新 pytest，覆盖 NumPy 对照行为，包括 dtype、shape、广播、标量、边界和错误场景。
5. **验证**：先运行聚焦测试，再运行更广的相关测试。如果测试失败，定位责任层并修复该层，不要在其他层掩盖问题。

删除操作要反向检查每一层：前端导出、wrapper、绑定、后端实现、测试、示例、生成符号列表和所有引用。

## 排错流程

1. 用最小相关测试命令复现失败。
2. 适用时用 NumPy 对照期望行为。
3. 定位失败层级：
   - 用户可见签名或参数校验错误：前端。
   - 类型转换、名称、所有权或异常边界错误：绑定。
   - 结果错误、CANN 报错、dtype/shape 处理、stream 或资源问题：后端。
   - 期望错误或缺少边界用例：测试。
4. 只修复责任层。
5. 重新运行聚焦失败测试和相关回归测试。

## 测试要求

行为变更优先测试先行。测试应尽量使用真实 asnumpy 行为，并在可行时用 NumPy 作为 oracle。

覆盖范围：
- 与 NumPy 匹配的正常场景。
- CANN/asnumpy 支持的 dtype 和 shape 组合。
- 相关时覆盖广播、标量输入、axis/keepdims、空数组、0 维数组和特殊值。
- 错误场景和不支持行为。
- 删除场景：必要时验证不可导入、不可导出且没有陈旧引用。

测试约束：
- `xfail` 应用于已知差异的最小用例，使用 `strict=True` 并写清 `[FIXABLE]` 或 `[UPSTREAM]` 原因。
- 不要随意新增未注册 pytest marker；项目启用 strict markers。
- 随机分布类测试通常验证 shape、dtype、参数和边界，不强行与 NumPy 逐元素对照。
- 运行测试依赖已构建扩展、CANN 环境和可用 Ascend NPU；硬件或版本限制要在结论中说明。

不要为了通过测试而削弱断言。如果 CANN 无法匹配 NumPy，应把刻意选择的 asnumpy 行为写入分析和测试。

## 快速参考

| 任务 | 必做事项 |
|---|---|
| 新增 API | NumPy 分析 → CANN 文档 → 覆盖矩阵 → 后端 → 绑定 → 前端 → 测试 |
| 修改 API | 识别行为变化 → 重新评估 CANN 覆盖 → 更新受影响层 → 回归测试 |
| 删除 API | 反向依赖搜索 → 移除所有层级痕迹 → 验证无陈旧导出/引用 |
| 修复失败 | 复现 → 对照 NumPy → 定位层级 → 修复责任层 → 重跑测试 |
| CANN 不确定 | 重新查官方最新或指定版本文档；不要依赖记忆 |

## 常见错误

| 错误 | 正确做法 |
|---|---|
| 未确认 CANN 覆盖能力就从 Python wrapper 开始 | 先调研 NumPy 和 CANN，再决定实现范围 |
| 凭记忆假设 `aclnn*` 名称或签名 | 检索官方最新或指定版本 CANN 文档 |
| 只匹配 happy path | 检查 dtype、shape、标量、广播和错误行为 |
| 在前端隐藏后端错误 | 诊断后修复责任层 |
| 只删除 Python 符号 | 移除或说明每个前端、绑定、后端和测试引用 |
| 把 CANN 不支持行为当成偶然问题 | 在分析和测试中明确 asnumpy 决策 |

## 红旗信号

如果准备做以下事情，立刻停止并重新检查流程：
- NumPy 相似 API 还没完成 NumPy/CANN 覆盖分析就写代码。
- 没查当前文档就使用记忆中的 CANN 签名。
- 因为变更“看起来只是后端”而跳过绑定或测试。
- 未对照 NumPy 或已声明的 asnumpy 限制，就修改测试适配当前行为。
- 删除 API 后仍在其他层保留导出或绑定。

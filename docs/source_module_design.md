# AsNumpy SourceModule 实现方案

> 对照 PyCUDA `SourceModule`，在 AsNumpy 中实现 Ascend NPU 自定义算子 JIT 编译与启动模块。

## 参考资料

- PyCUDA 官方文档：https://documen.tician.de/pycuda/
- Ascend C JIT 实践：https://ai6s.net/692ba5a0791c233193d12515.html
- [Ascend C 算子开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
- [Kernel Launch API 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/appdevgapi/aclcppdevg_03_1792.html)
- [异构编译文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/opdevg/BishengCompiler/atlas_bisheng_10_0011.html)

---

## 1. 对标分析

PyCUDA `SourceModule` 的核心工作流：

```
CUDA C 源码字符串 → nvcc 编译 → .cubin → cuModuleLoad → get_function → kernel(grid, block, args) → GPU
```

关键设计：
- **JIT 编译**: Python 中直接写 CUDA C 字符串，运行时调用 nvcc 编译
- **缓存**: 基于源码哈希，避免重复编译
- **Kernel 启动**: `get_function(name)` → 可调用对象，接受 grid/block/shared 参数
- **辅助工具**: `DynamicSourceModule`（模板变量）、`PreparedKernel`（性能计时）

构造函数签名：

```python
class SourceModule:
    def __init__(self, source, nvcc="nvcc", options=None, keep=False,
                 no_extern_c=False, arch=None, code=None,
                 cache_dir=None, include_dirs=[])
```

---

## 2. 核心决策

| 问题 | 结论 |
|------|------|
| **编译器** | `bisheng`（毕昇），路径 `${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin/bisheng`。可脱离 CMake 独立编译单 `.cpp` → `.o`，**但不自动生成 `.json`** |
| **运行时加载** | `aclrtBinaryLoadFromFile` + `aclrtBinaryGetFunction` + `aclrtLaunchKernelWithConfig`。**不需要 `.json`，不需要安装到 `opp/vendors/`** |
| **grid/block 映射** | `blockIdx`→`AscendC::GetBlockIdx()`，`gridDim`→`AscendC::GetBlockNum()`。**`threadIdx` 无等价物**——Ascend C 是 SPMD 模型，核内用向量指令而非线程 |
| **编译器路径稳定性** | `compiler/ccec_compiler/bin/bisheng` 在 CANN 7.0~8.5 中不变，由 `ASCEND_TOOLKIT_HOME` 环境变量定位 |

**编译命令**（已验证通过）:

```bash
ASCENDC_INC=${ASCEND_TOOLKIT_HOME}/aarch64-linux/ascendc/include

bisheng --cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore \
        --cce-aicore-lang --cce-aicore-arch=da-vinci \
        --std=c++17 -O2 -fPIC -shared \
        -I${ASCENDC_INC} \
        -I${ASCENDC_INC}/basic_api \
        -I${ASCENDC_INC}/highlevel_api \
        -I${ASCENDC_INC}/basic_api/impl \
        -I${ASCENDC_INC}/basic_api/inner_interface \
        -I${ASCENDC_INC}/basic_api/interface \
        kernel.cpp -o kernel.o
```

> **注**: `-shared` 必须保留——Ascend C kernel 引用了 `rtLaunch`/`rtFunctionRegister` 等运行时符号，去掉后 linker 会尝试生成可执行文件（要求 `main`）。`--cce-aicore-arch=da-vinci` 是本次为 CANN 8.2 兼容性新增的，待硬件验证。

**Kernel 加载与启动 API 序列**:

```cpp
aclrtBinaryLoadOptions *options = nullptr;  // NULL = 默认选项
aclrtBinaryLoadFromFile(binPath, options, &binHandle)       // 加载 .o
aclrtBinaryGetFunction(binHandle, kernelName, &func)        // 按名称获取核函数句柄
aclrtKernelArgsInit(func, &argsHandle)                      // 初始化参数句柄
aclrtKernelArgsAppend(argsHandle, ptr, size, ...)           // 逐个追加参数
aclrtLaunchKernelWithConfig(func, blockDim, stream, cfg,    // 启动
                             argsHandle, reserve)
// cfg 可为 NULL（默认配置），reserve 传 NULL
// 或使用简化版: aclrtLaunchKernel(func, blockDim, stream, argsHandle)
```

> **注**: `aclrtBinaryLoadOptions` 通常传 NULL 即可使用默认加载选项。
> `aclrtGetFunctionAddr(func, &aicAddr, &aivAddr)` 可用于获取 AI Core / AI Vector 入口地址供 `aclrtBinaryGetFunctionByEntry` 使用。

---

## 3. 实现流程

```
用户 Python: SourceModule(ascend_c_source, options)
    ↓
1. SHA256(source + options + arch + compiler_version) → 查缓存
    ↓ (miss)
2. 写临时文件 → bisheng 编译 → .o
    ↓
3. aclrtBinaryLoadFromFile 加载 .o
    ↓
4. aclrtBinaryGetFunction 获取 func handle（按 kernel 函数名）
    ↓
5. 构建 KernelFunction 对象 → 暴露 get_function(name)
    ↓ (user calls kernel)
6. aclrtKernelArgsInit/Append → aclrtLaunchKernelWithConfig → NPU
```

---

## 4. API 设计

```python
# --- 编译阶段 ---

class SourceModule:
    def __init__(
        self,
        source: str,                           # Ascend C kernel 源码
        options: list[str] | None = None,      # bisheng 编译选项，如 ["-O3"]
        include_dirs: list[str] | None = None, # 用户附加 -I 路径；Ascend C 系统路径自动推导
        cache_dir: str | None = None,          # None → ~/.asnumpy/cache/
        disable_cache: bool = False,
        keep: bool = False,                    # 保留中间产物（.cpp/.o/compile.log）
        verbose: bool = False,
        soc_version: str = "Ascend910B1",       # 芯片型号，默认 910B1
        core_type: str = "VecCore",            # VecCore / CubeCore / AICore
        compiler: str | None = None,           # bisheng 路径；None → 自动从 ASCEND_TOOLKIT_HOME 推导
    ):
        ...

    def get_function(self, name: str, signature: list[str] | None = None) -> KernelFunction:
        """获取 kernel 函数。signature 为 None 时自动从源码解析。"""
    def list_functions(self) -> list[str]: ...

# --- 启动阶段 ---

class KernelFunction:
    def __call__(
        self,
        *args,                                  # ndarray | ScalarLike，按位置对应 kernel 形参
        grid: tuple[int, ...] | None = None,    # AI Core 数量，如 (8,) 表示 8 个核
        block_shape: tuple[int, ...] | None = None, # 仅文档兼容，SPMD 无线程概念
        stream: object | None = None,           # 流对象，None = 默认流（同步执行）
    ) -> None: ...                              # kernel 无返回值，结果通过 __gm__ 输出参数写出

    def prepare(self) -> PreparedKernel: ...

class PreparedKernel:
    def __call__(self, *args, stream=None) -> None: ...
    @property
    def time(self) -> float: ...               # 上次执行耗时 (ms)
```

### 函数签名处理

**问题**: bisheng 编译的 `.o` 文件不含 DWARF 调试信息，`nm`/`readelf` 只能看到函数名 `T test_kernel`，无法获取参数类型。`aclrtBinaryGetFunction` 只返回函数句柄，不提供签名信息。因此参数类型必须从源码获取。

**方案: 从源码自动解析 + 运行时类型推断**

```
kernel 源码 → 正则匹配 extern "C" __global__ 行 → 解析形参列表
                                                  ↓
                              __gm__ float* a  →  NPUArray (dtype=float32)
                              __gm__ half*  b  →  NPUArray (dtype=float16)
                              int n            →  Python int → kernel int32
                              float s          →  Python float → kernel float32
                              int64_t n        →  Python int → kernel int64
                                                  ↓
                                          构建 ArgSpec 列表
```

**类型映射表**:

| Ascend C kernel 形参类型 | Python 传入类型 | `aclrtKernelArgsAppend` 传递方式 |
|--------------------------|----------------|-------------------------------|
| `__gm__ float*` / `__gm__ half*` / `__gm__ int*` 等 | `ndarray` | `&device_ptr` (8 字节 NPU 地址) |
| `int` / `int32_t` | Python `int` → 自动转为 `numpy.int32` | `&value` (4 字节) |
| `int64_t` / `long long` | Python `int` → 自动转为 `numpy.int64` | `&value` (8 字节) |
| `float` | Python `float` → 自动转为 `numpy.float32` | `&value` (4 字节) |
| `double` | Python `float` → 自动转为 `numpy.float64` | `&value` (8 字节) |
| `bool` | Python `bool` | `&value` (1 字节) |

**解析逻辑** (`source_module.py` 中的 `_parse_kernel_signature`):

```python
import re

_SIGNATURE_RE = re.compile(
    r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+'
    r'(\w+)\s*\(([^)]*)\)'
)

_GM_PTR_RE = re.compile(r'__gm__\s+(\w+)\s*\*\s*\w+')  # __gm__ float* name
_SCALAR_RE = re.compile(r'\b(int|int32_t|int64_t|float|double|bool|half|short)\s+\w+')

def _parse_kernel_signature(source: str, kernel_name: str) -> list[ArgSpec]:
    """从源码提取 kernel 函数的参数类型列表"""
    for match in _SIGNATURE_RE.finditer(source):
        if match.group(1) == kernel_name:
            params_str = match.group(2)
            return [_parse_param(p.strip()) for p in params_str.split(',') if p.strip()]
    raise ValueError(f"Kernel '{kernel_name}' not found in source")
```

**手动指定签名**（解析失败时的回退）:

```python
# get_function 支持显式指定签名
vec_add = mod.get_function("vector_add",
    signature=["float32*", "float32*", "float32*", "int32"])
```

### 使用示例（更新）

```python
# kernel 调用时，标量自动转换为正确类型
vec_add(a_npu, b_npu, c_npu, 1024, grid=(8,))
#                                 ^
# Python int 1024 根据源码解析结果自动转换为 np.int32

# 显式指定类型以避免歧义
vec_add(a_npu, b_npu, c_npu, np.int64(1024), grid=(8,))
```

### 使用示例

```python
import numpy as np
import asnumpy as ap
from asnumpy.compiler import SourceModule

# Ascend C kernel（SPMD 模型，无 threadIdx）
kernel_source = r"""
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void vector_add(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, int n)
{
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (n + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < n - start) ? per_block : (n - start);

    // CopyIn → Compute → CopyOut 范式
    LocalTensor<float> local_a = UB_ALLOC(float, count);
    LocalTensor<float> local_b = UB_ALLOC(float, count);
    DataCopy(local_a, a + start, count);
    DataCopy(local_b, b + start, count);
    VecAdd(local_a, local_a, local_b, count);
    DataCopy(c + start, local_a, count);
    UB_FREE(local_a);
    UB_FREE(local_b);
}
"""

mod = SourceModule(kernel_source, options=["-O3"])
vec_add = mod.get_function("vector_add")

a_npu = ap.ndarray.from_numpy(np.random.randn(1024).astype(np.float32))
b_npu = ap.ndarray.from_numpy(np.random.randn(1024).astype(np.float32))
c_npu = ap.empty((1024,), dtype=ap.float32)

vec_add(a_npu, b_npu, c_npu, 1024, grid=(8,))  # 8 个 AI Core
# kernel 无返回值，c_npu 被原地修改

result = c_npu.to_numpy()

# 性能测试
prepared = vec_add.prepare()
for _ in range(100):
    prepared(a_npu, b_npu, c_npu, 1024)
print(f"{prepared.time:.3f} ms")
```

---

## 5. 模块结构

```
asnumpy/compiler/                   # Python 前端 (新增)
├── __init__.py
├── source_module.py                # SourceModule：编排编译→加载→缓存主流程
├── bisheng_compiler.py             # subprocess 调用 bisheng，组装 include 路径
├── kernel_function.py              # KernelFunction + PreparedKernel（调用 C++ 绑定）
└── cache.py                        # SHA256 缓存

src/compiler/                       # C++ 后端 (新增)
├── CMakeLists.txt
└── kernel_launcher.cpp             # aclrtBinaryLoadFromFile → GetFunction → Launch

include/asnumpy/compiler/           # 头文件 (新增)
└── kernel_launcher.hpp

python/
└── bind_compiler.cpp               # pybind11 绑定 (新增)

tests/asnumpy_tests/compiler_tests/  # 测试 (新增)
└── test_source_module.py
```

**集成点**:
- `asnumpy/__init__.py`: `_LAZY_MAPPING["compiler"] = ".compiler"`
- `src/cann/`: 已有 device 管理（`set_device`、`init`、`finalize`）
- `src/utils/`: 已有 `NPUArray`（kernel 输入输出载体）

**分层原则**:
- **Python 层**: bisheng 编译（subprocess）、缓存管理（文件 I/O）、用户 API（SourceModule/KernelFunction）
- **C++ 层（pybind11）**: 仅 `kernel_launcher`——调用 ACL C API 加载 .o 和启动 kernel
- **理由**: subprocess、文件 I/O、SHA256 是 Python 标准库原生能力，放在 C++ 做只会增加不必要的绑定代码和编译复杂度

---

## 6. 缓存设计

```
~/.asnumpy/cache/
└── {sha256_hash}/
    ├── source.cpp
    ├── kernel.o
    ├── meta.json       # {"kernel_names": [...], "soc_version": "...", "compiler_version": "...", ...}
    └── compile.log
```

**缓存键** = SHA256(source + sorted(options) + soc_version + include_dirs + compiler_version)

> 单个 `.o` 文件可包含多个 `extern "C"` kernel 函数，`list_functions()` 通过 `nm` 或 `objdump -t` 解析符号表获取。用户 kernel 函数必须声明为 `extern "C"`，否则 `aclrtBinaryGetFunction` 无法按名称查找。

---

## 7. 环境验证记录（CANN 8.2.RC1.alpha003）

### 已验证通过的项

| 验证项 | 状态 | 说明 |
|--------|------|------|
| bisheng 编译器 | ✅ | clang 15.0.5，路径 `${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin/bisheng` |
| `--cce-soc-version` | ✅ | 可用值: `Ascend910B1`/`B2`/`B3`/`B4` |
| `--cce-soc-core-type` | ✅ | 可用值: `VecCore`/`CubeCore`/`AICore` |
| standalone kernel 编译 | ✅ | 需 6 个 Ascend C include 路径，成功生成 37KB ELF .o 文件 |
| `aclrtBinaryLoadFromFile` | ✅ | `acl_rt.h:2289`，需 `aclrtBinaryLoadOptions*`（可 NULL） |
| `aclrtBinaryGetFunction` | ✅ | `acl_rt.h:2141`，按 kernel 函数名获取句柄 |
| `aclrtBinaryGetFunctionByEntry` | ✅ | `acl_rt.h:2301`，按入口地址获取句柄 |
| `aclrtLaunchKernel` | ✅ | `acl_rt.h:2156`，简化版 4 参数启动 |
| `aclrtLaunchKernelWithConfig` | ✅ | `acl_rt.h:2458`，6 参数版，支持 `aclrtLaunchKernelCfg*` |
| `aclrtKernelArgsInit` | ✅ | `acl_rt.h:2380` |
| `aclrtKernelArgsAppend` | ✅ | `acl_rt.h:2405` |
| `kernel_operator.h` | ✅ | 位于 `aarch64-linux/ascendc/include/basic_api/` |

### 实现注意事项

1. **include 路径区分**: kernel 编译（bisheng）和 host 代码编译（ACL API）使用不同的 include 路径。Kernel 用 `ascendc/include/*`，host 用 `aarch64-linux/include/acl/`。

2. **soc_version 自动检测**: 运行时可通过 `npu-smi info` 或读取 `/usr/local/Ascend/ascend-toolkit/latest/platform.ini` 中的 `soc_version` 字段自动获取。默认应尝试 `Ascend910B1`。

3. **Kernel 源码模板**: SourceModule 应在用户提供的源码前自动插入 `using namespace AscendC;`，或在文档中明确要求。这避免了用户因命名空间问题导致编译失败。

4. **`aclrtLaunchKernelCfg`**: 用于配置 kernel 启动行为（如指定 engine type 为 AIC 还是 AIV），默认传 NULL 使用默认配置即可。

5. **性能计时**: `PreparedKernel.time` 使用 `aclrtCreateEvent` + `aclrtRecordEvent` + `aclrtSynchronizeEvent` + `aclrtEventElapsedTime` 实现，与 CUDA event 计时语义一致。

6. **清理**: `aclrtBinaryUnLoad` + `aclrtDestroyBinary` 在 `SourceModule.__del__` 中调用，确保二进制资源在模块销毁时释放。注意 `__del__` 在 atexit 时可能因 CANN 已 finalize 而失败——应在 `SourceModule` 中提供显式 `close()` 方法，并注册 `atexit` 回调在 `finalize()` 之前执行。

7. **`extern "C"` 要求**: `aclrtBinaryGetFunction` 按符号名查找，用户 kernel 函数必须声明为 `extern "C"` 以避免 C++ name mangling。SourceModule 可在编译前检测源码中是否缺少 `extern "C"` 并给出警告。

8. **`list_functions()` 实现**: 对编译后的 `.o` 文件执行 `objdump -t` 或 `nm` 解析符号表，筛选 `__aicore__` 入口函数。此操作在 Python 层完成，无需 C++ 介入。

9. **错误处理**: bisheng 编译失败时应捕获 stderr，抛出包含完整编译器输出的 `CompileError` 异常。ACL API 调用失败时应将 `aclGetRecentErrMsg()` 内容包含在异常消息中。

10. **线程安全**: `SourceModule` 的编译和缓存读写不是线程安全的——需在 `bisheng_compiler.py` 和 `cache.py` 中使用文件锁（`fcntl.flock` 或 `filelock` 包）防止并发编译同一源码。

---

## 8. 已知限制

### `aclrtBinaryLoadFromFile` 需要 CANN 8.5+

`aclrtBinaryLoadFromFile` API 在 CANN 8.2.RC1 中存在，但其 binary loader 期望特定格式的二进制文件（由 opp 构建系统生成），不接受 `-shared -fPIC` 产出的标准 ELF `.so` 文件。已调整为产出 `.o` + `--cce-aicore-arch=da-vinci` 方案，待硬件验证。

| CANN 版本 | `aclrtBinaryLoadFromFile` 支持自定义 kernel 二进制 |
|-----------|---------------------------------------------------|
| 8.2.RC1 | 待验证（`.o` + `--cce-aicore-arch=da-vinci` 方案） |
| 8.5+ | 支持（Kernel Launch API 文档即为 8.5alpha002） |

**影响**: 在当前 CANN 8.2.RC1 环境下，`SourceModule` 的编译和缓存功能正常工作，但加载和启动 kernel 需要升级 CANN 到 8.5+。依赖 binary loading 的测试已标记为 `skip`，升级 CANN 后自动激活（见下方 §9 测试策略）。

---
## 9. 测试策略

### 9.1 测试文件结构

```
tests/asnumpy_tests/compiler_tests/
├── __init__.py
├── conftest.py                      # kernel 源码 fixtures + mock fixtures + skip 标记
└── test_source_module.py            # 全部测试（52 个），按类组织
```

### 9.2 测试分层（6 层，按依赖从轻到重）

```
Layer 0 ─ 签名解析      纯 Python，无需 NPU / 编译器
Layer 1 ─ 编译          仅需 bisheng 编译器（可独立验证）
Layer 2 ─ 集成 (mock)   需 bisheng + mock C 扩展（逻辑正确性）
Layer 2'─ 集成 (real)   需 bisheng + aclrtBinaryLoadFromFile（CANN 8.5+）
Layer 3 ─ 执行 (mock)   需 mock C 扩展（参数编组 / 启动流程）
Layer 3'─ 执行 (real)   需 bisheng + NPU + aclrtBinaryLoadFromFile（CANN 8.5+，端到端正确性）
Layer 4 ─ 缓存          混合（纯 Python + 编译 + mock/real C 扩展）
Layer 5 ─ 边界情况      混合
```

**双轨测试设计**：对依赖 CANN 8.5+ binary loading 的测试类，提供两套实现：

| 轨道 | 机制 | 验证什么 | 何时运行 |
|------|------|---------|---------|
| **Mock 轨道** | `unittest.mock.patch` 替换 `_get_lib()` | Python 编排逻辑、参数编组、API 调用链 | CANN 8.2 即可运行 |
| **Real 轨道** | 真实 C 扩展 + NPU | 端到端正确性（与 NumPy 结果对比） | CANN 8.5+ / 硬件修复后 |

| 测试类 | 层 | 数量 | 轨道 | CANN 8.2 状态 |
|--------|----|------|------|--------------|
| `TestSignatureParsing` | 0 | 6 | — | ✅ 6 通过 |
| `TestCompilation` | 1 | 4 | — | ✅ 4 通过 |
| `TestSourceModuleMock` | 2 | 9 | mock | ✅ 9 通过 |
| `TestSourceModule` | 2' | 7 | real | ⏸ 7 skip |
| `TestKernelFunctionMock` | 3 | 7 | mock | ✅ 7 通过 |
| `TestPreparedKernelMock` | 3 | 4 | mock | ✅ 4 通过 |
| `TestKernelExecution` | 3' | 4 | real | ⏸ 4 skip |
| `TestCaching` | 4 | 8 | 混合 | ✅ 8 通过 |
| `TestEdgeCases` | 5 | 3 | 混合 | ✅ 3 通过 |
| **合计** | | **52** | | **41 通过 / 11 skip** |

### 9.3 Mock 机制

利用 `_get_lib()` 的**延迟导入**特性，通过 `unittest.mock.patch.object` 将其替换为返回 `MagicMock` 的 lambda：

```python
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

@contextmanager
def _mock_get_lib(module, mock_lib):
    with patch.object(module, "_get_lib", return_value=mock_lib):
        yield
```

**Mock C 扩展行为**（`conftest.py` 中的 `mock_compiler_lib` fixture）：

| 函数 | 模拟行为 |
|------|---------|
| `load_binary(path)` | 返回自增 handle (int) |
| `get_function(bin_handle, name)` | 返回自增 handle (int) |
| `unload_binary(bin_handle)` | no-op |
| `launch_kernel(func, block_dim, args, stream)` | no-op |
| `create_event()` / `create_stream()` | 返回自增 handle |
| `record_event()` / `synchronize_event()` | no-op |
| `elapsed_time_between(start, end)` | 返回 1.5 (ms) |
| `destroy_event()` / `destroy_stream()` | no-op |

**atexit 清理**：`conftest.py` 提供 autouse fixture `_clean_atexit_registry`，在每个测试前后清空 `source_module._atexit_registry`，防止跨测试的残留 SourceModule 实例干扰。

**PreparedKernel 安全清理**：mock 测试结束后 `PreparedKernel.__del__` 可能被 gc 触发并调用真实 C 扩展（导致 segfault），需在退出 mock 上下文前将 `_start_event`/`_end_event`/`_stream` 置为 None。

### 9.4 Skip 标记

```python
# conftest.py / test_source_module.py 中定义

requires_npu = pytest.mark.skipif(
    "ASCEND_TOOLKIT_HOME" not in os.environ
    and "ASCEND_HOME_PATH" not in os.environ,
    reason="Requires CANN NPU device",
)

requires_bisheng = pytest.mark.skipif(
    not Path(".../bisheng").exists()
    and "ASCEND_TOOLKIT_HOME" not in os.environ,
    reason="Requires bisheng compiler",
)

# CANN 8.2 的 aclrtBinaryLoadFromFile 不接受 bisheng 产出的标准 ELF
# 8.5+ 才完全支持自定义 kernel 二进制加载
requires_acl_binary_load = pytest.mark.skip(
    reason="aclrtBinaryLoadFromFile requires CANN 8.5+ for custom kernel binaries"
)
```

**Skip 的应用方式**：
- `TestSourceModule` 和 `TestKernelExecution`（real 轨道）在类级别应用：`pytestmark = [requires_acl_binary_load, ...]`
- `TestSourceModuleMock` 等 mock 轨道类**不使用** skip 标记，仅需 `@requires_bisheng`（编译需要）

### 9.5 Kernel 源码 Fixtures

`conftest.py` 提供 3 个 kernel 源码 fixture，已验证与 CANN 8.2.RC1 Ascend C API 兼容：

| Fixture | Kernel 函数 | 参数签名 | 用途 |
|---------|------------|---------|------|
| `vector_add_source` | `vector_add` | `(float*, float*, float*, int)` | 基本向量加法，SPMD 分块 |
| `fill_const_source` | `fill_const` | `(float*, float, int)` | 标量赋值，含 `Duplicate` |
| `multi_kernel_source` | `kernel_one` + `kernel_two` | `(float*, float*, int)` 各 | 多 kernel 单文件，`Add` vs `Mul` |

### 9.6 运行方式

```bash
# ========= 当前 CANN 8.2 环境 =========

# 全部测试（52 个，41 通过 + 11 skip）
pytest tests/asnumpy_tests/compiler_tests/ -v

# 仅运行 mock 轨道（排除 real 轨道 skip）
pytest tests/asnumpy_tests/compiler_tests/ -v \
    -k "not (acl_binary_load or npu)"

# 分层运行
pytest tests/asnumpy_tests/compiler_tests/ -v -k "SignatureParsing"       # Layer 0
pytest tests/asnumpy_tests/compiler_tests/ -v -k "Compilation"            # Layer 1
pytest tests/asnumpy_tests/compiler_tests/ -v -k "Mock"                   # Layer 2+3 mock
pytest tests/asnumpy_tests/compiler_tests/ -v -k "Caching"                # Layer 4

# 单个测试
pytest tests/asnumpy_tests/compiler_tests/test_source_module.py::TestCompilation::test_compiles_without_error -v

# ========= CANN 8.5+ 环境 =========
# 移除 requires_acl_binary_load 的 skip 后，全部 52 个测试可运行
# mock 轨道负责逻辑正确性，real 轨道负责端到端 NPU 正确性验证
```

### 9.7 各测试类详述

#### TestSignatureParsing（6 个，纯 Python）

无需任何外部依赖，测试签名解析引擎：

| 测试 | 验证点 |
|------|--------|
| `test_parse_simple_kernel` | `vector_add` → 3 个 `float*` + 1 个 `int32` |
| `test_parse_mixed_args` | `fill_const` → `float*` + `float32` + `int32` |
| `test_parse_kernel_not_found` | 不存在的 kernel 名 → `ValueError` |
| `test_parse_no_params` | 无参 kernel → 空列表 |
| `test_manual_signature_override` | 显式 `signature=["float32*", ...]` → 正确 `ArgSpec` |
| `test_manual_signature_scalar_types` | `int64`/`float64`/`bool` → 正确 `size_bytes` |

#### TestCompilation（4 个，仅需 bisheng）

验证 bisheng 编译管线：

| 测试 | 验证点 |
|------|--------|
| `test_compiles_without_error` | 编译成功，产出 `.o` 文件存在 |
| `test_compiled_o_is_elf` | 读取前 4 字节 = `\x7fELF` |
| `test_compile_invalid_source_raises` | 非法 C++ → `CompileError` |
| `test_compile_log_written` | `compile.log` 写入编译输出 |

#### TestSourceModuleMock（9 个，需 bisheng + mock C 扩展）

Mock 轨道。验证 SourceModule 的 Python 编排逻辑——编译→加载→函数查找→签名解析的全链路：

| 测试 | 验证点 |
|------|--------|
| `test_compilation_and_loading` | 编译→`load_binary`→`get_function` 调用链 |
| `test_get_function_returns_callable` | `get_function("vector_add")` 返回 `KernelFunction` |
| `test_get_function_unknown_name_raises` | 未知 kernel → `ValueError` |
| `test_close_calls_unload` | `close()` 调用 `unload_binary`，二次调用不重复 |
| `test_context_manager` | `with SourceModule(...)` 退出时调 `close()` |
| `test_multi_kernel_source` | 多 kernel 全部发现，`get_function` 调用≥2次 |
| `test_manual_signature` | 手动签名 override 源码解析 |
| `test_list_functions_empty_for_no_kernel_source` | 无 kernel 源码 → 空列表 |
| `test_disable_cache_skips_cache` | `disable_cache=True` 仍正常编译和加载 |

#### TestSourceModule（7 个，需 CANN 8.5+）

Real 轨道。当前全部 skip，与 mock 轨道测试项一一对应但使用真实 C 扩展：

| 测试 | 验证点 |
|------|--------|
| `test_compilation_and_loading` | 编译 → 加载 → `list_functions()` 包含 kernel 名 |
| `test_get_function_returns_callable` | `get_function("vector_add")` 返回 `KernelFunction` |
| `test_get_function_unknown_name_raises` | 未知 kernel → `ValueError` |
| `test_close` | 重复 `close()` 安全 |
| `test_context_manager` | `with SourceModule(...) as mod:` 用法 |
| `test_multi_kernel_source` | 单文件多 kernel → 全部出现在 `list_functions()` |
| `test_manual_signature` | `get_function(name, signature=[...])` 手动签名 |

#### TestKernelFunctionMock（7 个，需 mock C 扩展）

Mock 轨道。验证参数编组逻辑和 launch 调用：

| 测试 | 验证点 |
|------|--------|
| `test_marshal_pointer_args_packs_device_address` | 指针参数打包为 8 字节 device address |
| `test_marshal_pointer_arg_raises_for_non_ndarray` | 非 ndarray → `TypeError` |
| `test_marshal_wrong_arg_count_raises` | 参数数量不匹配 → `TypeError` |
| `test_marshal_mixed_args` | 指针+标量混合打包（8/4/4 字节） |
| `test_call_launches_kernel` | `__call__` 触发 `launch_kernel`，参数正确传递 |
| `test_call_default_grid_is_one` | 不传 grid → block_dim=1 |
| `test_marshal_scalar_types` | int64(8B)/float64(8B)/bool(1B) 尺寸验证 |

#### TestPreparedKernelMock（4 个，需 mock C 扩展）

Mock 轨道。验证事件管理和性能计时：

| 测试 | 验证点 |
|------|--------|
| `test_prepare_creates_events_and_stream` | `prepare()` 创建 2 个 event + 1 个 stream |
| `test_call_records_events_and_measures_elapsed` | 调用时 record start→launch→record end→sync→elapsed |
| `test_time_property_returns_elapsed` | `pk.time` 返回 `elapsed_time_between` 结果 |
| `test_del_cleans_up_events_and_stream` | `__del__` 调 `destroy_event`×2 + `destroy_stream` |

#### TestKernelExecution（4 个，需 NPU + CANN 8.5+）

Real 轨道。端到端测试，完整覆盖编译 → 加载 → 启动 → 结果验证：

| 测试 | 验证点 |
|------|--------|
| `test_vector_add_correctness` | NPU 结果 vs NumPy 参考，`rtol=1e-4, atol=1e-5` |
| `test_vector_add_different_sizes` | N = 256 / 1024 / 4096，遍历验证 |
| `test_scalar_kernel` | `fill_const(c_ap, 3.14, N)`，验证标量参数传递 |
| `test_default_grid` | 不传 `grid` 参数，默认 1 核执行 |

#### TestCaching（8 个）

| 测试 | 依赖 | 验证点 |
|------|------|--------|
| `test_cache_key_deterministic` | 无 | 相同输入 → 相同 SHA256 |
| `test_cache_key_different_source` | 无 | 不同源码 → 不同 key |
| `test_cache_key_different_options` | 无 | `-O2` vs `-O3` → 不同 key |
| `test_cache_key_different_soc` | 无 | 不同 soc_version → 不同 key |
| `test_get_cache_dir` | 无 | 返回 `~/.asnumpy/cache/` |
| `test_get_compiler_version` | 无 | bisheng `--version` 返回字符串 |
| `test_cache_store_and_hit` | bisheng | 编译 → 存储 → 命中 |
| `test_disable_cache` | bisheng + mock | `disable_cache=True` 仍正常工作 |

#### TestEdgeCases（3 个）

| 测试 | 验证点 |
|------|--------|
| `test_empty_source` | 空源码 → `extern "C"` 警告 |
| `test_source_without_kernels_compiles` | 纯注释源码不抛异常 |
| `test_source_without_extern_c_warns` | 缺 `extern "C"` → `UserWarning`，bisheng 编译通过 |

### 9.8 CANN 版本升级后的变化

当 CANN 升级到 8.5+ 后，只需修改 `requires_acl_binary_load` 标记：

```python
# 改前（当前）
requires_acl_binary_load = pytest.mark.skip(
    reason="aclrtBinaryLoadFromFile requires CANN 8.5+"
)

# 改后
requires_acl_binary_load = pytest.mark.skipif(
    not Path(".../bisheng").exists(),
    reason="Requires bisheng compiler"
)
```

全部 52 个测试将自动激活——mock 轨道覆盖 Python 逻辑正确性，real 轨道覆盖 NPU 端到端正确性。

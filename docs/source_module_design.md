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
        --cce-aicore-lang \
        --std=c++17 -O2 -fPIC \
        -I${ASCENDC_INC} \
        -I${ASCENDC_INC}/basic_api \
        -I${ASCENDC_INC}/highlevel_api \
        -I${ASCENDC_INC}/basic_api/impl \
        -I${ASCENDC_INC}/basic_api/inner_interface \
        -I${ASCENDC_INC}/basic_api/interface \
        kernel.cpp -o kernel.o
```

> **注**: `soc_version` 需使用 `Ascend910B1`（或 B2/B3/B4），`Ascend910B` 不被 bisheng 接受。
> Ascend C 头文件位于 `aarch64-linux/ascendc/include/` 而非 `include/`，kernel 编译不需要 host 侧的 ACL/ACLNN 头文件。

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

`aclrtBinaryLoadFromFile` API 在 CANN 8.2.RC1 中存在，但其 binary loader 期望特定格式的二进制文件（由 opp 构建系统生成），不接受 bisheng 直接编译产出的标准 ELF `.so`/`.o` 文件。

| CANN 版本 | `aclrtBinaryLoadFromFile` 支持自定义 kernel 二进制 |
|-----------|---------------------------------------------------|
| 8.2.RC1 | 不支持（报错: `program can not be null`, error 107000） |
| 8.5+ | 支持（temp.md 引用的 Kernel Launch API 文档即为 8.5alpha002） |

**影响**: 在当前 CANN 8.2.RC1 环境下，`SourceModule` 的编译和缓存功能正常工作，但加载和启动 kernel 需要升级 CANN 到 8.5+。`tests/asnumpy_tests/compiler_tests/` 中依赖 binary loading 的 13 个测试已标记为 `skip`，升级 CANN 后自动激活。

**临时验证方式**: 可通过独立测试 bisheng 编译管线（19 个已通过的测试覆盖编译、签名解析、缓存）确认基础设施正确性。

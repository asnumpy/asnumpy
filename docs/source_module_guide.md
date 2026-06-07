# AsNumpy SourceModule — Ascend C 内核 JIT 编译与启动指南

> 对照 PyCUDA `SourceModule`，在 AsNumpy 中实现 Ascend NPU 自定义算子 JIT 编译与启动。

## 概述

`asnumpy.compiler.SourceModule` 允许你在 Python 中编写 Ascend C 内核源码，运行时由毕昇（Bisheng）编译器编译为 NPU 可执行代码，直接启动到 Ascend 910B NPU 上。

与 PyCUDA 的核心对比：

| 概念 | PyCUDA (CUDA) | AsNumpy (Ascend) |
|------|---------------|-------------------|
| 编译器 | `nvcc` | `bisheng`（毕昇） |
| 编译产物 | `.cubin` | `.o`（含 `.aicore_binary` ELF 段） |
| 加载 API | `cuModuleLoad` | `rtDevBinaryRegister` + `rtFunctionRegister` |
| 启动 API | `cuLaunchKernel` | `rtKernelLaunch` |
| 并行模型 | SIMT（线程网格） | SPMD（AI Core 分块） |
| grid/block | `gridDim` + `blockIdx` + `threadIdx` | `GetBlockNum()` + `GetBlockIdx()`（无 threadIdx） |
| 参考文档 | [PyCUDA](https://documen.tician.de/pycuda/) | [Ascend C 开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html) |

---

## 快速开始

### 第一个内核：向量加法

```python
import numpy as np
import asnumpy as ap
from asnumpy.compiler import SourceModule

# 1. 编写 Ascend C 内核源码
kernel_src = r"""
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

    TPipe pipe;
    pipe.Init();

    TBuf<TPosition::VECIN> buf_a, buf_b;
    TBuf<TPosition::VECOUT> buf_out;
    pipe.InitBuffer(buf_a, static_cast<uint32_t>(count));
    pipe.InitBuffer(buf_b, static_cast<uint32_t>(count));
    pipe.InitBuffer(buf_out, static_cast<uint32_t>(count));

    GlobalTensor<float> gA, gB, gC;
    gA.SetGlobalBuffer(a + start, static_cast<uint64_t>(count));
    gB.SetGlobalBuffer(b + start, static_cast<uint64_t>(count));
    gC.SetGlobalBuffer(c + start, static_cast<uint64_t>(count));

    LocalTensor<float> local_a = buf_a.AllocTensor<float>();
    LocalTensor<float> local_b = buf_b.AllocTensor<float>();
    LocalTensor<float> local_c = buf_out.AllocTensor<float>();

    DataCopy(local_a, gA, count);
    DataCopy(local_b, gB, count);
    Add(local_c, local_a, local_b, count);
    DataCopy(gC, local_c, count);

    buf_out.FreeTensor(local_c);
    buf_b.FreeTensor(local_b);
    buf_a.FreeTensor(local_a);
}
"""

# 2. JIT 编译
mod = SourceModule(kernel_src, options=["-O3"], verbose=True)

# 3. 获取内核函数
vec_add = mod.get_function("vector_add")
print(f"Kernels found: {mod.list_functions()}")

# 4. 准备数据
N = 1024
rng = np.random.RandomState(42)
a_np = rng.randn(N).astype(np.float32)
b_np = rng.randn(N).astype(np.float32)

a_ap = ap.ndarray.from_numpy(a_np)
b_ap = ap.ndarray.from_numpy(b_np)
c_ap = ap.empty((N,), dtype=ap.float32)

# 5. 在 NPU 上启动内核（8 个 AI Core 并行）
vec_add(a_ap, b_ap, c_ap, N, grid=(8,))

# 6. 取回结果
result = c_ap.to_numpy()
expected = a_np + b_np
print(f"Max error: {np.max(np.abs(result - expected)):.2e}")

# 7. 性能计时
prepared = vec_add.prepare()
for _ in range(100):
    prepared(a_ap, b_ap, c_ap, N)
print(f"Avg kernel time: {prepared.time:.3f} ms")

mod.close()
```

### 更多示例

`examples/` 目录包含 11 个示例脚本：

| 脚本 | 内容 |
|------|------|
| `01_add.py` | `ap.add()` — 基础 NPU 加法 |
| `02_exp2.py` | `ap.exp2()` — 指数运算 |
| `03_multiply.py` | `ap.multiply()` — 乘法 |
| `04_all.py` | `ap.all()` — 逻辑归约 |
| `05_divide.py` | `ap.divide()` — 除法 |
| `06_vdot.py` | `ap.vdot()` — 向量点积 |
| `07_full.py` | `ap.full()` — 填充数组 |
| `08_linspace.py` | `ap.linspace()` — 线性空间 |
| `09_mean.py` | `ap.mean()` — 均值 |
| `10_sort.py` | `ap.sort()` — 排序 |
| **`11_custom_kernel.py`** | **SourceModule — 自定义 Ascend C 内核** |

---

## API 参考

### SourceModule

```python
class SourceModule:
    def __init__(
        self,
        source: str,                           # Ascend C 内核源码
        options: list[str] | None = None,      # bisheng 编译选项，如 ["-O3"]
        include_dirs: list[str] | None = None, # 用户附加 -I 路径
        cache_dir: str | None = None,          # None → ~/.asnumpy/cache/
        disable_cache: bool = False,           # 禁用缓存，强制重新编译
        keep: bool = False,                    # 保留中间产物 (.cpp/.o/compile.log)
        verbose: bool = False,                 # 打印编译命令与进度
        soc_version: str = "Ascend910B4",      # 芯片型号（npu-smi info 查看）
        core_type: str = "VecCore",            # VecCore / CubeCore / AICore
        compiler: str | None = None,           # bisheng 路径；None → 自动探测
    ):
        """JIT 编译 Ascend C 内核源码，注册到 CANN 运行时。"""
        ...

    def get_function(
        self, name: str, signature: list[str] | None = None
    ) -> KernelFunction:
        """获取可调用的内核函数。

        ``signature`` 为 None 时自动从源码解析参数类型；
        显式指定可覆盖自动解析，如 ``["float32*", "int32"]``。
        """
        ...

    def list_functions(self) -> list[str]:
        """返回本模块中所有内核函数名称列表。"""
        ...

    def close(self) -> None:
        """释放二进制句柄，可安全多次调用。"""
        ...

    def __enter__(self) -> "SourceModule": ...
    def __exit__(self, *args) -> None: ...
```

### KernelFunction

```python
class KernelFunction:
    """代表一个已编译的 Ascend C 内核函数，可调用。"""

    def __call__(
        self,
        *args,                                  # ndarray | Python 标量
        grid: tuple[int, ...] | None = None,    # AI Core 数量，如 (8,)
        stream: object | None = None,           # 流句柄，None = 默认同步流
    ) -> None:
        """在 NPU 上启动内核。

        - ``__gm__`` 指针参数 → 传入 ``ndarray``
        - 标量参数 → 传入 Python int/float/bool，自动转换为对应 np 类型
        - ``grid`` 指定使用的 AI Core 数量，默认 ``(1,)``
        """
        ...

    def prepare(self) -> PreparedKernel:
        """返回带性能计时的 :class:`PreparedKernel` 对象。"""
        ...
```

### PreparedKernel

```python
class PreparedKernel:
    """带性能计时的内核调用包装器，支持预热和重复计时。"""

    def __call__(self, *args, stream=None) -> None:
        """启动内核（与 KernelFunction 调用方式相同）。"""
        ...

    @property
    def time(self) -> float:
        """最近一次执行的耗时，单位为毫秒。"""
        ...
```

---

## 内核编写指南

### 基本范式：CopyIn → Compute → CopyOut

```cpp
extern "C" __global__ __aicore__ void my_kernel(
    __gm__ float* input, __gm__ float* output, int n)
{
    // 1. 计算当前 AI Core 负责的数据范围
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (n + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < n - start) ? per_block : (n - start);

    // 2. 初始化流水线和缓冲区
    TPipe pipe; pipe.Init();
    TBuf<TPosition::VECIN> buf_in;
    TBuf<TPosition::VECOUT> buf_out;
    pipe.InitBuffer(buf_in, count);
    pipe.InitBuffer(buf_out, count);

    // 3. 绑定全局内存
    GlobalTensor<float> gIn, gOut;
    gIn.SetGlobalBuffer(input + start, count);
    gOut.SetGlobalBuffer(output + start, count);

    // 4. CopyIn: 全局 → 局部
    LocalTensor<float> local = buf_in.AllocTensor<float>();
    DataCopy(local, gIn, count);

    // 5. 计算（在此使用 Ascend C 算子）
    // ...

    // 6. CopyOut: 局部 → 全局
    DataCopy(gOut, local, count);
    buf_out.FreeTensor(local);
}
```

### 参数类型映射

| Ascend C 参数类型 | Python 传入类型 | 传递大小 |
|-------------------|----------------|----------|
| `__gm__ float*` / `__gm__ half*` / `__gm__ int*` | `ndarray` | 8 字节（设备地址） |
| `int` / `int32_t` | Python `int` → `np.int32` | 4 字节 |
| `int64_t` / `long long` | Python `int` → `np.int64` | 8 字节 |
| `float` | Python `float` → `np.float32` | 4 字节 |
| `double` | Python `float` → `np.float64` | 8 字节 |
| `bool` | Python `bool` | 1 字节 |

### 重要约束

1. **`extern "C"` 必须**：内核函数必须声明为 `extern "C"`，否则运行时无法按名称查找。
2. **无 `threadIdx`**：Ascend C 使用 SPMD 模型，核内使用向量指令，不存在 CUDA 的线程概念。`grid` 参数映射到 AI Core 数量。
3. **`__gm__` 指针**：所有全局内存指针必须加 `__gm__` 修饰符。
4. **`TPipe` 初始化**：每个内核必须初始化 `TPipe` 并管理其缓冲区生命周期。

---

## 内核编译

SourceModule 内部调用 bisheng 编译器，完整命令等价于：

```bash
ASCENDC_INC=${ASCEND_TOOLKIT_HOME}/aarch64-linux/ascendc/include
ASC_INC=${ASCEND_TOOLKIT_HOME}/aarch64-linux/asc

bisheng --cce-soc-version=Ascend910B4 --cce-soc-core-type=VecCore \
        --cce-aicore-lang --cce-aicore-arch=da-vinci \
        --std=c++17 -O2 -fPIC -shared \
        -I${ASCENDC_INC} -I${ASCENDC_INC}/basic_api \
        -I${ASCENDC_INC}/highlevel_api \
        -I${ASCENDC_INC}/basic_api/impl \
        -I${ASCENDC_INC}/basic_api/interface \
        -I${ASC_INC} -I${ASC_INC}/include \
        kernel.cpp -o kernel.o
```

关键标志说明：
- `--cce-soc-version`：目标芯片型号（用 `npu-smi info` 查看实际硬件）
- `--cce-soc-core-type=VecCore`：AI Core 类型
- `-shared`：必须保留——Ascend C 内核引用运行时符号（`rtLaunch`/`rtFunctionRegister` 等）
- `--cce-aicore-arch=da-vinci`：目标 AI Core 架构

编译器路径自动探测（优先级从高到低）：
1. `${ASCEND_TOOLKIT_HOME}/tools/bisheng_compiler/bin/bisheng`（CANN 9.x）
2. `${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin/bisheng`（CANN 9.x alt）
3. `${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin/bisheng`（CANN 7.x/8.x）

---

## 缓存

SourceModule 默认使用基于 SHA256 的缓存，避免重复编译相同源码。

**缓存目录**：`~/.asnumpy/cache/{sha256_hash}/`

```
~/.asnumpy/cache/
└── {sha256_hash}/
    ├── kernel.cpp
    ├── kernel.o
    ├── meta.json
    └── compile.log
```

**缓存键** = `SHA256(source + sorted(options) + soc_version + include_dirs + compiler_version)`

同一编译单元内的多个内核函数共享一个缓存条目。`list_functions()` 通过 `objdump -t` 解析符号表获取函数名。

```python
# 禁用缓存（强制重新编译）
mod = SourceModule(source, disable_cache=True)

# 自定义缓存目录
mod = SourceModule(source, cache_dir="/path/to/cache")
```

---

## 架构总览

### 核心工作流

```
Python SourceModule(source, options)
    │
    ├─ 1. SHA256 缓存检查
    │      ├─ 命中 → 直接加载缓存 .o
    │      └─ 未命中 ↓
    ├─ 2. bisheng 编译 .cpp → .o
    ├─ 3. extract_aicore_elf: 从 .o 提取 .aicore_binary ELF 段
    ├─ 4. rtDevBinaryRegister: 注册到 CANN 运行时
    ├─ 5. rtFunctionRegister: 注册内核函数入口
    │
    └─ get_function(name) → KernelFunction
           │
           └─ __call__(*args, grid=(N,))
                  ├─ 参数编组（指针 8B + 标量 sizeof）
                  └─ rtKernelLaunch → NPU 执行
```

### 模块结构

```
src/asnumpy/compiler/          # Python 前端
├── __init__.py
├── source_module.py           # SourceModule：编排编译→加载→缓存
├── bisheng_compiler.py        # subprocess 调用 bisheng
├── kernel_function.py         # KernelFunction + PreparedKernel
├── cache.py                   # SHA256 缓存管理
└── _rts_loader.py             # ctypes 驱动的 RTS 加载/启动

src/compiler/
└── kernel_launcher.cpp        # C++ 后端：ACL/RTS 内核启动

include/asnumpy/compiler/
└── kernel_launcher.hpp        # C++ 头文件

bindings/python/
└── bind_compiler.cpp          # pybind11 绑定
```

**设计原则**：Python 层负责编译、缓存、用户 API——利用标准库的文件 I/O、subprocess、hashlib 能力；C++ 层仅负责必须用 CANN C API 的操作：二进制加载、内核启动、事件计时。

---

## 已知限制

### CANN 版本兼容性

| CANN 版本 | 内核加载 | 内核执行 | 说明 |
|-----------|---------|---------|------|
| 8.2.RC1 | ❌ | ❌ | 不支持 bisheng 标准 ELF 格式 |
| 8.5+ | ✅ | ✅ | 完全支持 |
| **9.1.0** | ✅ | ⚠️ 部分 | 见下方 910B4 问题 |

### CANN 9.1 + Ascend 910B4

在 CANN 9.1.0 + Ascend 910B4 环境下，bisheng 编译器对算术操作生成的 AI Core 指令与硬件不兼容：

- ✅ `DataCopy`、`Duplicate` — 正常工作
- ❌ `Add`、`Mul`、`GetValue`、`SetValue` — 触发 AI Core "Illegal instruction"（`errCode=0x10`, `fixp_error=0x5e/0x97`）

编译、加载、启动 API 调用均正常，仅 AI Core 执行时产生异常。根因是 bisheng 编译器目标架构与 910B4 硬件 ISA 不匹配，待 CANN 后续版本修复。

### 其他限制

1. **仅支持 1D grid** — `grid=(N,)` 格式，不支持多维网格
2. **无 JIT 模板** — 暂无 PyCUDA `DynamicSourceModule` 等价物
3. **无 `__shared__` 内存** — Ascend C 使用 UUB，分配方式不同于 CUDA shared memory

---

## 参考资料

- [PyCUDA 官方文档](https://documen.tician.de/pycuda/)
- [Ascend C 算子开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
- [Kernel Launch API 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/appdevgapi/aclcppdevg_03_1792.html)
- [毕昇异构编译文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/opdevg/BishengCompiler/atlas_bisheng_10_0011.html)
- [Ascend C JIT 实践](https://ai6s.net/692ba5a0791c233193d12515.html)

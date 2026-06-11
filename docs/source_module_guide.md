# AsNumpy SourceModule — Ascend C 内核 JIT 编译与启动指南

> 对照 PyCUDA `SourceModule`，在 AsNumpy 中实现 Ascend NPU 自定义算子 JIT 编译与启动。

## 概述

`asnumpy.compiler.SourceModule` 允许你在 Python 中编写 Ascend C 内核源码，运行时由毕昇（Bisheng）编译器编译为 NPU 可执行代码，直接启动到 Ascend 910B NPU 上。

与 PyCUDA 的核心对比：

| 概念 | PyCUDA (CUDA) | AsNumpy (Ascend) |
|------|---------------|-------------------|
| 框架 | `pycuda.compiler.SourceModule` (~200 行) | `asnumpy.compiler.SourceModule` (~240 行) |
| 编译器 | `nvcc` | `bisheng`（毕昇） |
| 编译产物 | `.cubin` | `.o`（含 `.aicore_binary` ELF 段） |
| 加载 API | `cuModuleLoadData` | `rtDevBinaryRegister` + `rtFunctionRegister` |
| 启动 API | `cuLaunchKernel` | `rtKernelLaunch` |
| 并行模型 | SIMT（线程网格） | SPMD（AI Core 分块） |
| grid/block | `gridDim` + `blockIdx` + `threadIdx` | `GetBlockNum()` + `GetBlockIdx()`（无 threadIdx） |
| 纯 Python | ✅ 零 C++ 编译依赖 | ✅ 零 C++ 编译依赖 |
| 参考文档 | [PyCUDA](https://documen.tician.de/pycuda/) | [Ascend C 开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html) |

---

## 快速开始

### 第一个内核：数据拷贝

```python
import numpy as np
import asnumpy as ap
from asnumpy.compiler import SourceModule

# 1. 编写 Ascend C 内核源码
kernel_src = r"""
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void data_copy(
    __gm__ float* src, __gm__ float* dst, int n)
{
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (n + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < n - start) ? per_block : (n - start);

    TPipe pipe;
    pipe.Init();

    GlobalTensor<float> gSrc, gDst;
    gSrc.SetGlobalBuffer(src + start, static_cast<uint64_t>(count));
    gDst.SetGlobalBuffer(dst + start, static_cast<uint64_t>(count));

    TBuf<AscendC::TPosition::VECIN> buf;
    pipe.InitBuffer(buf, static_cast<uint32_t>(count));
    LocalTensor<float> local = buf.AllocTensor<float>();

    DataCopy(local, gSrc, count);
    DataCopy(gDst, local, count);

    buf.FreeTensor(local);
}
"""

# 2. JIT 编译
mod = SourceModule(kernel_src, options=["-O3"])

# 3. 获取内核函数（显式签名）
kernel = mod.get_function("data_copy",
    signature=["float32*", "float32*", "int32"])

# 4. 准备数据
N = 1024
src_ap = ap.ndarray.from_numpy(np.random.randn(N).astype(np.float32))
dst_ap = ap.empty((N,), dtype=ap.float32)

# 5. 在 NPU 上启动内核（8 个 AI Core 并行）
kernel(src_ap, dst_ap, N, grid=(8,))

# 6. 取回结果
result = dst_ap.to_numpy()
```

### 完整示例

运行 `python examples/11_custom_kernel.py` 查看完整可执行示例。

---

## API 参考

### SourceModule

```python
class SourceModule:
    """JIT 编译 Ascend C 内核源码并暴露可调用函数。

    Parameters
    ----------
    source : str
        Ascend C 内核源码字符串。
    options : list[str] | None
        额外的 bisheng 编译选项（如 ``["-O3"]``）。
    arch : str
        目标芯片架构，默认 ``"Ascend910B4"``。
    core_type : str
        AI Core 类型，默认 ``"VecCore"``（可选 ``"CubeCore"``）。
    """
    def __init__(
        self,
        source: str,
        options: list[str] | None = None,
        arch: str = "Ascend910B4",
        core_type: str = "VecCore",
    ): ...

    def get_function(
        self, name: str, signature: list[str] | None = None
    ) -> KernelFunction:
        """获取可调用的内核函数。

        ``signature`` 为显式参数类型列表，如 ``["float32*", "int32"]``。
        ``T*`` 表示指针参数（对应 ``__gm__ T*``），其余为标量类型。
        """
        ...
```

**设计原则**：仿照 PyCUDA 的极简设计——

- 只做 JIT 编译 + 内核获取，不做缓存/符号解析/计时
- 无 `close()` / context manager — 依赖 Python GC (`__del__`) 自动释放
- 无 `list_functions()` — 用户自行管理内核名
- 无 `PreparedKernel` — 计时由用户层完成
- 零 C++ 编译依赖 — 纯 Python + ctypes

### KernelFunction

```python
class KernelFunction:
    """代表一个已编译的 Ascend C 内核函数，可调用。

    Parameters
    ----------
    name : str
        内核函数名。
    signature : list[str] | None
        显式参数类型列表，如 ``["float32*", "float32*", "int32"]``。
    """

    def __call__(
        self,
        *args,
        grid: tuple[int, ...] | None = None,
        stream=None,
    ) -> None:
        """在 NPU 上启动内核。

        - 指针参数（``float32*`` 等）→ 传入 ``ndarray``
        - 标量参数（``int32``, ``float32``, ``int64``, ``float64``, ``bool``）→ 传入 Python 数值
        - ``grid`` 指定 AI Core 数量，默认 ``(1,)``
        """
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
    TBuf<AscendC::TPosition::VECIN> buf_in;
    TBuf<AscendC::TPosition::VECOUT> buf_out;
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

| Ascend C 参数类型 | signature 值 | Python 传入类型 | 传递大小 |
|-------------------|-------------|----------------|----------|
| `__gm__ float*` | `"float32*"` | `ndarray` | 8 字节（设备地址） |
| `__gm__ half*` | `"float16*"` | `ndarray` | 8 字节 |
| `__gm__ int*` | `"int32*"` | `ndarray` | 8 字节 |
| `int` / `int32_t` | `"int32"` | Python `int` → `np.int32` | 4 字节 |
| `int64_t` | `"int64"` | Python `int` → `np.int64` | 8 字节 |
| `float` | `"float32"` | Python `float` → `np.float32` | 4 字节 |
| `double` | `"float64"` | Python `float` → `np.float64` | 8 字节 |
| `bool` | `"bool"` | Python `bool` | 1 字节 |

### 重要约束

1. **`extern "C"` 必须**：内核函数必须声明为 `extern "C"`，否则 C++ name mangling 导致符号无法查找。
2. **无 `threadIdx`**：Ascend C 使用 SPMD 模型，核内使用向量指令，不存在 CUDA 的线程概念。`grid` 参数映射到 AI Core 数量。
3. **`__gm__` 指针**：所有全局内存指针必须加 `__gm__` 修饰符。
4. **`TPipe` 初始化**：每个内核必须初始化 `TPipe` 并管理其缓冲区生命周期。

---

## 编译详解

SourceModule 内部调用 bisheng 编译器，完整命令等价于：

```bash
ASCENDC_INC=${ASCEND_TOOLKIT_HOME}/aarch64-linux/ascendc/include
ASC_INC=${ASCEND_TOOLKIT_HOME}/aarch64-linux/asc

bisheng \
  --cce-soc-version=Ascend910B4 --cce-soc-core-type=VecCore \
  --cce-aicore-lang --cce-aicore-arch=da-vinci \
  --std=c++17 -O2 -fPIC -shared \
  -I${ASCENDC_INC} \
  -I${ASCENDC_INC}/basic_api \
  -I${ASCENDC_INC}/highlevel_api \
  -I${ASCENDC_INC}/basic_api/impl \
  -I${ASCENDC_INC}/basic_api/interface \
  -I${ASC_INC} -I${ASC_INC}/include \
  kernel.cpp -o kernel.o
```

关键标志说明：
- `--cce-soc-version`：目标芯片型号（用 `npu-smi info` 查看实际硬件）
- `--cce-soc-core-type=VecCore`：AI Core 类型
- `-shared`：必须保留（Ascend C 内核引用运行时符号）
- `--cce-aicore-arch=da-vinci`：目标 AI Core 架构

编译器路径自动探测（优先级从高到低）：
1. `${ASCEND_TOOLKIT_HOME}/tools/bisheng_compiler/bin/bisheng`（CANN 9.x）
2. `${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin/bisheng`（CANN 9.x alt）
3. `${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin/bisheng`（CANN 7.x/8.x）

---

## 架构总览

### 核心工作流

```
Python SourceModule(source, options)
    │
    ├─ 1. bisheng 编译 .cpp → .o（subprocess）
    ├─ 2. objcopy 提取 .aicore_binary ELF 段
    ├─ 3. rtDevBinaryRegister: 注册原始 ELF 到 CANN 运行时
    ├─ 4. rtFunctionRegister: 按名称注册内核函数入口
    │
    └─ get_function(name) → KernelFunction
           │
           └─ __call__(*args, grid=(N,))
                  ├─ 参数编组（指针 8B + 标量对齐到 8B）
                  └─ rtKernelLaunch → NPU 执行
```

### 模块结构（3 个文件，~528 行纯 Python，零 C++）

```
src/asnumpy/compiler/
├── __init__.py              # 导出 SourceModule, KernelFunction
├── source_module.py         # SourceModule + bisheng 编译 + ELF 提取
├── kernel_function.py       # KernelFunction + 参数编组
└── _rt.py                   # ctypes 封装 RTS API (rtDevBinaryRegister 等)
```

**设计原则（KISS）**：仿照 PyCUDA 的 ~200 行极简设计。Python 层通过 `subprocess` 调用 bisheng，通过 `ctypes` 调用 `libruntime.so`。零 C++ 编译依赖，代码可审查。

---

## 已知限制

### CANN 9.1 + Ascend 910B4

在 CANN 9.1.0 + Ascend 910B4 环境下，bisheng 编译器生成的 AI Core 指令与硬件不兼容：

- 编译、加载、启动 API 调用均正常工作
- ❌ 内核执行后产生错误的数值结果（所有 Ascend C 算子均受影响：DataCopy、Duplicate、Add、Mul 等）

根因是 bisheng 编译器目标架构与 910B4 硬件 ISA 不完全匹配，待 CANN 后续版本修复。此问题已通过测试套件中的 `requires_kernel_exec` 标记记录。

### 其他限制

1. **仅支持 1D grid** — `grid=(N,)` 格式，不支持多维网格
2. **无编译缓存** — 每次 `SourceModule(source)` 都重新编译；缓存可由用户层实现
3. **无 JIT 模板** — 暂无 PyCUDA `DynamicSourceModule` 等价物
4. **无内置计时** — 计时由用户层使用 `time.time()` 或其他工具完成
5. **无 `context manager`** — 资源释放由 Python GC (`__del__`) 处理

---

## 参考资料

- [PyCUDA 官方文档](https://documen.tician.de/pycuda/)
- [Ascend C 算子开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
- [毕昇异构编译文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/opdevg/BishengCompiler/atlas_bisheng_10_0011.html)

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

### 第一个内核：向量加法

```python
import numpy as np
import asnumpy as ap
from asnumpy.compiler import SourceModule

# 1. 编写 Ascend C 内核源码（遵循官方 CANN 模式）
kernel_src = r"""
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void vector_add(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, int totalLength)
{
    TPipe pipe;

    // UB tiling: 每个 tile 256 个 float（1024 字节）
    constexpr int TILE_ELEMS = 256;
    constexpr int PIPELINE_DEPTH = 2;

    // 双缓冲队列（CopyIn → Compute → CopyOut 流水线）
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueA, inQueueB;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueueC;
    uint32_t tileBytes = TILE_ELEMS * sizeof(float);
    pipe.InitBuffer(inQueueA, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(inQueueB, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(outQueueC, PIPELINE_DEPTH, tileBytes);

    // SPMD: 计算当前 AI Core 负责的数据范围
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (totalLength + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < totalLength - start)
        ? per_block : (totalLength - start);

    GlobalTensor<float> gA, gB, gC;
    gA.SetGlobalBuffer(a + start, static_cast<uint64_t>(count));
    gB.SetGlobalBuffer(b + start, static_cast<uint64_t>(count));
    gC.SetGlobalBuffer(c + start, static_cast<uint64_t>(count));

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    int tileNum = count / TILE_ELEMS;
    int tailElems = count - tileNum * TILE_ELEMS;

    for (int i = 0; i < tileNum; ++i) {
        int offset = i * TILE_ELEMS;
        // CopyIn
        LocalTensor<float> localA = inQueueA.AllocTensor<float>();
        LocalTensor<float> localB = inQueueB.AllocTensor<float>();
        copyParams.blockLen = TILE_ELEMS * sizeof(float);
        DataCopyPad(localA, gA[offset], copyParams, padParams);
        DataCopyPad(localB, gB[offset], copyParams, padParams);
        inQueueA.EnQue(localA);
        inQueueB.EnQue(localB);
        // Compute
        localA = inQueueA.DeQue<float>();
        localB = inQueueB.DeQue<float>();
        LocalTensor<float> localC = outQueueC.AllocTensor<float>();
        Add(localC, localA, localB, TILE_ELEMS);
        outQueueC.EnQue(localC);
        inQueueA.FreeTensor(localA);
        inQueueB.FreeTensor(localB);
        // CopyOut
        localC = outQueueC.DeQue<float>();
        DataCopyPad(gC[offset], localC, copyParams);
        outQueueC.FreeTensor(localC);
    }
    // 处理尾部元素（此处省略，完整代码见 examples/11_custom_kernel.py）
}
"""

# 2. JIT 编译
mod = SourceModule(kernel_src, options=["-O3"])

# 3. 获取内核函数（显式签名）
kernel = mod.get_function("vector_add",
    signature=["float32*", "float32*", "float32*", "int32"])

# 4. 准备数据
N = 1024
a_ap = ap.ndarray.from_numpy(np.random.randn(N).astype(np.float32))
b_ap = ap.ndarray.from_numpy(np.random.randn(N).astype(np.float32))
c_ap = ap.empty((N,), dtype=ap.float32)

# 5. 在 NPU 上启动内核（8 个 AI Core 并行）
kernel(a_ap, b_ap, c_ap, N, grid=(8,))

# 6. 取回结果
result = c_ap.to_numpy()
```

### 完整示例

- `examples/11_add_kernel.py` — 向量加法内核（最简入门）
- `examples/12_custom_kernel.py` — 多内核示例（vector_mul + scalar_mul，演示模板复用）

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

### 基本范式：分块 + 双缓冲 CopyIn → Compute → CopyOut

在 910B4 上编写 Ascend C 内核必须遵循以下模式（参考 ``ops-math`` 官方 add 算子）：

```cpp
extern "C" __global__ __aicore__ void my_kernel(
    __gm__ float* input, __gm__ float* output, int totalLength)
{
    TPipe pipe;

    // 1. UB 分块（tiling）：按 UB 容量切分数据
    constexpr int TILE_ELEMS = 256;   // 每个 tile 的元素数
    constexpr int PIPELINE_DEPTH = 2; // 双缓冲深度

    // 2. 双缓冲队列（硬件流水线：DMA 与计算重叠）
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueue;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueue;
    pipe.InitBuffer(inQueue, PIPELINE_DEPTH, TILE_ELEMS * sizeof(float));
    pipe.InitBuffer(outQueue, PIPELINE_DEPTH, TILE_ELEMS * sizeof(float));

    // 3. SPMD 数据分片
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (totalLength + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < totalLength - start) ? per_block : (totalLength - start);

    GlobalTensor<float> gIn, gOut;
    gIn.SetGlobalBuffer(input + start, static_cast<uint64_t>(count));
    gOut.SetGlobalBuffer(output + start, static_cast<uint64_t>(count));

    // 4. 使用 DataCopyPad + 显式参数（**不可用 plain DataCopy**）
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    // 5. 主循环：CopyIn → Compute → CopyOut
    int tileNum = count / TILE_ELEMS;
    for (int i = 0; i < tileNum; ++i) {
        int offset = i * TILE_ELEMS;
        // CopyIn
        auto localIn = inQueue.AllocTensor<float>();
        copyParams.blockLen = TILE_ELEMS * sizeof(float);
        DataCopyPad(localIn, gIn[offset], copyParams, padParams);
        inQueue.EnQue(localIn);
        // Compute
        localIn = inQueue.DeQue<float>();
        auto localOut = outQueue.AllocTensor<float>();
        // ... Ascend C 算子 (Add, Mul 等) ...
        outQueue.EnQue(localOut);
        inQueue.FreeTensor(localIn);
        // CopyOut
        localOut = outQueue.DeQue<float>();
        DataCopyPad(gOut[offset], localOut, copyParams);
        outQueue.FreeTensor(localOut);
    }
    // 6. 尾部处理（tailElems > 0 时，同上三阶段）
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
5. **❌ 禁止 `TBuf` + `DataCopy`（910B4）**：在 Ascend 910B4 上，使用 plain ``TBuf`` 和 ``DataCopy`` 进行全局→局部 DMA 会产生**错误数据**。必须使用 ``TQue`` 双缓冲 + ``DataCopyPad`` + ``DataCopyExtParams`` 模式（参见上方代码模板和官方 ``ops-math`` 示例）。
6. **UB 分块（tiling）必须**：不能一次性加载全部数据到 UB，必须按 tile 分块处理（UB 通常 192KB，每个 tile 建议 ≤ 1024 个 float32 元素）。
7. **`InitBuffer` 参数为字节数**：``pipe.InitBuffer(buf, byteSize)`` 的第二个参数是**字节数**（不是元素数）。对于 ``float32``，应传入 ``count * 4``。

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

### Ascend 910B4 DMA 约束

在 CANN 9.1.0 + Ascend 910B4 环境下，经系统性测试（对比 ``ops-math`` 官方算子库），确认以下内核编写约束：

- ``DataCopy``（plain）全局→局部 DMA 在运行时 copy count 不满足最小向量宽度（< 8 个 float32 / 32 字节）时产生错误数据
- ``TBuf`` 单缓冲模式下该问题尤为明显；``TQue`` 双缓冲 + ``DataCopyPad`` + ``DataCopyExtParams`` 模式完全正常
- ``Duplicate`` / ``DataCopy`` 局部→全局方向一切正常
- 根因是 910B4 向量引擎 32 字节对齐 DMA 与 bisheng 编译器对该模式的代码生成存在边界情况

**结论**：内核必须使用官方 CANN 推荐的 ``TQue`` 双缓冲 + ``DataCopyPad`` + UB tiling 模式，不可使用简化的 ``TBuf`` + ``DataCopy``。

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

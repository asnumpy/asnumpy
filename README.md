<div align="center">

<img src="docs/images/AsNumpy Logo.png" alt="AsNumpy Logo" width="280">

# 📘 AsNumpy

### 昇腾NPU原生Numpy

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![CANN](https://img.shields.io/badge/CANN-8.2RC1+-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-Ascend%20910B-green.svg)]()

</div>

在人工智能与深度学习飞速发展的当下，高效、友好的计算工具成为开发者们的迫切需求。随着算力需求的不断攀升，专用计算芯片及相应的软件生态愈发重要。值得注意的是，**截至2025年8月2日，Python 以26.14% 的占比成为 Tiobe 历史上最受欢迎的编程语言**，在科学计算、数据分析与人工智能领域均占据主导地位。而在 Python 生态中，**Numpy 是基石性的数学运算库**，为后续的诸多深度学习框架和工具提供了底层支撑。

> ❗ 为进一步提升开发者在昇腾 NPU 上的计算体验，哈尔滨工业大学计算学部苏统华教授团队联合华为团队打造了 **昇腾 NPU 原生 Numpy —— AsNumpy**。

AsNumpy 是一款 **深度支持昇腾 NPU 并高度兼容 Numpy 接口的轻量级 Python 数学运算库**。  
通过精心设计的 **NPUArray 核心数据结构**，AsNumpy 在 Python 层完全兼容 Numpy API，同时在 C++ 底层深度集成了 **华为 Ascend C 算子库**，对 NPU 算子（包括数学运算、线性代数、随机采样等）进行系统化封装，并高效管理 `dtype`、`shape` 以及 `aclTensor` 等底层资源，实现了与主机端 `numpy.ndarray` 的双向拷贝。借助 **pybind11** 实现高效的 Python-C++ 接口绑定，使用户能够像使用 Numpy 一样在 NPU 上无缝创建、操作张量。  

---
## ✨核心优势

与 Numpy 接口高度兼容，几乎无需额外学习成本，即可释放昇腾 NPU 的高效原生计算能力。

---
## 💡设计理念

### 数据结构 NPUArray

`NPUArray` 是 AsNumpy 的核心数据结构，设计理念主要体现在以下三个方面：

#### 🔄 兼容性
在 Python 层完全兼容 **Numpy API**，为用户提供与 `ndarray` 无缝衔接的体验。  
这意味着你可以像使用 **Numpy** 一样使用 `NPUArray`。

#### 📦 内部封装
`NPUArray` 内部封装了 `dtype`、`shape` 等基本信息，以及底层的数据 `aclTensor`。  
用户在 Python 层操作时，无需关心底层复杂的细节。

#### 🗂️ 资源管理
`NPUArray` 提供了方便的构造函数，并且在对象不再使用时，会自动调用析构函数释放资源。  
这种设计避免了手动管理内存可能引发的内存泄漏问题，使用户能够更加专注于算法逻辑。

---

### **API 架构**

AsNumpy 的 API 架构分为 **功能模块** 与 **基础模块** 两大类。

API 架构设计示意图如下所示：

<div align="center">
<img src="docs/images/功能模块.png" alt="API 架构设计图" width="50%">
</div>

#### 📊 具体功能
AsNumpy 已实现涵盖 **[数学运算](docs/math/)**（三角函数、指数、对数等常见数学计算）、线性代数（矩阵运算与分解）、**[随机抽样](docs/random/)**、**[逻辑函数](docs/logic/)**、**数组创建**、**输入输出**等多个功能模块，以及 **辅助工具**、**内存池管理**、**数据类型** 等基础模块，为开发者提供全面的科学计算支持。

详细 API 列表请参阅 [docs](docs/) 目录下的各模块文档。  

---

### 🚀 NPU 扩展功能模块

由于 CANN 内置算子主要面向 AI 编程与神经网络训练等深度学习场景，而 AsNumpy 作为通用科学计算库，更侧重于数据计算、处理与分析等传统数值计算场景，两者的应用领域存在差异。在将 Numpy 的完整 API 体系迁移到 NPU 的过程中，AsNumpy 会面临算子覆盖度不足的挑战。为此，引入了 **NPU 扩展功能模块**，通过自研算子与开源算子库相结合的方式，提升 API 覆盖率与计算性能。

NPU扩展功能模块设计示意图如下所示：

<div align="center">
<img src="docs/images/NPU扩展功能模块.png" alt="NPU扩展功能模块设计图" width="50%">
</div>

#### ⚠️ 现有问题
- AsNumpy 需要封装 CANN 内置算子。  
- CANN 内置算子无法覆盖 NumPy 的全部 API。  

#### 💡 解决方案
- 对缺失的算子进行手动开发。  
- 借助 **OpenBOAT** 开源算子库来补齐算子。  
- 优先安排缺失算子的开发计划。  

#### 🌟 价值体现
- **OpenBOAT** 为 AsNumpy 提供算子支持。  
- 推动 AsNumpy 兼容更多 NumPy API。  
- 加速 AsNumpy 项目的落地与完善。  

> 📌 **说明**：  
> OpenBOAT 是由 **哈尔滨工业大学计算学部苏统华教授团队** 设计的 Ascend C 算子仓库，能够为 AsNumpy 提供灵活的算子扩展支持。

---
## 📐 使用方式对比

> 💻 **一个例子说明一切**：实现大规模矩阵元素乘积与求和

<table>
<tr>
<td width="50%">

**使用 Numpy（CPU）**

```python
import numpy as np

rows, cols = 20000, 20000
m1 = np.random.normal(0, 1, (rows, cols))
m2 = np.random.normal(0, 1, (rows, cols))

# 计算
product = np.multiply(m1, m2)
result = np.sum(product)
```

</td>
<td width="50%">

**使用 AsNumpy（NPU）**

```python
import asnumpy as ap

# 数据迁移到 NPU
m1_npu = ap.ndarray.from_numpy(m1)
m2_npu = ap.ndarray.from_numpy(m2)

# 在 NPU 上计算
product = ap.multiply(m1_npu, m2_npu) # ⚡ 加速！
result = ap.sum(product)
```

</td>
</tr>
</table>

✨ **只需几行代码改动，即可享受 NPU 加速！**

---
## 🚀 功能支持状态

| 模块                     | 状态      | 说明                                  |
| ------------------------ | --------- | ------------------------------------- |
| 🔧 AscendCL 运行时管理   | ✅ 已完成 | 系统配置与资源管理                    |
| 📦 NPUArray 核心数据结构 | ✅ 已完成 | 与 Numpy ndarray 高度兼容             |
| 🎨 数组创建函数          | ✅ 已完成 | `ones`, `zeros`, `empty`, `arange`... |
| 🔄 与 Numpy 数据互转     | ✅ 已完成 | `to_numpy()`, `from_numpy()`          |
| 🧮 数学运算算子          | ✅ 已完成 | 三角函数、指数、对数等                |
| 📐 线性代数模块          | 🚧 进行中 | 矩阵运算、分解、求解等                |
| 🎲 随机采样模块          | 🚧 进行中 | 各类分布的随机数生成                  |
| 📊 完整 Numpy API 兼容   | 🚧 进行中 | 覆盖前 100 个高频 API                 |

---

## 📊 性能测试

针对 `multiply()` 函数（`float32` 类型）进行基准测试，对比 AsNumpy（NPU）与 Numpy（CPU）的性能表现。

> 💡 **关键发现**：随着张量规模增大，NPU 的并行计算优势愈发明显，最高可达 **128.70×** 加速比！

| 张量形状         | AsNumpy (NPU) | Numpy (CPU) | 加速比      |
| ---------------- | ------------- | ----------- | ----------- |
| (500, 500)       | 1.9355s       | 0.1708s     | 0.09×       |
| (1000, 1000)     | 0.0692s       | 0.7029s     | 🚀 10.16×  |
| (2000, 2000)     | 0.1033s       | 3.8387s     | 🚀 37.17×  |
| (3000, 3000)     | 0.1115s       | 14.3567s    | 🚀 128.70× |

<sub>*测试环境：Ascend 910B NPU vs CPU，单次运行时间*</sub>

---

## ⚙️ 快速开始

### 📦 环境要求

<details>
<summary>点击展开查看详细环境要求</summary>

**硬件平台**
- CPU：AArch64 或 X86_64
- NPU：昇腾 910B

**系统要求**
- 主流 Linux 系统（推荐 Ubuntu 22.04+）

**软件依赖**
- 编译工具：`GCC >= 11.2`、`CMake >= 3.26`
- Python 环境：`Python >= 3.9`
- CANN：`8.2.RC1.alpha003` 及以上版本

</details>

---

### 🚀 安装方式

#### 方式 1：使用 uv（推荐）

[uv](https://github.com/astral-sh/uv) 是现代化的 Python 包管理工具，速度更快。

```bash
# 克隆仓库
git clone --recursive https://gitcode.com/cann/asnumpy.git
cd asnumpy

# 使用 uv 构建并安装
uv sync
```

#### 方式 2：使用传统方式

```bash
# 克隆仓库
git clone --recursive https://gitcode.com/cann/asnumpy.git
cd asnumpy

# 构建并安装
python -m build
pip install dist/*.whl
```

---

### 🧪 快速验证

```python
import asnumpy as ap

# 创建 NPU 数组
arr = ap.ones((1000, 1000), dtype=ap.float32)
print(f"✅ AsNumpy 安装成功！数组形状：{arr.shape}")
```

---

### 🔧 开发与测试

#### 开发模式安装

如果您需要修改代码并进行测试，建议使用可编辑模式安装：

```bash
# 克隆仓库
git clone --recursive https://gitcode.com/cann/asnumpy.git
cd asnumpy

# 安装开发依赖并以可编辑模式安装
python -m pip install -e .
```

#### 运行测试

安装完成后，可以从项目根目录运行测试：

```bash
# 从项目根目录运行所有测试
python -m pytest tests
# 运行特定测试文件
python -m pytest tests/asnumpy_tests/math_tests/test_arithmetic.py
# 运行带详细输出的测试
python -m pytest tests -v
```

**注意事项：**
- 必须先以可编辑模式安装包（`pip install -e .`），这样可以确保 C++ 扩展被正确编译和链接
- pytest 配置已优化，支持从项目根目录直接运行测试
- 测试会自动检测 NPU 设备，如果没有 NPU，相关测试会被跳过

---

## ❓ 常见问题

<details>
<summary><b>如何检查 CANN 是否正确安装？</b></summary>

```bash
# 检查 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

</details>

<details>
<summary><b>遇到编译错误怎么办？</b></summary>

1. 确认 CMake 版本 >= 3.26：`cmake --version`
2. 确认 GCC 版本 >= 11.2：`gcc --version`
3. 确保已正确设置 CANN 环境变量

</details>

<details>
<summary><b>AsNumpy 与 Numpy 的兼容性如何？</b></summary>

AsNumpy 设计目标是与 Numpy API 高度兼容，但由于底层算子覆盖度限制，当前版本尚未实现全部 Numpy API。我们正在积极扩展算子库，目标是 v1.0 覆盖前 100 个高频 API。

</details>

---

## 🧭 下一步计划（欢迎贡献）

| 阶段       | 目标                                                                 |
| ---------- | -------------------------------------------------------------------- |
| **v0.2**   | 内部代码 **面向对象** 重构，结构更清晰                                |
| **v0.3**   | 支持逐元素 & 规约算子：`sum`、`matmul` …                              |
| **v1.0**   | **兼容 Numpy 使用率前 100 的 API，支持大部分 Numpy 算法迁移至 AsNumpy** |
| **v2.0**   | **扩展 Ascend C 算子库，支持使用自定义 Ascend C 算子**                |

---

## 🔗 相关链接

- 📚 [完整文档](docs/)
- 🐛 [问题反馈](https://gitcode.com/cann/asnumpy/issues)
- 🌟 [OpenBOAT 算子库](https://gitcode.com/hit1920/openboat)

---

## 🙏 致谢

特别感谢以下团队的贡献：
- 哈尔滨工业大学计算学部苏统华教授团队
- 哈尔滨工业大学计算学部王甜甜老师团队
- 华为 CANN 团队

---

## 📄 开源协议

Apache License, Version 2.0 © 2024-2025 AsNumpy 贡献者

---

<div align="center">

**如果 AsNumpy 对你有帮助，请给我们一个 ⭐ Star！**

</div>

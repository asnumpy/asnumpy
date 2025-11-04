# AsNumpy API 

本文档介绍 AsNumpy 目前已实现的 API 情况。AsNumpy 致力于在昇腾 NPU 上提供与 Numpy 高度兼容的科学计算接口。

---

## 📊 API 完成情况总览

AsNumpy 目前已实现多个核心功能模块的 API，涵盖数学运算、逻辑函数、随机抽样等常用科学计算功能。各模块的 API 实现正在持续完善中。

---

## 📚 已实现的 API 模块

### 🧮 数学运算模块 (Math)

数学运算模块是 AsNumpy 的核心模块之一，提供了丰富的数学计算功能：

- [算术运算 (Arithmetic Operations)](math/arithmetic_operations.md) - 基础算术运算符
- [指数与对数 (Exponents and Logarithms)](math/exponents_and_logarithms.md) - 指数、对数函数
- [三角函数 (Trigonometric Functions)](math/trigonometric_functions.md) - sin, cos, tan 等三角函数
- [双曲函数 (Hyperbolic Functions)](math/hyperbolic_functions.md) - sinh, cosh, tanh 等双曲函数
- [浮点数例程 (Floating Point Routines)](math/floating_point_routines.md) - 浮点数处理函数
- [舍入 (Rounding)](math/rounding.md) - 各类舍入函数
- [求和、乘积、差分 (Sums, Products, Differences)](math/sums_products_differences.md) - 聚合运算
- [复数处理 (Handling Complex Numbers)](math/handling_complex_numbers.md) - 复数运算
- [有理数例程 (Rational Routines)](math/rational_routines.md) - 有理数相关函数
- [特殊函数 (Other Special Functions)](math/other_special_functions.md) - Sinc 等特殊数学函数
- [杂项函数 (Miscellaneous)](math/miscellaneous.md) - Clip、Maximum、Minimum、Square 等工具函数

### 🎲 随机抽样模块 (Random)

随机模块提供各类概率分布的随机数生成功能：

- [概率分布 (Distributions)](random/distributions.md) - 各类概率分布函数

### 🔍 逻辑函数模块 (Logic)

逻辑模块提供数组比较、逻辑运算等功能：

- [逻辑函数 (Logic Functions)](logic/logic.md) - 逻辑运算与比较函数

---

## 🚧 开发进度说明

**总体进展**  
- 目前 AsNumpy 共规划约 **260 个接口**
- 已实现并合并：**113 个**
- 待合并：**10 个**
- 内部 utils：**11 个**

**各模块进展**  
- **数学模块**：当前项目的主要进展，已实现大部分常用数学函数，是目前完成度最高的模块
- **随机模块**：由于复杂性进度相对缓慢，部分分布函数仍在开发中
- **逻辑模块**：已实现基础逻辑运算功能
- **线性代数模块**：在整理后逐步合并中

---

## 📝 说明

- 部分 API 可能对参数 `shape` 有特殊限制，具体请参阅各 API 的详细文档
- 部分 API 功能尚未完全支持，会在文档中标注
- API 文档持续更新中





## Attributes & Methods （属性与方法，TODO）

**计划支持的 API 数量：10**  
**未来计划支持的 API：** [shape](#shape), [dtype](#dtype), [ndim](#ndim), [size](#size), [strides](#strides), [itemsize](#itemsize), [nbytes](#nbytes), [flat](#flat), [real](#real), [imag](#imag)

> 说明：本节为 NPUArray 的属性与方法规划，当前仍处于设计阶段，具体实现与表现形式可能在后续版本中调整。

<span id="shape">1. <mark> **shape** </mark></span>

- **计划属性来源：**  
  `NPUArray.shape`（只读属性）

- **计划类型与含义：**  
  类型为整型元组；返回数组各维度长度，用于描述数组结构。

- **计划功能描述：**  
  提供数组基本形状信息，是后续运算与检查的基础属性。

---

<span id="dtype">2. <mark> **dtype** </mark></span>

- **计划属性来源：**  
  `NPUArray.dtype`（只读属性）

- **计划类型与含义：**  
  类型描述对象；表示数组元素在 NPU 上的存储与计算数据类型。

- **计划功能描述：**  
  用于查询与判断数组元素类型，指导算子选择与类型转换。

---

<span id="ndim">3. <mark> **ndim** </mark></span>

- **计划属性来源：**  
  `NPUArray.ndim`（只读属性）

- **计划类型与含义：**  
  整型标量；等于 `shape` 的长度。

- **计划功能描述：**  
  直接返回数组维度个数，便于快速分支控制。

---

<span id="size">4. <mark> **size** </mark></span>

- **计划属性来源：**  
  `NPUArray.size`（只读属性）

- **计划类型与含义：**  
  整型标量；所有维度长度乘积，即数组元素总数。

- **计划功能描述：**  
  查询数组包含的元素数量，用于内存估算与循环控制。

---

<span id="strides">5. <mark> **strides** </mark></span>

- **计划属性来源：**  
  `NPUArray.strides`（只读属性）

- **计划类型与含义：**  
  整型元组；按字节给出各维度相邻元素在内存中的步长。

- **计划功能描述：**  
  描述数组在 NPU 内存中的布局，用于索引计算与视图操作。

---

<span id="itemsize">6. <mark> **itemsize** </mark></span>

- **计划属性来源：**  
  `NPUArray.itemsize`（只读属性）

- **计划类型与含义：**  
  整型标量；单个元素占用的字节数，由 `dtype` 决定。

- **计划功能描述：**  
  结合 `size` 估算数组总字节大小，用于内存管理。

---

<span id="nbytes">7. <mark> **nbytes** </mark></span>

- **计划属性来源：**  
  `NPUArray.nbytes`（只读属性）

- **计划类型与含义：**  
  整型标量；数组实际占用的总字节数，约等于 `size * itemsize`。

- **计划功能描述：**  
  直接给出内存占用大小，方便做显存监控与优化。

---

<span id="flat">8. <mark> **flat** </mark></span>

- **计划属性来源：**  
  `NPUArray.flat`（可迭代视图）

- **计划类型与含义：**  
  一维迭代器对象；顺序访问数组的所有元素。

- **计划功能描述：**  
  提供统一的一维遍历接口，便于逐元素操作与调试。

---

<span id="real">9. <mark> **real** </mark></span>

- **计划属性来源：**  
  `NPUArray.real`（只读视图或副本）

- **计划类型与含义：**  
  数组；形状与原数组一致，包含每个元素的实部。

- **计划功能描述：**  
  从复数数组中提取实部，也兼容实数数组的直接访问。

---

<span id="imag">10. <mark> **imag** </mark></span>

- **计划属性来源：**  
  `NPUArray.imag`（只读视图或副本）

- **计划类型与含义：**  
  数组；形状与原数组一致，包含每个元素的虚部。

- **计划功能描述：**  
  从复数数组中提取虚部，用于复数分解与分析。

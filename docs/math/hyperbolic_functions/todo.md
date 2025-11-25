## Hyperbolic Functions （双曲函数，TODO）

**计划支持的 API 数量：3**  
**未来计划支持的 API：** [asinh](#asinh), [acosh](#acosh), [atanh](#atanh)

> 说明：本节为反双曲函数相关的逐元素运算规划，当前仍处于设计阶段，具体实现可能在后续版本中调整。

---

<span id="asinh">1. <mark> **asinh** </mark></span>

- **计划参数：**  
  x：输入数组或标量；  
  out：可选，输出数组。

- **计划返回类型：**  
  NPUArray（与 x 广播后形状一致）

- **计划支持的 dtype：**  
  浮点或整型（整型会在内部提升为浮点计算）。

- **计划功能描述：**  
  逐元素计算反双曲正弦 `asinh(x)`。

---

<span id="acosh">2. <mark> **acosh** </mark></span>

- **计划参数：**  
  x：输入数组或标量（通常要求 x ≥ 1）；  
  out：可选，输出数组。

- **计划返回类型：**  
  NPUArray（浮点）

- **计划支持的 dtype：**  
  浮点或整型（整型会在内部提升为浮点计算）。

- **计划功能描述：**  
  逐元素计算反双曲余弦 `acosh(x)`。

---

<span id="atanh">3. <mark> **atanh** </mark></span>

- **计划参数：**  
  x：输入数组或标量（通常要求 |x| < 1）；  
  out：可选，输出数组。

- **计划返回类型：**  
  NPUArray（浮点）

- **计划支持的 dtype：**  
  浮点或整型（整型会在内部提升为浮点计算）。

- **计划功能描述：**  
  逐元素计算反双曲正切 `atanh(x)`。

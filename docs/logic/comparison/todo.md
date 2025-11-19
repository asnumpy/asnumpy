## Comparison （比较函数，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [allclose](#allclose), [isclose](#isclose), [array_equal](#array_equal), [array_equiv](#array_equiv)

> 说明：本节用于比较两个数组在数值或形状上的相等与“近似相等”，当前仍处于设计阶段，具体实现可能在后续版本中调整。

---

<span id="allclose">1. <mark> **allclose** </mark></span>

- **计划参数：**  
  a, b：待比较的输入数组；  
  rtol：相对误差容忍度；  
  atol：绝对误差容忍度；  
  equal_nan：是否将 NaN 视为相等。

- **计划返回类型：**  
  bool 标量。

- **计划功能描述：**  
  判断两个数组在给定容差下是否“足够接近”（值级判断，支持广播）。

---

<span id="isclose">2. <mark> **isclose** </mark></span>

- **计划参数：**  
  a, b：待比较的输入数组；  
  rtol：相对误差容忍度；  
  atol：绝对误差容忍度；  
  equal_nan：是否将 NaN 视为相等。

- **计划返回类型：**  
  NPUArray（bool 型，形状为广播后结果）

- **计划功能描述：**  
  逐元素判断两个数组是否在容差范围内“接近”，支持广播。

---

<span id="array_equal">3. <mark> **array_equal** </mark></span>

- **计划参数：**  
  a, b：待比较的数组；  
  equal_nan：是否将 NaN 视为相等。

- **计划返回类型：**  
  bool 标量。

- **计划功能描述：**  
  判断两个数组形状完全相同且所有元素逐一相等。

---

<span id="array_equiv">4. <mark> **array_equiv** </mark></span>

- **计划参数：**  
  a, b：待比较的数组。

- **计划返回类型：**  
  bool 标量。

- **计划功能描述：**  
  判断两个数组在形状兼容且广播后，所有元素是否逐一相等。

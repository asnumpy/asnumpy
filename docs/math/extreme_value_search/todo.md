## Extreme Value Search （极值查找，TODO）

**计划支持的 API 数量：6**  
**未来计划支持的 API：** [max](#max), [amax](#amax), [nanmax](#nanmax), [min](#min), [amin](#amin), [nanmin](#nanmin)

> 说明：本节为逐元素极值与按轴极值计算的规划，当前仍处于设计阶段，具体实现细节可能在后续版本中调整。

---

<span id="max">1. <mark> **max** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选，沿指定轴求最大值；  
  keepdims：可选，是否保留被约简的轴。

- **计划返回类型：**  
  NPUArray 或标量。

- **计划支持的 dtype：**  
  浮点或整数。

- **计划功能描述：**  
  计算数组整体或按轴的最大值。

---

<span id="amax">2. <mark> **amax** </mark></span>

- **计划说明：**  
  与 `max` 接口保持一致，仅作为别名存在，方便兼容 NumPy 习惯。

---

<span id="nanmax">3. <mark> **nanmax** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选；  
  keepdims：可选。

- **计划返回类型：**  
  NPUArray 或标量。

- **计划支持的 dtype：**  
  浮点或整数。

- **计划功能描述：**  
  忽略 `NaN`，在剩余元素中计算最大值；若全为 `NaN`，行为后续约定。

---

<span id="min">4. <mark> **min** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选；  
  keepdims：可选。

- **计划返回类型：**  
  NPUArray 或标量。

- **计划支持的 dtype：**  
  浮点或整数。

- **计划功能描述：**  
  计算数组整体或按轴的最小值。

---

<span id="amin">5. <mark> **amin** </mark></span>

- **计划说明：**  
  与 `min` 接口保持一致，仅作为别名存在，方便兼容 NumPy 习惯。

---

<span id="nanmin">6. <mark> **nanmin** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选；  
  keepdims：可选。

- **计划返回类型：**  
  NPUArray 或标量。

- **计划支持的 dtype：**  
  浮点或整数。

- **计划功能描述：**  
  忽略 `NaN`，在剩余元素中计算最小值。

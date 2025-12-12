## Array Contents （数组内容，TODO）

**计划支持的 API 数量：2**  
**未来计划支持的 API：** [isnan](#isnan), [isnat](#isnat)

> 说明：本节主要用于检查数组中元素内容是否为缺失或非法值，当前仍处于设计阶段，具体实现可能在后续版本中调整。

---

<span id="isnan">1. <mark> **isnan** </mark></span>

- **计划参数：**  
  x：输入数组或标量，dtype 可为任意数值或复数类型。

- **计划返回类型：**  
  NPUArray（bool 型，形状与 x 相同）

- **计划功能描述：**  
  逐元素判断是否为 IEEE 754 NaN，用于数据清洗与异常检测。

---

<span id="isnat">2. <mark> **isnat** </mark></span>

- **计划参数：**  
  x：输入数组或标量，dtype 为 datetime64 或 timedelta64。

- **计划返回类型：**  
  NPUArray（bool 型，形状与 x 相同）

- **计划功能描述：**  
  逐元素判断是否为 NaT（not-a-time），用于检测缺失时间数据。

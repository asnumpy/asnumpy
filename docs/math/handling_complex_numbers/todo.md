## Handling Complex Numbers （复数处理函数，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [angle](#angle), [imag](#imag), [conj](#conj), [conjugate](#conjugate)

> 说明：本节为复数数组的相位、虚部与共轭等操作规划，当前仍处于设计阶段，具体实现可能在后续版本中调整。

---

<span id="angle">1. <mark> **angle** </mark></span>

- **计划参数：**  
  z：复数数组或标量；  
  deg：可选，是否以角度（度）返回，默认 False 表示弧度。

- **计划返回类型：**  
  NPUArray（浮点，形状与 z 相同）

- **计划功能描述：**  
  返回每个复数的辐角（相位）。

---

<span id="imag">2. <mark> **imag** </mark></span>

- **计划参数：**  
  z：复数数组或标量。

- **计划返回类型：**  
  NPUArray（浮点或整数，形状与 z 相同）

- **计划功能描述：**  
  提取复数的虚部，对实数输入返回 0。

---

<span id="conj">3. <mark> **conj** </mark></span>

- **计划参数：**  
  z：复数或实数数组。

- **计划返回类型：**  
  NPUArray（与 z 同形状、同 dtype）

- **计划功能描述：**  
  返回逐元素共轭结果，实数保持不变。

---

<span id="conjugate">4. <mark> **conjugate** </mark></span>

- **计划说明：**  
  接口行为与 `conj` 完全一致，作为别名提供，便于兼容 NumPy 使用习惯。

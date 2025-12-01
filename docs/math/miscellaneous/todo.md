## Miscellaneous Math Functions （杂项数学函数，TODO）

**计划支持的 API 数量：6**  
**未来计划支持的 API：** [convolve](#convolve), [sqrt](#sqrt), [cbrt](#cbrt), [absolute](#absolute), [real_if_close](#real_if_close), [interp](#interp)

> 说明：本节为一些常用但分类较杂的数学工具函数规划，当前仍处于设计阶段，具体实现可能在后续版本中调整。

---

<span id="convolve">1. <mark> **convolve** </mark></span>

- **计划参数：**  
  a：一维输入序列；  
  v：一维卷积核序列；  
  mode：可选，"full"、"same" 或 "valid"。

- **计划返回类型：**  
  NPUArray（一维）

- **计划功能描述：**  
  计算两个一维序列的离散线性卷积。

---

<span id="sqrt">2. <mark> **sqrt** </mark></span>

- **计划参数：**  
  x：输入数组或标量；  
  out：可选，输出数组。

- **计划返回类型：**  
  NPUArray

- **计划功能描述：**  
  逐元素计算平方根，对负数输入可扩展为复数结果。

---

<span id="cbrt">3. <mark> **cbrt** </mark></span>

- **计划参数：**  
  x：输入数组或标量。

- **计划返回类型：**  
  NPUArray

- **计划功能描述：**  
  逐元素计算立方根，对负数输入保持实数结果。

---

<span id="absolute">4. <mark> **absolute** </mark></span>

- **计划参数：**  
  x：实数或复数数组。

- **计划返回类型：**  
  NPUArray

- **计划功能描述：**  
  实数组返回绝对值；复数组返回模长 √(a²+b²)。

---

<span id="real_if_close">5. <mark> **real_if_close** </mark></span>

- **计划参数：**  
  x：实数或复数数组；  
  tol：可选，接近于零的阈值系数。

- **计划返回类型：**  
  NPUArray（实数或复数）

- **计划功能描述：**  
  若复数虚部在给定阈值内接近 0，则转为实数数组。

---

<span id="interp">6. <mark> **interp** </mark></span>

- **计划参数：**  
  x：需要插值的位置；  
  xp：已知自变量样本点（一维递增）；  
  fp：与 xp 对应的函数值。

- **计划返回类型：**  
  NPUArray（一维或广播后形状）

- **计划功能描述：**  
  对给定样本点做一维线性插值，并对 x 进行逐点评估。

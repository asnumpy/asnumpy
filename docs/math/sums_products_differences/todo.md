## Sums, Products & Differences （求和、乘积与差分函数，TODO）

**计划支持的 API 数量：7**  
**未来计划支持的 API：**  
[cumulative_prod](#cumulative_prod), [cumulative_sum](#cumulative_sum), [diff](#diff), [ediff1d](#ediff1d), [gradient](#gradient), [cross](#cross), [trapezoid](#trapezoid)

> 说明：本节为数组沿轴的累积运算、差分与梯度等函数规划，当前仍处于设计阶段，具体实现可能在后续版本中调整。

---

<span id="cumulative_prod">1. <mark> **cumulative_prod** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选，累积乘积的轴；  
  dtype：可选，运算使用的数据类型；  
  out：可选，输出数组。

- **计划返回类型：**  
  NPUArray（形状与输入相同）

- **计划功能描述：**  
  沿指定轴计算元素的累积乘积。

---

<span id="cumulative_sum">2. <mark> **cumulative_sum** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选，累积求和的轴；  
  dtype：可选；  
  out：可选。

- **计划返回类型：**  
  NPUArray（形状与输入相同）

- **计划功能描述：**  
  沿指定轴计算元素的累积和。

---

<span id="diff">3. <mark> **diff** </mark></span>

- **计划参数：**  
  a：输入数组；  
  n：差分次数，默认 1；  
  axis：可选，进行差分的轴。

- **计划返回类型：**  
  NPUArray（在 axis 上长度减少 n）

- **计划功能描述：**  
  计算相邻元素差值，用于离散导数近似等场景。

---

<span id="ediff1d">4. <mark> **ediff1d** </mark></span>

- **计划参数：**  
  a：一维输入数组；  
  to_begin：可选，前缀元素；  
  to_end：可选，后缀元素。

- **计划返回类型：**  
  NPUArray（一维）

- **计划功能描述：**  
  返回一维数组相邻元素差分，可选在首尾拼接额外元素。

---

<span id="gradient">5. <mark> **gradient** </mark></span>

- **计划参数：**  
  f：输入 N 维数组；  
  spacing：可选，网格步长或坐标；  
  axis：可选，计算梯度的轴或轴序列。

- **计划返回类型：**  
  NPUArray 或 NPUArray 元组（每个轴一个梯度分量）

- **计划功能描述：**  
  使用有限差分近似计算数组的数值梯度。

---

<span id="cross">6. <mark> **cross** </mark></span>

- **计划参数：**  
  a：输入向量或向量数组；  
  b：第二个向量或向量数组；  
  axis：可选，向量所在的轴。

- **计划返回类型：**  
  NPUArray（与输入广播后形状一致）

- **计划功能描述：**  
  计算三维向量的叉积。

---

<span id="trapezoid">7. <mark> **trapezoid** </mark></span>

- **计划参数：**  
  y：待积分的函数值数组；  
  x：可选，自变量网格；  
  dx：可选，均匀网格的步长。

- **计划返回类型：**  
  NPUArray 或标量。

- **计划功能描述：**  
  使用复合梯形公式对数据进行一维数值积分。

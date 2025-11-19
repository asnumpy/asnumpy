
## Interpolation （插值算法，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [interp](#interp), [piecewise](#piecewise), [spline](#spline), [pchip](#pchip)

> 说明：本节提供一维插值及分段插值接口，包括线性插值、样条插值与单调保持插值，用于在离散数据点之间构造平滑或保形的函数。

---

<span id="interp">1. <mark> **interp** </mark></span>

- **计划参数：**  
  x：插值位置，一维数组或广播兼容的 NPUArray；  
  xp：已知自变量网格，一维单调数组；  
  fp：对应的函数值数组，shape 与 xp 兼容；  
  left / right：可选，外推时在区间外侧的填充值；  
  period：可选，周期性插值的周期。
- **计划返回类型：**  
  NPUArray，shape 与 x 相同。
- **计划功能描述：**  
  对一维数据执行线性插值，在 xp 给定的节点之间按线性段连续地估计中间值。

---

<span id="piecewise">2. <mark> **piecewise** </mark></span>

- **计划参数：**  
  x：输入自变量数组，维度 1–3；  
  condlist：条件数组或布尔表达式列表，每个元素与 x 形状兼容；  
  funclist：与 condlist 等长的函数或标量列表，用于定义各区间上的函数形式；  
  default：可选，默认值或默认函数。
- **计划返回类型：**  
  NPUArray，与 x 形状一致。
- **计划功能描述：**  
  按条件列表将定义域拆分为多个子区间，并在各区间上应用不同的函数，实现灵活的分段函数/插值构造。

---

<span id="spline">3. <mark> **spline** </mark></span>

- **计划参数：**  
  x：节点自变量数组，一维递增序列；  
  y：节点函数值数组；  
  order：可选样条阶数，默认 3（立方样条）；  
  bc_type：可选，边界条件类型（自然、夹持等）；  
  axis：可选，y 中作为样条维度的轴。
- **计划返回类型：**  
  一个样条对象或其系数表示，可与专用 `splev`/`spline_eval` 等接口配合使用。
- **计划功能描述：**  
  构造满足一定光滑条件的多段多项式（通常为三次样条），在各节点处函数值相等并在内部节点处保证一阶/二阶导数连续。

---

<span id="pchip">4. <mark> **pchip** </mark></span>

- **计划参数：**  
  x：已排序的一维节点数组；  
  y：对应函数值数组；  
  axis：可选，用于多通道数据时指定插值维度；  
  extrapolate：bool，控制区间外是否允许外推。
- **计划返回类型：**  
  单调保持样条插值对象，可在任意点上调用其 `__call__` 或专用评估函数获取插值值。
- **计划功能描述：**  
  实现 PCHIP（Piecewise Cubic Hermite Interpolating Polynomial）单调保持插值，在保持插值曲线形状与单调性的同时提供较好的平滑性，适合对峰谷和非振荡要求较高的场景。

## Indexing & Copy （索引与复制，TODO）

**计划支持的 API 数量：10**  
**未来计划支持的 API：**  
[getitem](#getitem), [setitem](#setitem), [take](#take), [put](#put), [copyto](#copyto), [astype](#astype), [view](#view), [item](#item), [tolist](#tolist), [fill](#fill)

> 说明：本节为 NPUArray 的索引、复制与类型转换等操作规划，当前仍在设计阶段，具体实现可能调整。

---

<span id="getitem">1. <mark> **getitem** </mark></span>

- **计划参数：**  
  a：输入数组；  
  index：整数、切片或多维索引。

- **计划返回类型：**  
  NPUArray 或标量。

- **计划功能描述：**  
  根据索引获取子数组或单个元素，支持切片与多维访问。

---

<span id="setitem">2. <mark> **setitem** </mark></span>

- **计划参数：**  
  a：输入数组；  
  index：索引或切片；  
  value：要写入的值。

- **计划功能描述：**  
  根据索引修改数组局部元素，支持广播赋值。

---

<span id="take">3. <mark> **take** </mark></span>

- **计划参数：**  
  a：输入数组；  
  indices：索引数组；  
  axis：可选，默认 0。

- **计划返回类型：**  
  NPUArray

- **计划功能描述：**  
  从指定位置提取元素，等价于按索引收集。

---

<span id="put">4. <mark> **put** </mark></span>

- **计划参数：**  
  a：输入数组；  
  indices：索引数组；  
  values：写入值。

- **计划功能描述：**  
  将元素放置到给定索引位置，常用于 inplace 更新。

---

<span id="copyto">5. <mark> **copyto** </mark></span>

- **计划参数：**  
  dst：目标数组；  
  src：源数组。

- **计划功能描述：**  
  将源内容复制到目标，支持设备到设备拷贝。

---

<span id="astype">6. <mark> **astype** </mark></span>

- **计划参数：**  
  dtype：目标数据类型；  
  copy：可选，默认 True。

- **计划功能描述：**  
  执行数组类型转换，必要时调用后端 Cast。

---

<span id="view">7. <mark> **view** </mark></span>

- **计划参数：**  
  dtype：新的视图类型。

- **计划返回类型：**  
  新的 NPUArray 视图（不复制数据）。

- **计划功能描述：**  
  按新 dtype 重新解释底层内存，不重新分配。

---

<span id="item">8. <mark> **item** </mark></span>

- **计划功能描述：**  
  返回数组中单个 Python 标量（需从设备复制回主机）。

---

<span id="tolist">9. <mark> **tolist** </mark></span>

- **计划功能描述：**  
  将 NPUArray 完整转换为 Python 嵌套 list（主机侧执行）。

---

<span id="fill">10. <mark> **fill** </mark></span>

- **计划参数：**  
  value：标量数值。

- **计划功能描述：**  
  将数组整体填充为指定值，用于初始化。

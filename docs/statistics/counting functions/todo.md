
## Counting Functions （计数与索引函数，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [count_nonzero](#count_nonzero), [nonzero](#nonzero), [argwhere](#argwhere), [flatnonzero](#flatnonzero)

> 说明：本节提供基于条件的计数与位置索引接口，用于统计非零元素数量或定位满足条件的元素坐标。

---

<span id="count_nonzero">1. <mark> **count_nonzero** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为布尔或整数/浮点；  
  axis：可选，int 或 tuple[int]，指定沿哪个维度统计非零数目；  
  keepdims：bool。
- **计划返回类型：**  
  NPUArray（整数），形状为按 axis 归约后的结果。
- **计划功能描述：**  
  统计数组中非零（或 True）元素的数量，支持按轴统计。

---

<span id="nonzero">2. <mark> **nonzero** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 任意；  
- **计划返回类型：**  
  一个长度为 ndim 的元组，每个元素为 1D NPUArray，表示相应维度上非零元素的索引。
- **计划功能描述：**  
  返回数组中所有非零元素的位置索引，是稀疏数据和条件过滤常用的基础工具。

---

<span id="argwhere">3. <mark> **argwhere** </mark></span>

- **计划参数：**  
  a：输入数组，通常为布尔型或条件表达式结果；  
- **计划返回类型：**  
  NPUArray，形状为 (K, ndim)，其中 K 为满足条件的元素个数，每行给出一个坐标。
- **计划功能描述：**  
  将非零（或 True）元素的位置打平为二维坐标列表，便于遍历或进一步索引。

---

<span id="flatnonzero">4. <mark> **flatnonzero** </mark></span>

- **计划参数：**  
  a：输入数组；  
- **计划返回类型：**  
  一维 NPUArray（整数），为在扁平化后的数组上非零元素的索引。
- **计划功能描述：**  
  在数组扁平化展开后返回所有非零元素的一维下标，适用于需要线性索引的场景。

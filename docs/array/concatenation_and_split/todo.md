## Concatenation & Split （拼接与分割，TODO）

**计划支持的 API 数量：10**  
**未来计划支持的 API：** [concatenate](#concatenate), [stack](#stack), [vstack](#vstack), [hstack](#hstack), [dstack](#dstack), [split](#split), [array_split](#array_split), [hsplit](#hsplit), [vsplit](#vsplit), [dsplit](#dsplit)

> 说明：本节接口均为未来计划支持的功能，当前仍处于设计阶段，具体实现与参数细节可能在后续版本中调整。

<span id="concatenate">1. <mark> **concatenate** </mark></span>

- **计划参数：**  
  seq：数组序列；  
  axis：可选，要拼接的轴，默认 0。

- **计划返回类型：**  
  NPUArray

- **计划支持的 dtype：**  
  保持输入数组类型一致。

- **计划 shape 特征：**  
  多个数组在给定轴上长度相加，其他轴形状相同。

- **计划功能描述：**  
  沿现有轴将多个数组直接拼接在一起。

---

<span id="stack">2. <mark> **stack** </mark></span>

- **计划参数：**  
  seq：数组序列，形状相同；  
  axis：可选，新插入轴位置，默认 0。

- **计划返回类型：**  
  NPUArray

- **计划支持的 dtype：**  
  与输入数组一致。

- **计划 shape 特征：**  
  在新轴上堆叠多个数组，结果维度数比输入多 1。

- **计划功能描述：**  
  在新维度上将多组数据打包成一个更高维数组。

---

<span id="vstack">3. <mark> **vstack** </mark></span>

- **计划参数：**  
  seq：数组序列；  

- **计划返回类型：**  
  NPUArray

- **计划支持的 dtype：**  
  与输入一致。

- **计划 shape 特征：**  
  按行方向（axis=0）堆叠，行数相加。

- **计划功能描述：**  
  以“竖直方向”将多个数组按行拼接。

---

<span id="hstack">4. <mark> **hstack** </mark></span>

- **计划参数：**  
  seq：数组序列；  

- **计划返回类型：**  
  NPUArray

- **计划支持的 dtype：**  
  与输入一致。

- **计划 shape 特征：**  
  对二维数组按列方向（axis=1）堆叠，列数相加。

- **计划功能描述：**  
  以“水平方向”将多个数组按列拼接。

---

<span id="dstack">5. <mark> **dstack** </mark></span>

- **计划参数：**  
  seq：数组序列；  

- **计划返回类型：**  
  NPUArray

- **计划支持的 dtype：**  
  与输入一致。

- **计划 shape 特征：**  
  沿深度方向（axis=2）堆叠形成三维结构。

- **计划功能描述：**  
  将二维数组在第三维上叠加成“层”。

---

<span id="split">6. <mark> **split** </mark></span>

- **计划参数：**  
  a：输入数组；  
  indices_or_sections：整型或切分索引；  
  axis：可选，要切分的轴，默认 0。

- **计划返回类型：**  
  NPUArray 列表

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  沿给定轴等分数组，各块形状相同。

- **计划功能描述：**  
  将数组按等长方式拆分为多个子数组。

---

<span id="array_split">7. <mark> **array_split** </mark></span>

- **计划参数：**  
  a：输入数组；  
  indices_or_sections：整型或切分索引；  
  axis：可选，要切分的轴，默认 0。

- **计划返回类型：**  
  NPUArray 列表

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  沿给定轴进行不完全等分，各块长度可以相差 1。

- **计划功能描述：**  
  对无法整除的情况进行“近似平均”的切分。

---

<span id="hsplit">8. <mark> **hsplit** </mark></span>

- **计划参数：**  
  a：输入数组；  
  indices_or_sections：切分参数。

- **计划返回类型：**  
  NPUArray 列表

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  沿列方向（axis=1）切分二维数组。

- **计划功能描述：**  
  按列把数组拆分成若干块。

---

<span id="vsplit">9. <mark> **vsplit** </mark></span>

- **计划参数：**  
  a：输入数组；  
  indices_or_sections：切分参数。

- **计划返回类型：**  
  NPUArray 列表

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  沿行方向（axis=0）切分二维数组。

- **计划功能描述：**  
  按行把数组拆分成若干块。

---

<span id="dsplit">10. <mark> **dsplit** </mark></span>

- **计划参数：**  
  a：输入数组（三维）；  
  indices_or_sections：切分参数。

- **计划返回类型：**  
  NPUArray 列表

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  沿深度方向（axis=2）切分三维数组。

- **计划功能描述：**  
  在第三维上把三维数组拆成多层。

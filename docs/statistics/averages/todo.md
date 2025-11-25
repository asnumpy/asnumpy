
## Averages （平均与方差类函数，TODO）

**计划支持的 API 数量：6**  
**未来计划支持的 API：** [mean](#mean), [var](#var), [std](#std), [average](#average), [median](#median), [mean_absolute_deviation](#mean_absolute_deviation)

> 说明：本节提供各类平均值、方差及相关统计量的计算接口，统一支持按 axis 聚合，适用于 1–4 维张量的基础统计分析。

---

<span id="mean">1. <mark> **mean** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为浮点或可提升为浮点；  
  axis：可选，int 或 tuple[int]，指定归约维度，默认对所有元素求平均；  
  keepdims：bool，是否保留被归约轴为长度 1；  
  dtype：可选，输出计算使用的精度，默认跟随输入或提升为 `float32/float64`。
- **计划返回类型：**  
  NPUArray（浮点型），shape 为按 axis 归约后的形状。
- **计划功能描述：**  
  计算输入数组的算术平均值，是后续方差、标准差等统计量的基础。

---

<span id="var">2. <mark> **var** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为浮点；  
  axis：可选，int 或 tuple[int]，指定方差计算的归约轴；  
  ddof：自由度修正参数，默认 0，对应总体方差；  
  keepdims：bool，是否保留被归约轴；  
  dtype：可选，计算精度。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  计算输入数组在指定轴上的方差，度量数据围绕均值的离散程度。

---

<span id="std">3. <mark> **std** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为浮点；  
  axis：可选，指定归约轴；  
  ddof：自由度修正参数；  
  keepdims：bool；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  计算标准差，即方差的平方根，用于衡量数据波动幅度。

---

<span id="average">4. <mark> **average** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为浮点；  
  axis：可选，指定加权平均的归约轴；  
  weights：可选，与 a 在 axis 上可广播的权重数组；  
  returned：bool，若为 True，则同时返回权重和；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray（浮点型），或 (average, sum_of_weights) 二元组。
- **计划功能描述：**  
  计算加权平均值，支持指定不同元素的重要性，用于统计加权指标或样本权重场景。

---

<span id="median">5. <mark> **median** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为浮点；  
  axis：可选，指定沿哪个维度求中位数；  
  keepdims：bool；  
  overwrite_input：可选，是否允许在原数组上进行部分重排以节省内存。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  计算沿指定轴的中位数（按排序后的中间位置取值），对极端值更鲁棒。

---

<span id="mean_absolute_deviation">6. <mark> **mean_absolute_deviation** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为浮点；  
  axis：可选，指定归约轴；  
  center：可选，中心值，默认使用 mean；  
  keepdims：bool。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  计算平均绝对偏差（Mean Absolute Deviation, MAD 的一种形式），即 |x - center| 的平均值，用于衡量数据的稳健离散程度。

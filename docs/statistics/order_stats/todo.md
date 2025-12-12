
## Order Statistics （排序与分位数函数，TODO）

**计划支持的 API 数量：5**  
**未来计划支持的 API：** [percentile](#percentile), [quantile](#quantile), [nanpercentile](#nanpercentile), [nanquantile](#nanquantile), [median_absolute_deviation](#median_absolute_deviation)

> 说明：本节实现按给定分位数位置提取统计量的接口，支持常规与 NaN 安全版本，用于描述数据分布的尾部和中位特征。

---

<span id="percentile">1. <mark> **percentile** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为浮点；  
  q：标量或数组，取值范围 [0, 100]，表示百分位数；  
  axis：可选，指定归约轴；  
  interpolation / method：可选，分位数插值方式；  
  keepdims：bool。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  按百分位数计算分位值，例如 q=50 对应中位数，q=25/75 对应四分位数。

---

<span id="quantile">2. <mark> **quantile** </mark></span>

- **计划参数：**  
  a：输入数组；  
  q：标量或数组，取值范围 [0, 1]，表示分位点；  
  axis：可选；  
  interpolation / method：可选；  
  keepdims：bool。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  与 `percentile` 类似，但 q 使用 [0,1] 范围的分位值表示，更方便与概率语义对接。

---

<span id="nanpercentile">3. <mark> **nanpercentile** </mark></span>

- **计划参数：**  
  a：输入数组，允许包含 NaN；  
  q：百分位数 [0, 100]；  
  axis：可选；  
  keepdims：bool；  
  interpolation / method：可选。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  与 `percentile` 一致，但在计算时忽略 NaN 元素，仅对非 NaN 数据求分位数。

---

<span id="nanquantile">4. <mark> **nanquantile** </mark></span>

- **计划参数：**  
  a：输入数组，允许包含 NaN；  
  q：分位点 [0,1]；  
  axis：可选；  
  keepdims：bool；  
  interpolation / method：可选。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  NaN 安全版本的 `quantile`，在存在缺失值的数据集上计算稳健的分位统计量。

---

<span id="median_absolute_deviation">5. <mark> **median_absolute_deviation** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选，指定归约维度；  
  center：可选，默认使用 median；  
  scale：可选，常用 √2·erfc⁻¹(1/2) 等缩放系数以使得在正态分布下与标准差可比；  
  keepdims：bool。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  计算基于中位数的绝对偏差（MAD），对异常值高度鲁棒，用于稳健性统计和异常检测。

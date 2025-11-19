
## NaN-safe Reductions （NaN 安全聚合函数，TODO）

**计划支持的 API 数量：7**  
**未来计划支持的 API：** [nanmean](#nanmean), [nanvar](#nanvar), [nanstd](#nanstd), [nanmin](#nanmin), [nanmax](#nanmax), [nansum](#nansum), [nanprod](#nanprod)

> 说明：本节的接口在聚合运算时自动忽略 NaN，用于处理包含缺失值的张量，避免手动掩码操作。

---

<span id="nanmean">1. <mark> **nanmean** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为浮点，可包含 NaN；  
  axis：可选，指定归约轴；  
  keepdims：bool；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  计算忽略 NaN 后的算术平均值；若某一归约切片全部为 NaN，则结果为 NaN。

---

<span id="nanvar">2. <mark> **nanvar** </mark></span>

- **计划参数：**  
  a：输入数组，可包含 NaN；  
  axis：可选；  
  ddof：自由度修正；  
  keepdims：bool；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  计算忽略 NaN 后的方差，用于缺失值下的波动性度量。

---

<span id="nanstd">3. <mark> **nanstd** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选；  
  ddof：可选；  
  keepdims：bool；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  在忽略 NaN 的前提下计算标准差。

---

<span id="nanmin">4. <mark> **nanmin** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选；  
  keepdims：bool。
- **计划返回类型：**  
  NPUArray。
- **计划功能描述：**  
  计算沿指定轴的最小值，自动忽略 NaN；如果某切片全为 NaN，则结果为 NaN。

---

<span id="nanmax">5. <mark> **nanmax** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选；  
  keepdims：bool。
- **计划返回类型：**  
  NPUArray。
- **计划功能描述：**  
  与 `nanmin` 类似，但计算最大值。

---

<span id="nansum">6. <mark> **nansum** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选；  
  keepdims：bool；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray。
- **计划功能描述：**  
  忽略 NaN 的求和操作，可用于简单聚合统计而不受缺失值影响。

---

<span id="nanprod">7. <mark> **nanprod** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选；  
  keepdims：bool；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray。
- **计划功能描述：**  
  忽略 NaN 的连乘操作，适用于概率连乘、比率链式计算等。

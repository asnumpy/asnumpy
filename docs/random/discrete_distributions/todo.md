
## Discrete Distributions （离散分布采样，TODO）

**计划支持的 API 数量：7**  
**未来计划支持的 API：** [binomial](#binomial), [poisson](#poisson), [geometric](#geometric), [multinomial](#multinomial), [hypergeometric](#hypergeometric), [negative_binomial](#negative_binomial), [randint](#randint)

> 说明：本节覆盖常见离散概率分布的随机采样接口，统一返回整型数组，支持 1–4 维输出。主要用于计数数据建模、事件模拟与组合抽样等场景。

---

<span id="binomial">1. <mark> **binomial** </mark></span>

- **计划参数：**  
  n：整数标量或可广播数组，表示独立试验次数（n ≥ 0）；  
  p：浮点标量或可广播数组，表示单次试验成功概率（0 ≤ p ≤ 1）；  
  size：整数或整型元组，指定输出形状，支持维度 1–4；  
  dtype：整型类型，暂定为 `int32` / `int64`。
- **计划返回类型：**  
  NPUArray（整型）。
- **计划功能描述：**  
  生成服从二项分布 Binomial(n, p) 的随机样本，可用于建模固定次数伯努利试验中的成功次数。

---

<span id="poisson">2. <mark> **poisson** </mark></span>

- **计划参数：**  
  lam：浮点标量或可广播数组，表示泊松分布的强度参数 λ (> 0)；  
  size：整数或整型元组，指定输出形状，支持维度 1–4；  
  dtype：整型类型，暂定为 `int32` / `int64`。
- **计划返回类型：**  
  NPUArray（整型）。
- **计划功能描述：**  
  生成服从泊松分布 Poisson(λ) 的非负整数样本，常用于建模单位时间或单位区域内事件发生次数。

---

<span id="geometric">3. <mark> **geometric** </mark></span>

- **计划参数：**  
  p：浮点标量或可广播数组，表示每次试验成功概率（0 < p ≤ 1）；  
  size：整数或整型元组，指定输出形状，支持维度 1–4；  
  dtype：整型类型，暂定为 `int32` / `int64`。
- **计划返回类型：**  
  NPUArray（整型）。
- **计划功能描述：**  
  生成服从几何分布的随机样本，表示直到首次成功所需的试验次数，适用于“重试直到成功”的场景建模。

---

<span id="multinomial">4. <mark> **multinomial** </mark></span>

- **计划参数：**  
  n：整数标量，表示总试验次数；  
  pvals：一维概率向量或二维概率矩阵，表示各类别事件发生概率，最后一维长度为 k，并要求各行元素和为 1；  
  size：可选，整数或整型元组，用于广播生成批量样本（与 pvals 批次维度对齐）；  
  dtype：整型类型，暂定为 `int32` / `int64`。
- **计划返回类型：**  
  NPUArray（整型），形状一般为 (..., k)，对应每个类别的计数。
- **计划功能描述：**  
  生成服从多项分布的随机样本，在给定总次数 n 和类别概率 pvals 的条件下，返回各类别出现次数，常用于分类统计与抽签模拟。

---

<span id="hypergeometric">5. <mark> **hypergeometric** </mark></span>

- **计划参数：**  
  ngood：整数标量或可广播数组，总体中成功元素的数量；  
  nbad：整数标量或可广播数组，总体中失败元素的数量；  
  nsample：整数标量或可广播数组，每次不放回抽样的样本量；  
  size：整数或整型元组，指定输出形状，支持维度 1–3；  
  dtype：整型类型，暂定为 `int32` / `int64`。
- **计划返回类型：**  
  NPUArray（整型）。
- **计划功能描述：**  
  实现超几何分布采样，模拟在有限总体中不放回抽样时，样本中成功元素的个数，常用于抽签、抽检等场景。

---

<span id="negative_binomial">6. <mark> **negative_binomial** </mark></span>

- **计划参数：**  
  r：浮点或整数标量/数组，表示目标成功次数（r > 0）；  
  p：浮点标量或可广播数组，表示单次试验成功概率（0 < p ≤ 1）；  
  size：整数或整型元组，指定输出形状，支持维度 1–3；  
  dtype：整型类型，暂定为 `int32` / `int64`。
- **计划返回类型：**  
  NPUArray（整型）。
- **计划功能描述：**  
  生成服从负二项分布的随机样本，通常表示在达到 r 次成功之前经历的失败次数或试验总次数，用于过度离散计数数据建模。

---

<span id="randint">7. <mark> **randint** </mark></span>

- **计划参数：**  
  low：整数标量或可广播数组，包含在内的下界；  
  high：整数标量或可广播数组，不包含在内的上界（必须大于 low）；  
  size：整数或整型元组，指定输出形状，支持维度 1–4；  
  dtype：整型类型，暂定为 `int32` / `int64`。
- **计划返回类型：**  
  NPUArray（整型）。
- **计划功能描述：**  
  在区间 [low, high) 上生成均匀分布的整数随机数，是最基础的整型随机采样接口，可用于索引、标签或离散参数的快速构造。


## Continuous Distributions （连续分布采样，TODO）

**计划支持的 API 数量：9**  
**未来计划支持的 API：** [normal](#normal), [lognormal](#lognormal), [exponential](#exponential), [gamma](#gamma), [beta](#beta), [laplace](#laplace), [logistic](#logistic), [gumbel](#gumbel), [weibull](#weibull)

> 说明：本节覆盖常见连续概率分布的随机采样接口，统一采用浮点 dtype，支持 1–4 维张量输出，用于统计建模、蒙特卡洛模拟与测试数据构造等场景。

---

<span id="normal">1. <mark> **normal** </mark></span>

- **计划参数：**  
  loc：浮点标量或可广播数组，表示正态分布均值；  
  scale：浮点标量或可广播数组，表示标准差 (> 0)；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从正态分布 𝒩(loc, scale²) 的随机样本，用作最基础的高斯噪声接口。

---

<span id="lognormal">2. <mark> **lognormal** </mark></span>

- **计划参数：**  
  mean：浮点标量或可广播数组，对应底层正态分布的均值；  
  sigma：浮点标量或可广播数组，对应底层正态分布的标准差 (> 0)；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从对数正态分布的随机样本，其中 `log(X)` 服从正态分布，常用于建模正值且右偏的数据（如收入、尺度变量等）。

---

<span id="exponential">3. <mark> **exponential** </mark></span>

- **计划参数：**  
  scale：浮点标量或可广播数组，表示尺度参数 (> 0)，其倒数为速率 λ；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从指数分布的随机样本，密度函数 f(x) = λ · exp(-λx)，常用于建模等待时间、寿命等非负随机变量。

---

<span id="gamma">4. <mark> **gamma** </mark></span>

- **计划参数：**  
  shape：浮点标量或可广播数组，表示形状参数 α (> 0)；  
  scale：浮点标量或可广播数组，表示尺度参数 θ (> 0)；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从伽马分布的随机样本，可用于建模等待时间、方差先验（如 Gamma–Poisson、Gamma–Exponential 共轭结构）等。

---

<span id="beta">5. <mark> **beta** </mark></span>

- **计划参数：**  
  a：浮点标量或可广播数组，表示 Beta 分布的第一个形状参数 α (> 0)；  
  b：浮点标量或可广播数组，表示 Beta 分布的第二个形状参数 β (> 0)；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从 Beta(α, β) 分布的随机样本，取值范围在 [0, 1]，常用于概率/比例型变量建模与贝叶斯先验。

---

<span id="laplace">6. <mark> **laplace** </mark></span>

- **计划参数：**  
  loc：浮点标量或可广播数组，表示拉普拉斯分布的中心位置；  
  scale：浮点标量或可广播数组，表示尺度参数 (> 0)；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从拉普拉斯（双指数）分布的随机样本，相比高斯分布尾部更重，可用于鲁棒建模或稀疏噪声场景。

---

<span id="logistic">7. <mark> **logistic** </mark></span>

- **计划参数：**  
  loc：浮点标量或可广播数组，表示分布位置参数；  
  scale：浮点标量或可广播数组，表示尺度参数 (> 0)；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从逻辑分布的随机样本，其 CDF 为 logistic 形式，常用于逻辑回归、分类阈值建模等场景。

---

<span id="gumbel">8. <mark> **gumbel** </mark></span>

- **计划参数：**  
  loc：浮点标量或可广播数组，表示位置参数；  
  scale：浮点标量或可广播数组，表示尺度参数 (> 0)；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从 Gumbel 极值分布的随机样本，常用于极端事件建模（如最大风速、峰值负载等）。

---

<span id="weibull">9. <mark> **weibull** </mark></span>

- **计划参数：**  
  a：浮点标量或可广播数组，表示形状参数 k (> 0)；  
  size：整数或整型元组，指定输出形状（维度：1–4）；  
  dtype：浮点类型，暂定为 `float32` / `float64`。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从 Weibull 分布的随机样本，常用于可靠性分析、寿命建模等场景；当形状参数取特定值时可退化为指数分布等特殊情形。

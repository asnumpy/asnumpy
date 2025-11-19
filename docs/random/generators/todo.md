## Random Generators （随机流与基础生成，TODO）

**计划支持的 API 数量：6**  
**未来计划支持的 API：** [seed](#seed), [uniform](#uniform), [rand](#rand), [randn](#randn), [random_sample](#random_sample), [standard_normal](#standard_normal)

> 说明：本节覆盖基础随机数生成接口，包括设置随机种子、均匀分布和正态分布采样等，用于构建上层随机算子与测试样本。

---

<span id="seed">1. <mark> **seed** </mark></span>

- **计划参数：**  
  seed：整数标量，用于设置全局随机数种子。
- **计划返回类型：**  
  None
- **计划功能描述：**  
  设置全局随机数生成器状态，保证后续随机结果可复现。

---

<span id="uniform">2. <mark> **uniform** </mark></span>

- **计划参数：**  
  low, high：浮点标量或可广播数组；  
  size：输出形状。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  在区间 [low, high) 上生成均匀分布随机样本。

---

<span id="rand">3. <mark> **rand** </mark></span>

- **计划参数：**  
  *size*：若干整数参数或整型元组，表示输出数组形状。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从 [0, 1) 区间均匀分布的随机数组，是最基础的连续均匀随机接口。

---

<span id="randn">4. <mark> **randn** </mark></span>

- **计划参数：**  
  *size*：若干整数参数或整型元组，表示输出数组形状。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  生成服从标准正态分布 N(0, 1) 的随机数组。

---

<span id="random_sample">5. <mark> **random_sample** </mark></span>

- **计划参数：**  
  *size*：若干整数参数或整型元组，表示输出数组形状。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  提供与 `rand` 等价的均匀分布采样接口，用作别名以兼容 NumPy 调用习惯。

---

<span id="standard_normal">6. <mark> **standard_normal** </mark></span>

- **计划参数：**  
  *size*：若干整数参数或整型元组，表示输出数组形状。
- **计划返回类型：**  
  NPUArray（浮点型）。
- **计划功能描述：**  
  提供与 `randn` 等价的标准正态分布采样接口，均值为 0、方差为 1。

<a id="distributions"></a>
## distributions
**目前已完成的 API：**  
[Generator_Pareto](#generator_pareto) ·  
[Generator_Rayleigh](#generator_rayleigh) ·  
[Generator_Normal](#generator_normal) ·  
[Generator_Uniform](#generator_uniform) ·  
[Generator_Standard_normal](#generator_standard_normal) ·  
[Generator_Standard_cauchy](#generator_standard_cauchy) ·  
[Generator_Weibull](#generator_weibull) ·  
[Binomial](#binomial) ·  
[Exponential](#exponential) ·  
[Geometric](#geometric) ·  
[Gumbel](#gumbel) ·  
[Laplace](#laplace) ·  
[Logistic](#logistic) ·  
[Lognormal](#lognormal) 

1. <mark> **<a id="generator_pareto"></a>Generator_Pareto** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  a：float，分布的形状参数(必须满足 a > 0)  
  size：std::vector<int64_t>，输出形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从具有指定形状的帕累托分布中抽取随机样本。

2. <mark> **<a id="generator_rayleigh"></a>Generator_Rayleigh** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  scale：float，尺度参数(必须满足 scale > 0)  
  size：std::vector<int64_t>，输出形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从瑞利分布抽取随机样本。

3. <mark> **<a id="generator_normal"></a>Generator_Normal** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  loc：float，分布的均值 μ  
  scale：float，分布的标准差 σ（必须满足 scale > 0）    
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从正态(高斯)分布 N(μ, σ²) 中抽取随机样本。

4. <mark> **<a id="generator_uniform"></a>Generator_Uniform** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  low：double，输出区间的下限。所有生成的值都将大于或等于 low。    
  high：double，输出区间的上限。所有生成的值都将小于 high。    
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从均匀分布 U(low, high) 中抽取随机样本。

5. <mark> **<a id="generator_standard_normal"></a>Generator_Standard_normal** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  size：std::vector<int64_t>，输出张量形状    
- **返回类型：**  
  NPUArray  
- **功能：**  
  从标准正态分布 N(0, 1) 中抽取随机样本。

6. <mark> **<a id="generator_standard_cauchy"></a>Generator_Standard_cauchy** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从均值 = 0 的标准柯西分布中抽取随机样本。

7. <mark> **<a id="generator_weibull"></a>Generator_Weibull** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  a：float，分布的形状参数(必须满足 a > 0)  
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从威布尔分布中抽取随机样本。

8. <mark> **<a id="binomial"></a>Binomial** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  n：int，试验次数（需 n ≥ 0）  
  p：float，成功概率（需 0 ≤ p ≤ 1）  
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从二项分布 Binomial(n, p) 中抽取随机样本。

9. <mark> **<a id="exponential"></a>Exponential** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  scale：float，尺度（均值 = scale，满足 scale > 0）  
  size：std::vector<int64_t>，输出张量形状    
- **返回类型：**  
  NPUArray  
- **功能：**  
  从指数分布 Exp(scale) 中抽取随机样本。

10. <mark> **<a id="geometric"></a>Geometric** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  p：float，单次试验成功概率（满足 0 < p ≤ 1，p = 1 时结果恒为 1）  
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从几何分布中抽取随机样本。

11. <mark> **<a id="gumbel"></a>Gumbel** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  loc：double，分布众数的位置  
  scale：double，分布的尺度参数（满足 scale > 0）  
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从 Gumbel 分布中抽取随机样本。

12. <mark> **<a id="laplace"></a>Laplace** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  loc：double，分布峰值的位置 μ  
  scale：double，λ，指数衰减（满足 scale > 0）    
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从具有指定位置（或均值）和尺度（衰减）参数的拉普拉斯分布或双指数分布中抽取随机样本。

13. <mark> **<a id="logistic"></a>Logistic** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  loc：double，位置或均值  
  scale：double，尺度（满足 scale > 0）  
  size：std::vector<int64_t>，输出张量形状  
- **返回类型：**  
  NPUArray  
- **功能：**  
  从逻辑分布中抽取样本。

14. <mark> **<a id="lognormal"></a>Lognormal** </mark>  
[返回顶部](#distributions)  
- **参数：**  
  mean：double，底层正态分布的均值    
  sigma：double，底层正态分布的标准差（满足 sigma > 0）  
  size：std::vector<int64_t>，输出张量形状    
- **返回类型：**  
  NPUArray  
- **功能：**  
  从具有指定均值、标准差和数组形状的对数正态分布中抽取随机样本。

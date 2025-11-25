
## Polynomial Fitting （多项式拟合与逼近，TODO）

**计划支持的 API 数量：3**  
**未来计划支持的 API：** [polyfit](#polyfit), [chebfit](#chebfit), [legfit](#legfit)

> 说明：本节提供基于最小二乘的多项式拟合与 Chebyshev / Legendre 基函数逼近接口，用于数据拟合与函数近似。

---

<span id="polyfit">1. <mark> **polyfit** </mark></span>

- **计划参数：**  
  x：自变量样本，一维浮点数组，shape (N,) 或可广播到该形状；  
  y：因变量样本，一维或二维数组，shape 与 x 兼容；  
  deg：整数或整数列表，指定拟合多项式次数；  
  w：可选，权重数组，shape (N,)，表示每个样本的相对重要性；  
  rcond / full / cov：可选，控制拟合残差与协方差输出。
- **计划返回类型：**  
  NPUArray 或列表，每个元素为拟合得到的系数数组。
- **计划功能描述：**  
  利用最小二乘法在给定数据 (x, y) 上拟合普通幂基多项式，支持加权拟合和误差信息返回。

---

<span id="chebfit">2. <mark> **chebfit** </mark></span>

- **计划参数：**  
  x：自变量样本，一般要求位于区间 [-1, 1] 或通过映射归一化到该区间；  
  y：因变量样本；  
  deg：Chebyshev 多项式的最高阶；  
  w：可选权重；  
  domain：可选，原始 x 的定义域，用于自动缩放到标准区间。
- **计划返回类型：**  
  NPUArray，Chebyshev 基下的系数数组。
- **计划功能描述：**  
  在 Chebyshev 多项式基底上进行最小二乘拟合，相比普通多项式具有更好的数值稳定性和逼近性质。

---

<span id="legfit">3. <mark> **legfit** </mark></span>

- **计划参数：**  
  x：自变量样本，通常位于 [-1, 1]；  
  y：因变量样本；  
  deg：Legendre 多项式最高阶；  
  w：可选权重；  
  domain：可选定义域信息。
- **计划返回类型：**  
  NPUArray，Legendre 基下的系数数组。
- **计划功能描述：**  
  基于 Legendre 多项式构建的最小二乘拟合，用于在特定权重下对函数进行正交展开和逼近。


## Basic Polynomial Operations （基础多项式运算，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [polyadd](#polyadd), [polysub](#polysub), [polymul](#polymul), [polydiv](#polydiv)

> 说明：本节针对以 1D 系数数组表示的多项式，提供加减乘除等基础代数运算，默认多项式按降幂排列系数。

---

<span id="polyadd">1. <mark> **polyadd** </mark></span>

- **计划参数：**  
  p：NPUArray 或序列，浮点或复数类型，shape (n,)；  
  q：NPUArray 或序列，浮点或复数类型，shape (m,)；  
  dtype：可选，输出系数类型，默认为广播后的公共 dtype。
- **计划返回类型：**  
  NPUArray，shape (k,)，k = max(n, m)。
- **计划功能描述：**  
  计算两个多项式 p(x) 与 q(x) 的和 p(x) + q(x)，系数按最大次数对齐后逐项相加。

---

<span id="polysub">2. <mark> **polysub** </mark></span>

- **计划参数：**  
  p：被减多项式系数数组；  
  q：减数多项式系数数组；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray，shape (k,)。
- **计划功能描述：**  
  计算多项式差值 p(x) - q(x)，与 `polyadd` 类似但对第二个多项式取相反数后相加。

---

<span id="polymul">3. <mark> **polymul** </mark></span>

- **计划参数：**  
  p：多项式系数数组，shape (n,)；  
  q：多项式系数数组，shape (m,)；  
  mode：可选，决定使用直接卷积还是 FFT 加速；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray，shape (n + m - 1,)。
- **计划功能描述：**  
  计算两个多项式的乘积 p(x) · q(x)，本质为一维离散卷积，可选择在高阶多项式时用 FFT 提升性能。

---

<span id="polydiv">4. <mark> **polydiv** </mark></span>

- **计划参数：**  
  p：被除多项式系数数组；  
  q：除数多项式系数数组（最高次项系数需非零）；  
  dtype：可选。
- **计划返回类型：**  
  (quotient, remainder) 二元组，均为 1D NPUArray。
- **计划功能描述：**  
  对多项式进行长除法，返回商多项式和余数多项式，使得 p(x) = q(x)·quotient(x) + remainder(x)。

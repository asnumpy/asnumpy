## Solving Equations and Inverting Matrices （矩阵求解与逆运算，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [inv](#inv), [pinv](#pinv), [solve](#solve), [lstsq](#lstsq)

> 说明：本节提供求逆矩阵、解线性方程组以及最小二乘解等功能，是线性系统求解与回归分析的基础。

---

<span id="inv">1. <mark> **inv** </mark></span>

- **计划参数：**  
  a：方阵。
- **计划返回类型：**  
  NPUArray，方阵。
- **计划功能描述：**  
  计算矩阵的普通逆，一般用于小规模问题或教学示例。

---

<span id="pinv">2. <mark> **pinv** </mark></span>

- **计划参数：**  
  a：矩阵。
- **计划返回类型：**  
  NPUArray，矩阵。
- **计划功能描述：**  
  计算矩阵的 Moore–Penrose 伪逆，适用于奇异或非方阵。

---

<span id="solve">3. <mark> **solve** </mark></span>

- **计划参数：**  
  A：系数矩阵；b：右端项向量或矩阵。
- **计划返回类型：**  
  x，使得 Ax ≈ b。
- **计划功能描述：**  
  求解线性方程组，支持批量右端项。

---

<span id="lstsq">4. <mark> **lstsq** </mark></span>

- **计划参数：**  
  A：设计矩阵；b：观测向量；rcond：截断参数。
- **计划返回类型：**  
  (x, residuals, rank, s)。
- **计划功能描述：**  
  求解线性最小二乘问题，返回最优解及残差等信息。

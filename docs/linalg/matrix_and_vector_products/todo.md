## Matrix and Vector Products （向量与矩阵乘积，TODO）

**计划支持的 API 数量：8**  
**未来计划支持的 API：** [dot](#dot), [inner](#inner), [outer](#outer), [matmul](#matmul), [vdot](#vdot), [tensordot](#tensordot), [cross](#cross), [multi_dot](#multi_dot)

> 说明：本节覆盖向量内积、矩阵乘法、多维张量乘积等基础线性代数运算，主要用于数值计算与深度学习中的线性层实现。

---

<span id="dot">1. <mark> **dot** </mark></span>

- **计划参数：**  
  a, b：一维或二维数组。
- **计划返回类型：**  
  NPUArray，形状为 (m, k) × (k, n) → (m, n) 或标量。
- **计划功能描述：**  
  计算向量点积或矩阵乘积，支持高阶的广播与批量乘法。

---

<span id="inner">2. <mark> **inner** </mark></span>

- **计划参数：**  
  a, b：一维或高维数组。
- **计划返回类型：**  
  NPUArray，一维结果。
- **计划功能描述：**  
  计算最后一维上的内积，相当于按行展开后的点积。

---

<span id="outer">3. <mark> **outer** </mark></span>

- **计划参数：**  
  a, b：一维向量。
- **计划返回类型：**  
  NPUArray，二维矩阵。
- **计划功能描述：**  
  计算两个向量的外积，生成矩阵。

---

<span id="matmul">4. <mark> **matmul** </mark></span>

- **计划参数：**  
  a, b：二维或更高维数组。
- **计划返回类型：**  
  NPUArray，矩阵或批量矩阵。
- **计划功能描述：**  
  实现矩阵乘法和批量矩阵乘法，是 `@` 运算符的核心实现。

---

<span id="vdot">5. <mark> **vdot** </mark></span>

- **计划参数：**  
  a, b：一维向量。
- **计划返回类型：**  
  标量。
- **计划功能描述：**  
  计算向量点积，并对复数输入自动取共轭。

---

<span id="tensordot">6. <mark> **tensordot** </mark></span>

- **计划参数：**  
  a, b：多维张量；axes：指定求和轴。
- **计划返回类型：**  
  NPUArray，多维张量。
- **计划功能描述：**  
  在给定轴上进行张量乘积与求和，统一表达多种复杂线性代数运算。

---

<span id="cross">7. <mark> **cross** </mark></span>

- **计划参数：**  
  a, b：三维向量或其批量。
- **计划返回类型：**  
  NPUArray，三维向量。
- **计划功能描述：**  
  计算三维向量叉积，用于几何和物理计算。

---

<span id="multi_dot">8. <mark> **multi_dot** </mark></span>

- **计划参数：**  
  *arrays：多个二维矩阵序列。
- **计划返回类型：**  
  NPUArray，矩阵。
- **计划功能描述：**  
  自动选择矩阵乘法顺序以减少乘法次数，优化一串矩阵连乘。

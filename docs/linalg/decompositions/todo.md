## Matrix Decompositions （矩阵分解，TODO）

**计划支持的 API 数量：7**  
**未来计划支持的 API：** [det](#det), [slogdet](#slogdet), [svd](#svd), [qr](#qr), [eig](#eig), [eigh](#eigh), [cholesky](#cholesky)

> 说明：本节包含常用矩阵分解，如行列式、奇异值分解、特征值分解等，是数值线性代数与优化算法的基础组件。

---

<span id="det">1. <mark> **det** </mark></span>

- **计划参数：**  
  a：方阵。
- **计划返回类型：**  
  标量或批量标量。
- **计划功能描述：**  
  计算矩阵的行列式，用于判断可逆性和体积缩放因子。

---

<span id="slogdet">2. <mark> **slogdet** </mark></span>

- **计划参数：**  
  a：方阵。
- **计划返回类型：**  
  (sign, logdet) 元组。
- **计划功能描述：**  
  返回行列式的符号和对数值，避免大规模矩阵下数值溢出。

---

<span id="svd">3. <mark> **svd** </mark></span>

- **计划参数：**  
  a：矩阵；full_matrices, compute_uv：控制输出形式。
- **计划返回类型：**  
  (U, S, Vh)。
- **计划功能描述：**  
  对矩阵做奇异值分解，用于降维、压缩和稳健求逆。

---

<span id="qr">4. <mark> **qr** </mark></span>

- **计划参数：**  
  a：矩阵；mode：分解模式。
- **计划返回类型：**  
  (Q, R)。
- **计划功能描述：**  
  进行 QR 分解，将矩阵拆为正交矩阵与上三角矩阵。

---

<span id="eig">5. <mark> **eig** </mark></span>

- **计划参数：**  
  a：方阵。
- **计划返回类型：**  
  (w, v)，其中 w 为特征值，v 为特征向量。
- **计划功能描述：**  
  对一般方阵进行特征值与特征向量分解。

---

<span id="eigh">6. <mark> **eigh** </mark></span>

- **计划参数：**  
  a：对称或厄米矩阵。
- **计划返回类型：**  
  (w, v)。
- **计划功能描述：**  
  专门针对对称/厄米矩阵的特征分解，精度更高、效率更好。

---

<span id="cholesky">7. <mark> **cholesky** </mark></span>

- **计划参数：**  
  a：对称正定矩阵。
- **计划返回类型：**  
  L，下三角矩阵。
- **计划功能描述：**  
  进行 Cholesky 分解，常用于解线性方程组与高斯过程等场景。

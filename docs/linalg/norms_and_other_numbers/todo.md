## Norms and Other Numbers （范数及其他数值，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [norm](#norm), [matrix_power](#matrix_power), [cond](#cond), [trace](#trace)

> 说明：本节提供向量/矩阵范数、矩阵幂、条件数和迹等常用数值量，有助于评估矩阵大小、稳定性和结构特征。

---

<span id="norm">1. <mark> **norm** </mark></span>

- **计划参数：**  
  x：向量或矩阵；ord：范数类型；axis：可选轴参数。
- **计划返回类型：**  
  标量或按轴归约后的数组。
- **计划功能描述：**  
  计算向量或矩阵的 p 范数，用于度量大小或误差。

---

<span id="matrix_power">2. <mark> **matrix_power** </mark></span>

- **计划参数：**  
  a：方阵；n：整数幂。
- **计划返回类型：**  
  NPUArray，方阵。
- **计划功能描述：**  
  计算矩阵的整数次幂，支持正幂、零幂和负幂。

---

<span id="cond">3. <mark> **cond** </mark></span>

- **计划参数：**  
  a：矩阵；p：范数类型。
- **计划返回类型：**  
  标量。
- **计划功能描述：**  
  计算矩阵的条件数，用于评估线性系统的数值稳定性。

---

<span id="trace">4. <mark> **trace** </mark></span>

- **计划参数：**  
  a：矩阵；offset：可选偏移。
- **计划返回类型：**  
  标量。
- **计划功能描述：**  
  计算矩阵对角线（或偏移对角线）元素之和。


## Polynomial Evaluation & Calculus （多项式求值与微积分，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [polyval](#polyval), [polyder](#polyder), [polyint](#polyint), [roots](#roots)

> 说明：本节在系数形式的基础上提供多项式求值、求导、积分及求根等操作，用于构建基础解析解算子。

---

<span id="polyval">1. <mark> **polyval** </mark></span>

- **计划参数：**  
  p：多项式系数数组，shape (n,)；  
  x：自变量取值，标量或 NPUArray，shape 支持 1–2 维，并与广播规则兼容；  
  dtype：可选，输出类型。
- **计划返回类型：**  
  NPUArray，与输入 x 广播后的形状一致。
- **计划功能描述：**  
  在给定点 x 上计算多项式 p(x) 的值，内部可采用 Horner 法提高数值稳定性与计算效率。

---

<span id="polyder">2. <mark> **polyder** </mark></span>

- **计划参数：**  
  p：多项式系数数组；  
  m：可选，非负整数，表示求导阶数，默认 1；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray，shape (k,)。
- **计划功能描述：**  
  计算多项式的一阶或高阶导数，对系数执行解析求导，输出导函数的系数表示。

---

<span id="polyint">3. <mark> **polyint** </mark></span>

- **计划参数：**  
  p：多项式系数数组；  
  m：可选，非负整数，表示积分次数，默认 1；  
  k：可选，积分常数数组，长度 m，用于设置多重积分中的常数项；  
  dtype：可选。
- **计划返回类型：**  
  NPUArray，shape (n + m,)。
- **计划功能描述：**  
  对多项式进行不定积分，多次积分时依次累积，并在每一步加入对应积分常数。

---

<span id="roots">4. <mark> **roots** </mark></span>

- **计划参数：**  
  p：多项式系数数组，dtype 通常为复数或浮点；  
  max_iter / tol：可选，数值求根的迭代与收敛控制参数。
- **计划返回类型：**  
  NPUArray（复数），shape (n-1,)，表示多项式的所有根。
- **计划功能描述：**  
  求解多项式的全部复根，可基于伴随矩阵特征分解或其他稳定的多项式求根算法实现。

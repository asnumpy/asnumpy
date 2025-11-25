## Array Type Testing （数组类型测试，TODO）

**计划支持的 API 数量：6**  
**未来计划支持的 API：** [iscomplex](#iscomplex), [iscomplexobj](#iscomplexobj), [isfortran](#isfortran), [isreal](#isreal), [isrealobj](#isrealobj), [isscalar](#isscalar)

> 说明：本节用于判断数组或对象的类型特征（是否复数、实数、Fortran 连续、标量等），当前仍处于设计阶段，具体实现可能在后续版本中调整。

---

<span id="iscomplex">1. <mark> **iscomplex** </mark></span>

- **计划参数：**  
  x：任意 dtype 的输入数组或标量。

- **计划返回类型：**  
  NPUArray（bool 型，形状与 x 相同）

- **计划功能描述：**  
  逐元素判断是否具有非零虚部（值级判断，不看 dtype）。

---

<span id="iscomplexobj">2. <mark> **iscomplexobj** </mark></span>

- **计划参数：**  
  x：任意对象或数组。

- **计划返回类型：**  
  bool 标量。

- **计划功能描述：**  
  判断输入是否为复数 dtype（类型级判断，仅看 dtype）。

---

<span id="isfortran">3. <mark> **isfortran** </mark></span>

- **计划参数：**  
  a：数组对象。

- **计划返回类型：**  
  bool 标量。

- **计划功能描述：**  
  判断数组是否为 Fortran 连续且非 C 连续（仅检查内存布局）。

---

<span id="isreal">4. <mark> **isreal** </mark></span>

- **计划参数：**  
  x：任意 dtype 的输入数组或标量。

- **计划返回类型：**  
  NPUArray（bool 型，形状与 x 相同）

- **计划功能描述：**  
  逐元素判断是否为实数（值级判断，复数虚部为 0 也视为 True）。

---

<span id="isrealobj">5. <mark> **isrealobj** </mark></span>

- **计划参数：**  
  x：任意对象或数组。

- **计划返回类型：**  
  bool 标量。

- **计划功能描述：**  
  判断输入是否为非复数 dtype（类型级判断，只看 dtype，不看数值）。

---

<span id="isscalar">6. <mark> **isscalar** </mark></span>

- **计划参数：**  
  x：任意 Python 对象。

- **计划返回类型：**  
  bool 标量。

- **计划功能描述：**  
  判断输入是否为标量类型（普通 Python 标量或 0 维数组视为 True）。

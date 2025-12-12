## Trigonometric Functions & Angle Conversion （三角函数与角度变换，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [degrees](#degrees), [deg2rad](#deg2rad), [unwrap](#unwrap), [rad2deg](#rad2deg)

> 说明：本节主要涵盖角度与弧度之间的转换，以及相位展开等操作，当前仍处于设计阶段，具体实现可能在后续版本中调整。

---

<span id="degrees">1. <mark> **degrees** </mark></span>

- **计划参数：**  
  x：输入数组或标量（弧度制）。

- **计划返回类型：**  
  NPUArray（与 x 形状一致）

- **计划功能描述：**  
  将弧度制角度转换为角度制，逐元素计算 `x * 180 / π`。

---

<span id="deg2rad">2. <mark> **deg2rad** </mark></span>

- **计划参数：**  
  x：输入数组或标量（角度制）。

- **计划返回类型：**  
  NPUArray（与 x 形状一致）

- **计划功能描述：**  
  将角度制角度转换为弧度制，逐元素计算 `x * π / 180`，等价于 `radians` 的别名。

---

<span id="unwrap">3. <mark> **unwrap** </mark></span>

- **计划参数：**  
  p：相位数组（通常为弧度）；  
  discont：可选，判定跳变的阈值；  
  axis：可选，展开的轴。

- **计划返回类型：**  
  NPUArray（与 p 形状一致）

- **计划功能描述：**  
  通过在适当位置加减 2π，消除相位在 ±π 处的跳变，使相位曲线更加平滑。

---

<span id="rad2deg">4. <mark> **rad2deg** </mark></span>

- **计划参数：**  
  x：输入数组或标量（弧度制）。

- **计划返回类型：**  
  NPUArray（与 x 形状一致）

- **计划功能描述：**  
  将弧度制角度转换为角度制，逐元素计算 `x * 180 / π`，与 `degrees` 行为一致。

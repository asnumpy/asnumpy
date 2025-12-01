## Floating Point Routines （浮点数例程，TODO）

**计划支持的 API 数量：5**  
**未来计划支持的 API：** [frexp](#frexp), [ldexp](#ldexp), [nextafter](#nextafter), [spacing](#spacing), [copysign](#copysign)

> 说明：本节为浮点数分解、还原及精度相关工具函数的规划，当前仍处于设计阶段，后续可能补充更多细节和异常约定。

---

<span id="frexp">1. <mark> **frexp** </mark></span>

- **计划参数：**  
  x：输入浮点数组或标量。

- **计划返回类型：**  
  (mantissa, exponent) 两个 NPUArray，mantissa 与 x 同形状、浮点类型；exponent 为整型。

- **计划功能描述：**  
  将每个元素分解为尾数和 2 的指数，使得 `x = mantissa * 2**exponent`。

---

<span id="ldexp">2. <mark> **ldexp** </mark></span>

- **计划参数：**  
  mantissa：浮点数组或标量；  
  exponent：整型数组或标量，可与 mantissa 广播。

- **计划返回类型：**  
  NPUArray（浮点）

- **计划功能描述：**  
  计算 `mantissa * 2**exponent`，与 `frexp` 互为逆操作。

---

<span id="nextafter">3. <mark> **nextafter** </mark></span>

- **计划参数：**  
  x1：起点浮点数组或标量；  
  x2：目标方向浮点数组或标量，可广播。

- **计划返回类型：**  
  NPUArray（与 x1 同形状、同类型）

- **计划功能描述：**  
  返回从 x1 沿着指向 x2 的方向前进的下一个可表示浮点数。

---

<span id="spacing">4. <mark> **spacing** </mark></span>

- **计划参数：**  
  x：输入浮点数组或标量。

- **计划返回类型：**  
  NPUArray（与 x 同形状）

- **计划功能描述：**  
  返回每个 x 对应的、间隔最近的两个可表示浮点数之间的距离。

---

<span id="copysign">5. <mark> **copysign** </mark></span>

- **计划参数：**  
  x：浮点数组或标量，提供数值；  
  y：浮点数组或标量，提供符号。

- **计划返回类型：**  
  NPUArray（与广播后形状一致）

- **计划功能描述：**  
  用 y 的符号替换 x 的符号，得到“值来自 x、符号来自 y”的结果。

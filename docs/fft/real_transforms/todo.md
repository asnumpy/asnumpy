
## Real FFT Transforms （实数傅里叶变换，TODO）

**计划支持的 API 数量：6**  
**未来计划支持的 API：** [rfft](#rfft), [irfft](#irfft), [rfft2](#rfft2), [irfft2](#irfft2), [rfftn](#rfftn), [irfftn](#irfftn)

> 说明：本节针对实数输入提供半谱输出的 FFT 变换族，相比复数 FFT 在频域上只保留非冗余部分，可节省一半左右的存储与计算。

---

<span id="rfft">1. <mark> **rfft** </mark></span>

- **计划参数：**  
  a：实数输入数组，dtype 为浮点，shape (..., N)，维度 1–3；  
  n：可选，变换长度；  
  axis：可选，默认最后一维；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（复数），shape 为 (..., N//2 + 1)。
- **计划功能描述：**  
  对实数输入执行一维 FFT，仅返回非冗余频率采样点。

---

<span id="irfft">2. <mark> **irfft** </mark></span>

- **计划参数：**  
  a：复数频谱输入，shape (..., N//2 + 1)；  
  n：可选，重建出的信号长度；  
  axis：可选；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（浮点），shape (..., N)。
- **计划功能描述：**  
  对 `rfft` 的输出执行逆变换，恢复实数时域信号。

---

<span id="rfft2">3. <mark> **rfft2** </mark></span>

- **计划参数：**  
  a：实数输入数组，shape (m, n) 或 (..., m, n)；  
  s：可选，(m_out, n_out)；  
  axes：可选；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（复数），shape 通常为 (..., m_out, n_out//2 + 1)。
- **计划功能描述：**  
  对二维实数输入执行 2D FFT，并返回非冗余半谱结果。

---

<span id="irfft2">4. <mark> **irfft2** </mark></span>

- **计划参数：**  
  a：复数频谱输入，shape (..., m, n//2 + 1)；  
  s：可选，输出空间域尺寸；  
  axes：可选；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（浮点）。
- **计划功能描述：**  
  计算二维实数逆 FFT，将半谱频域恢复到原始实数图像/场。

---

<span id="rfftn">5. <mark> **rfftn** </mark></span>

- **计划参数：**  
  a：实数输入数组，维度 1–4；  
  s：可选，各维输出长度；  
  axes：可选；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（复数），在最后一个被变换维度上长度约为 N//2 + 1。
- **计划功能描述：**  
  对多维实数输入执行 N 维 FFT，返回压缩后的半谱结果。

---

<span id="irfftn">6. <mark> **irfftn** </mark></span>

- **计划参数：**  
  a：复数频谱输入，来自 `rfftn`；  
  s：可选，输出尺寸；  
  axes：可选；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（浮点）。
- **计划功能描述：**  
  对多维实数半谱执行逆变换，重建原始的多维实数信号。

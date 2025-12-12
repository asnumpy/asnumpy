
## Complex FFT Transforms （复数傅里叶变换，TODO）

**计划支持的 API 数量：6**  
**未来计划支持的 API：** [fft](#fft), [ifft](#ifft), [fft2](#fft2), [ifft2](#ifft2), [fftn](#fftn), [ifftn](#ifftn)

> 说明：本节实现对复数（或可视为复数的浮点）输入进行一维、二维及多维快速傅里叶变换及其逆变换，主要用于频域分析、卷积加速与信号处理场景。

---

<span id="fft">1. <mark> **fft** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为复数或浮点，可视为 shape (..., N)，支持维度 1–3；  
  n：可选，输出长度，若大于输入长度则零填充，小于则截断；  
  axis：可选，指定变换所在轴，默认最后一维；  
  norm：可选，归一化方式，支持 `None` / `"ortho"`。
- **计划返回类型：**  
  NPUArray，dtype 为复数。
- **计划功能描述：**  
  对一维信号执行快速傅里叶变换 (FFT)，将时域数据映射到频域。

---

<span id="ifft">2. <mark> **ifft** </mark></span>

- **计划参数：**  
  a：复数输入数组，shape 与 `fft` 输出一致；  
  n：可选，逆变换长度；  
  axis：可选；  
  norm：可选。
- **计划返回类型：**  
  NPUArray，dtype 为复数。
- **计划功能描述：**  
  计算一维逆快速傅里叶变换，将频域信号还原到时域/空间域。

---

<span id="fft2">3. <mark> **fft2** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为复数或浮点，shape (m, n) 或带批次的 (..., m, n)；  
  s：可选，长度为 2 的元组 (m_out, n_out)，指定输出尺寸；  
  axes：可选，需要执行 2D FFT 的两个维度；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（复数），shape 为指定的二维频域尺寸。
- **计划功能描述：**  
  对二维输入执行 2D 快速傅里叶变换，典型应用为图像频域处理、2D 卷积加速等。

---

<span id="ifft2">4. <mark> **ifft2** </mark></span>

- **计划参数：**  
  a：复数频域数组，shape (m, n) 或 (..., m, n)；  
  s：可选，输出空间域尺寸；  
  axes：可选；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（复数）。
- **计划功能描述：**  
  计算二维逆 FFT，将二维频谱恢复为时空域信号。

---

<span id="fftn">5. <mark> **fftn** </mark></span>

- **计划参数：**  
  a：输入数组，dtype 为复数或浮点，shape 任意，维度 1–4；  
  s：可选，整型元组，指定各维的输出长度；  
  axes：可选，需要执行 FFT 的多个维度；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（复数）。
- **计划功能描述：**  
  对多维输入执行 N 维快速傅里叶变换，统一封装 1D/2D/3D 等多种情况。

---

<span id="ifftn">6. <mark> **ifftn** </mark></span>

- **计划参数：**  
  a：复数频域数组；  
  s：可选，输出尺寸；  
  axes：可选；  
  norm：可选。
- **计划返回类型：**  
  NPUArray（复数）。
- **计划功能描述：**  
  对多维频谱执行逆 FFT，将其还原为对应维度的时空域信号。


## Frequency Helpers （频率辅助函数，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [fftfreq](#fftfreq), [rfftfreq](#rfftfreq), [fftshift](#fftshift), [ifftshift](#ifftshift)

> 说明：本节提供频率坐标生成与频谱重排等辅助工具函数，通常配合 FFT/实数 FFT 使用以完成频域分析和可视化。

---

<span id="fftfreq">1. <mark> **fftfreq** </mark></span>

- **计划参数：**  
  n：整数，FFT 采样点数 N；  
  d：可选，采样间隔（时间步长）`d`，默认 1.0。
- **计划返回类型：**  
  一维 NPUArray（浮点），shape 为 (N,)。
- **计划功能描述：**  
  生成长度为 N 的频率采样坐标数组，对应常规复数 FFT 的频率轴。

---

<span id="rfftfreq">2. <mark> **rfftfreq** </mark></span>

- **计划参数：**  
  n：整数，实数 FFT 的原始采样点数 N；  
  d：可选，采样间隔。
- **计划返回类型：**  
  一维 NPUArray（浮点），shape 为 (N//2 + 1,)。
- **计划功能描述：**  
  为 `rfft`/`rfftn` 生成非冗余频谱对应的频率采样点。

---

<span id="fftshift">3. <mark> **fftshift** </mark></span>

- **计划参数：**  
  x：输入数组，dtype 为浮点或复数，维度 1–4；  
  axes：可选，需要做频谱平移的轴，默认全部轴。
- **计划返回类型：**  
  NPUArray，与输入形状一致。
- **计划功能描述：**  
  将零频率分量移动到频谱中心位置，便于频域图像的可视化和后处理。

---

<span id="ifftshift">4. <mark> **ifftshift** </mark></span>

- **计划参数：**  
  x：输入数组；  
  axes：可选。
- **计划返回类型：**  
  NPUArray。
- **计划功能描述：**  
  执行 `fftshift` 的逆操作，将中心频率重新移回零频位置，通常在进行逆 FFT 前使用以恢复原始频谱排列。

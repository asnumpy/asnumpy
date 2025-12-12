## Array Creation （数组创建相关，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [arange](#arange), [linspace](#linspace), [logspace](#logspace), [geomspace](#geomspace)

> 说明：本节接口均为未来计划支持的功能，当前仍处于设计阶段，具体实现与参数细节可能在后续版本中调整。

<span id="arange">1. <mark> **arange** </mark></span>

- **计划参数：**  
  start：标量，序列起始值（包含）；  
  stop：标量，可选，序列结束值（不包含）；当省略时，默认为 `start`，此时起始值默认为 0；  
  step：标量，可选，步长，默认 1；  
  dtype：可选，输出数组的数据类型；  
  device：可选，目标设备（如 "npu"），默认使用当前全局设备。

- **计划返回类型：**  
  NPUArray（一维数组）

- **计划支持的 dtype：**  
  所有整数类型、所有浮点类型以及 bool 类型。

- **计划 shape 特征：**  
  一维，长度约为 10² ~ 10⁶ 级别，视设备显存与实现策略而定。

- **计划功能描述：**  
  生成等差数列，用于索引、循环与简单网格。

---

<span id="linspace">2. <mark> **linspace** </mark></span>

- **计划参数：**  
  start：标量，区间起始值（包含）；  
  stop：标量，区间结束值（默认包含，后续可能增加 `endpoint` 控制）；  
  num：整数，可选，生成样本点个数，默认 50；  
  endpoint：bool，可选，是否包含终点 `stop`，默认 True；  
  dtype：可选，输出数组的数据类型；  
  device：可选，目标设备，默认使用当前全局设备。

- **计划返回类型：**  
  NPUArray（一维数组）

- **计划支持的 dtype：**  
  所有整数类型、所有浮点类型（内部会优先采用浮点计算）。

- **计划 shape 特征：**  
  一维向量，长度约为 10² ~ 10⁵ 级别。

- **计划功能描述：**  
  在线性区间上均匀采样，用于插值、绘图和参数扫描。

---

<span id="logspace">3. <mark> **logspace** </mark></span>

- **计划参数：**  
  start：标量，对数空间起点（通常为底数的指数）；  
  stop：标量，对数空间终点；  
  num：整数，可选，生成样本点个数，默认 50；  
  base：标量，可选，对数底数，默认 10；  
  endpoint：bool，可选，是否包含终点 `stop`，默认 True；  
  dtype：可选，输出数组的数据类型；  
  device：可选，目标设备。

- **计划返回类型：**  
  NPUArray（一维数组）

- **计划支持的 dtype：**  
  输出以 float32 或 float64 为主，后续优先考虑支持 complex128。

- **计划 shape 特征：**  
  一维向量，长度约为 10² ~ 10⁵ 级别。

- **计划功能描述：**  
  在对数空间均匀采样，生成等比数列，用于指数刻度与跨量级扫描。

---

<span id="geomspace">4. <mark> **geomspace** </mark></span>

- **计划参数：**  
  start：标量或复数，序列起始值（包含）；  
  stop：标量或复数，序列结束值（通常包含，且计划要求与起始值符号一致）；  
  num：整数，可选，生成样本点个数，默认 50；  
  endpoint：bool，可选，是否包含终点 `stop`，默认 True；  
  dtype：可选，输出数组的数据类型；  
  device：可选，目标设备。

- **计划返回类型：**  
  NPUArray（一维数组）

- **计划支持的 dtype：**  
  输出以 float32 或 float64 为主，计划扩展支持 complex128，以适配复数等比序列场景。

- **计划 shape 特征：**  
  一维向量，长度约为 10² ~ 10⁵ 级别。

- **计划功能描述：**  
  生成首尾符号一致的等比序列，用于比例型参数与仿真。

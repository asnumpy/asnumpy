
## Correlations & Covariance （协方差与相关性函数，TODO）

**计划支持的 API 数量：4**  
**未来计划支持的 API：** [cov](#cov), [corrcoef](#corrcoef), [corr](#corr), [zscore](#zscore)

> 说明：本节提供计算变量间协方差、相关系数以及标准化分数的接口，主要服务于统计分析与特征工程。

---

<span id="cov">1. <mark> **cov** </mark></span>

- **计划参数：**  
  m：形状为 (n, n) 或 (n, k) 的二维浮点数组，其中行/列代表变量或观测；  
  rowvar：bool，若为 True，每一行视为一个变量，否则每一列视为变量；  
  bias：bool，是否采用 N 而非 N-1 归一化；  
  ddof：可选，自由度修正；  
  fweights/aweights：可选，频数权重或观测权重。
- **计划返回类型：**  
  NPUArray，shape 为 (n, n) 的协方差矩阵。
- **计划功能描述：**  
  计算变量之间的协方差矩阵，用于刻画线性依赖程度和联合波动性。

---

<span id="corrcoef">2. <mark> **corrcoef** </mark></span>

- **计划参数：**  
  m：形状为 (n, n) 或 (n, k) 的二维浮点数组；  
  rowvar：bool，含义同 `cov`；  
  bias / ddof：可选，用于控制底层方差估计。
- **计划返回类型：**  
  NPUArray，shape 为 (n, n) 的相关系数矩阵。
- **计划功能描述：**  
  基于协方差矩阵归一化得到皮尔逊相关系数矩阵，取值范围 [-1, 1]，衡量线性相关程度。

---

<span id="corr">3. <mark> **corr** </mark></span>

- **计划参数：**  
  x：输入数组，shape 与输入一致；  
  y：可选，第二组数据，与 x 在样本维度上对齐；  
  axis：可选，指定样本维度；  
  method：可选，相关性类型（例如 "pearson"、"spearman" 等）。
- **计划返回类型：**  
  NPUArray，标量或矩阵，视输入而定。
- **计划功能描述：**  
  计算两组数据的一维或二维相关性指标，提供比 corrcoef 更灵活的接口封装。

---

<span id="zscore">4. <mark> **zscore** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选，指定标准化的轴；  
  ddof：可选，用于标准差计算；  
  keepdims：bool；  
  nan_policy：可选，控制 NaN 处理策略（例如 "propagate"、"omit"）。
- **计划返回类型：**  
  NPUArray（浮点型），与输入形状一致或按 keepdims 调整。
- **计划功能描述：**  
  对输入进行标准化变换 `z = (x - mean) / std`，将不同尺度的特征转换为零均值、单位方差，便于后续建模。

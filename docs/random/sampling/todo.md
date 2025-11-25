
## Sampling & Permutations （随机采样与排列，TODO）

**计划支持的 API 数量：5**  
**未来计划支持的 API：** [choice](#choice), [permutation](#permutation), [shuffle](#shuffle), [random_integers](#random_integers), [sample](#sample)

> 说明：本节提供从数组或整数区间中进行随机采样、随机打乱与排列的接口，统一与 NumPy 语义对齐，用于构造训练/验证集拆分、随机打乱批次、子集抽样等通用场景。

---

<span id="choice">1. <mark> **choice** </mark></span>

- **计划参数：**  
  a：一维或多维 NPUArray / Python 序列，表示候选元素或上界（当为整数时表示从 `arange(a)` 中采样）；  
  size：可选，整数或整型元组，指定输出形状，支持维度 1–4；  
  replace：bool，是否允许重复采样，`True` 表示有放回，`False` 表示无放回；  
  p：可选，一维概率向量，与候选元素长度一致，用于指定带权重的采样分布，要求元素和为 1；  
  dtype：与输入 `a` 一致或通过推导得到。
- **计划返回类型：**  
  NPUArray，元素类型与 `a` 一致。
- **计划功能描述：**  
  从给定数组或整数区间中进行随机采样，可选择是否带权重、是否允许重复，用于子集抽样、类别重采样等场景。

---

<span id="permutation">2. <mark> **permutation** </mark></span>

- **计划参数：**  
  x：整数 n 或 NPUArray；  
  当 `x` 为整数 n 时：生成 `[0, 1, ..., n-1]` 的一个随机排列；  
  当 `x` 为数组时：返回其第一维元素的随机排列副本；  
  dtype：与输入数组保持一致。
- **计划返回类型：**  
  NPUArray，形状与输入一致，维度支持 1–4。
- **计划功能描述：**  
  返回输入的随机排列版本，不修改原数组。常用于打乱索引、顺序敏感任务的随机重排等。

---

<span id="shuffle">3. <mark> **shuffle** </mark></span>

- **计划参数：**  
  x：NPUArray，可为 1–4 维；  
  axis：可选，整数，指定需要打乱的维度，默认打乱第一维；  
  in_place：是否原地打乱，默认 `True`；若为 `False`，返回打乱后的副本。  
- **计划返回类型：**  
  当 `in_place=True` 时返回 None（与 NumPy 行为一致）；  
  当 `in_place=False` 时返回打乱后的 NPUArray。
- **计划功能描述：**  
  对数组沿指定维度进行随机打乱，常用于 DataLoader 中随机化样本顺序或构造洗牌训练序列。

---

<span id="random_integers">4. <mark> **random_integers** </mark></span>

- **计划参数：**  
  low：整数标量或可广播数组，闭区间下界；  
  high：可选，整数标量或可广播数组，闭区间上界，若省略则区间为 `[1, low]`；  
  size：整数或整型元组，指定输出形状，支持维度 1–3；  
  dtype：整型类型，建议默认 `int32`。
- **计划返回类型：**  
  NPUArray（整型）。
- **计划功能描述：**  
  在闭区间 `[low, high]` 上均匀地生成整数随机数，是 `randint` 的旧版别名接口，主要为了兼容部分遗留代码。

---

<span id="sample">5. <mark> **sample** </mark></span>

- **计划参数：**  
  x：一维或多维 NPUArray / 序列，作为总体；  
  size：可选，整数或整型元组，指定抽样数量或抽样形状，支持维度 1–3；  
  axis：可选，抽样所在的维度，默认第一维；  
  replace：bool，是否允许重复采样，默认 `False`（无放回）；  
  dtype：与输入一致。
- **计划返回类型：**  
  NPUArray，形状与 `size` 和 `axis` 设置有关。
- **计划功能描述：**  
  从输入数组中执行无放回（或可选有放回）随机抽样，与 `choice` 功能类似但更偏向“从已有数组中选取子集”，用于构建 mini‑batch、负采样集合等。

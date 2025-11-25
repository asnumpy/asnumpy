## Shape & Manipulation （数组形状与变换，TODO）

**计划支持的 API 数量：14**  
**未来计划支持的 API：** [reshape](#reshape), [ravel](#ravel), [flatten](#flatten), [resize](#resize), [squeeze](#squeeze), [expand_dims](#expand_dims), [transpose](#transpose), [swapaxes](#swapaxes), [moveaxis](#moveaxis), [rollaxis](#rollaxis), [T](#T), [broadcast_to](#broadcast_to), [repeat](#repeat), [tile](#tile)

> 说明：本节接口均为未来计划支持的功能，当前仍处于设计阶段，具体实现与参数细节可能在后续版本中调整。

<span id="reshape">1. <mark> **reshape** </mark></span>

- **计划参数：**  
  a：输入数组；  
  newshape：目标形状，可以是整数或整型元组；  
  order：可选，数据在内存中的读取顺序，默认 "C"。

- **计划返回类型：**  
  NPUArray（与输入元素个数相同）

- **计划支持的 dtype：**  
  保持原类型，支持所有整数、浮点和 bool。

- **计划 shape 特征：**  
  元素总数不变，维度数 1–4。

- **计划功能描述：**  
  仅改变视图形状，不改变底层数据布局。

---

<span id="ravel">2. <mark> **ravel** </mark></span>

- **计划参数：**  
  a：输入数组；  
  order：可选，展平顺序，默认 "C"。

- **计划返回类型：**  
  NPUArray（一维视图或副本）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  输出为一维，长度为原数组元素总数。

- **计划功能描述：**  
  快速将任意形状展平为一维视图。

---

<span id="flatten">3. <mark> **flatten** </mark></span>

- **计划参数：**  
  a：输入数组；  
  order：可选，展平顺序，默认 "C"。

- **计划返回类型：**  
  NPUArray（一维数组的副本）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  输出为一维，长度为原数组元素总数。

- **计划功能描述：**  
  展平数组并总是返回新数组。

---

<span id="resize">4. <mark> **resize** </mark></span>

- **计划参数：**  
  a：输入数组；  
  newshape：目标形状；  
  refcheck：可选，是否检查引用。

- **计划返回类型：**  
  NPUArray

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  维度 1–4，目标形状任意，超出部分填 0 或截断。

- **计划功能描述：**  
  调整数组大小并在必要时重新分配内存。

---

<span id="squeeze">5. <mark> **squeeze** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：可选，要压缩的轴或轴元组。

- **计划返回类型：**  
  NPUArray（视图）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  移除长度为 1 的维度，输出维度数 0–3。

- **计划功能描述：**  
  去掉多余的长度为 1 的轴。

---

<span id="expand_dims">6. <mark> **expand_dims** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：整数或整型元组，表示插入新轴的位置。

- **计划返回类型：**  
  NPUArray（视图）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  维度增加 1，最终维度数 1–4。

- **计划功能描述：**  
  在指定位置插入长度为 1 的新轴。

---

<span id="transpose">7. <mark> **transpose** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axes：可选，新的轴顺序。

- **计划返回类型：**  
  NPUArray（视图）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  维度顺序重排，维度数 1–4。

- **计划功能描述：**  
  对数组各轴进行一般性的维度置换。

---

<span id="swapaxes">8. <mark> **swapaxes** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis1：第一个轴；  
  axis2：第二个轴。

- **计划返回类型：**  
  NPUArray（视图）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  维度数 1–4，仅交换两维位置。

- **计划功能描述：**  
  交换两个给定轴的位置。

---

<span id="moveaxis">9. <mark> **moveaxis** </mark></span>

- **计划参数：**  
  a：输入数组；  
  source：要移动的轴或轴序列；  
  destination：目标位置或位置序列。

- **计划返回类型：**  
  NPUArray（视图）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  维度数 1–4，根据映射重新排列各轴。

- **计划功能描述：**  
  将指定轴移动到目标位置以便对齐广播维度。

---

<span id="rollaxis">10. <mark> **rollaxis** </mark></span>

- **计划参数：**  
  a：输入数组；  
  axis：要移动的轴；  
  start：插入的位置，默认 0。

- **计划返回类型：**  
  NPUArray（视图）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  维度数 1–4，轴沿序列前移。

- **计划功能描述：**  
  将指定轴滚动到给定位置，用于兼容旧版语义。

---

<span id="T">11. <mark> **T** </mark></span>

- **计划属性来源：**  
  NPUArray.T（只读属性）

- **计划返回类型：**  
  NPUArray（视图）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  对 2D 数组为转置，对 3D 为最后两个轴互换。

- **计划功能描述：**  
  提供快捷的转置视图，等价于 `transpose()`。

---

<span id="broadcast_to">12. <mark> **broadcast_to** </mark></span>

- **计划参数：**  
  a：输入数组；  
  shape：目标形状。

- **计划返回类型：**  
  NPUArray（只读视图）

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  将较小数组广播到目标形状，维度数 1–4。

- **计划功能描述：**  
  将数组按广播规则扩展为给定形状。

---

<span id="repeat">13. <mark> **repeat** </mark></span>

- **计划参数：**  
  a：输入数组；  
  repeats：每个元素重复次数；  
  axis：可选，沿指定轴重复。

- **计划返回类型：**  
  NPUArray

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  维度 1–3，指定轴长度按倍数放大。

- **计划功能描述：**  
  沿给定轴重复元素以实现数据扩增或复制。

---

<span id="tile">14. <mark> **tile** </mark></span>

- **计划参数：**  
  a：输入数组；  
  reps：各轴的重复次数（标量或元组）。

- **计划返回类型：**  
  NPUArray

- **计划支持的 dtype：**  
  保持原类型。

- **计划 shape 特征：**  
  维度 1–3，沿各维度按 reps 重复拼接。

- **计划功能描述：**  
  沿各维复制并拼接数组构造更大网格。

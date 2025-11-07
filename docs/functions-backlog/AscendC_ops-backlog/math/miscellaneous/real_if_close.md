# real_if_close

- **算子名称**：`real_if_close`
- **对应函数**：[`numpy.real_if_close`](https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html)
- **开发人员**：刘骏@kbryantttt，2305258805@qq.com
- **难易度（预估）**：中
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/16

## 相关内容
- **dtype**：aclTensor，数据类型为所有 `int`、浮点数和复数类型（随输入 dtype 而定）。  
- **shape**：各维长度常为 16 ~ 2048，极限通常不超过 4096；总元素常为 10^2 ~ 10^6，极限通常不超过 10^7。  
- **维度**：常为 1 ~ 3，极限通常不超过 5。  
- **功能**：如果输入为复数且其**虚部的绝对值接近 0**，则将该元素转换为**实数**返回；否则保持复数不变。判定条件近似为：`abs(imag(a)) <= tol * eps(a)`（`eps(a)` 为该 dtype 的机器精度，`tol` 默认为 100），更精确定义参考 NumPy 文档。

## np 示例
```python
import numpy as np

# 典型：虚部极小，转换为实数
np.real_if_close([1 + 1e-14j, 2 + 0j])
# array([1., 2.])

# 调整 tol：虚部不够小，保持复数
np.real_if_close([1 + 1e-3j], tol=1e-6)
# array([1.+0.001j])
```

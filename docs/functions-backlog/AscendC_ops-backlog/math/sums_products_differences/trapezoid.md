# trapezoid

- **算子名称**：`trapezoid`
- **对应函数**：[`numpy.trapezoid`](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html)
- **开发人员**：李雯@m0_62621784，muyu1522021@163.com
- **难易度（预估）**：易
- **开发状态**：开发中
- **更新时间**：2025.11.4

## 相关内容
- **dtype**：aclTensor 或 aclTensor 元组，数据类型为所有浮点数类型，最好可以支持复数类型。  
- **shape**：各维长度常为 10^1 ~ 10^3，极限通常不超过 10^6；总元素常为 10^2 ~ 10^6，极限通常不超过 10^8。  
- **维度**：常为 1 ~ 3，极限通常不超过 5。  
- **功能**：使用**复合梯形法则**沿给定轴进行积分；支持非均匀采样点 `x` 或固定间距 `dx`，结果标量或按 `axis` 保持维度。

## np 示例
```python
import numpy as np

np.trapezoid([1, 2, 3])
# 4.0

np.trapezoid([1, 2, 3], x=[4, 6, 8])
# 8.0

np.trapezoid([1, 2, 3], dx=2)
# 8.0
```

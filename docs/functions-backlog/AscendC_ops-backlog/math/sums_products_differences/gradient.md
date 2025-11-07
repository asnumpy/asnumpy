# gradient

- **算子名称**：`gradient`
- **对应函数**：[`numpy.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html#numpy.gradient)
- **开发人员**：梁杨琳@liang-yanglin，3431470978@qq.com
- **难易度（预估）**：难
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/10

## 相关内容
- **dtype**：aclTensor 或 aclTensor 元组，数据类型为所有浮点数类型；建议兼容复数类型（complex64/complex128）。  
- **shape**：各维长度常为 10^2 ~ 10^4，极限通常不超过 10^6；总元素常为 10^4 ~ 10^6，极限通常不超过 10^7。  
- **维度**：常为 1 ~ 4，极限通常不超过 5。  
- **功能**：返回 N 维数组的梯度。在内部点使用三阶精确的中心差分；在边界处使用一阶或二阶精确的单侧（前向或后向）差分。因此，返回的梯度与输入数组具有相同的形状。  

## np 示例
```python
import numpy as np

f = np.array([1, 2, 4, 7, 11, 16])
np.gradient(f)
# array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])

np.gradient(f, 2)
# array([0.5 , 0.75, 1.25, 1.75, 2.25, 2.5 ])
```

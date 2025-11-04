# frexp

- **算子名称**：`frexp`
- **对应函数**：[`numpy.frexp`](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html)
- **开发人员**：石向阳@shi-xiangyang225，2678490361@qq.com
- **难易度（预估）**：中
- **开发状态**：开发中
- **更新时间**：2025.11.4

## 相关内容
- **dtype**：两个返回值，均为 aclTensor；第一个返回值（尾数）为所有浮点数类型，第二个返回值（指数）为所有整型类型。  
- **shape**：总元素常为 10^2 ~ 10^7，极限通常不超过 10^8。  
- **维度**：常为 1 ~ 4。  
- **功能**：将输入 `x` 的每个元素分解为**尾数**与**以 2 为底的指数**，满足 `x = mantissa * 2**exponent`；对 0 元素返回 `(0.0, 0)`。

## np 示例
```python
import numpy as np

x = np.arange(9)
y1, y2 = np.frexp(x)

y1
# array([0.   , 0.5  , 0.5  , 0.75 , 0.5  , 0.625, 0.75 , 0.875, 0.5  ])

y2
# array([0, 1, 2, 2, 3, 3, 3, 3, 4], dtype=int32)
```

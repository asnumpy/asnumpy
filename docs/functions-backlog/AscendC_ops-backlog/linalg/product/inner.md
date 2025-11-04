# inner

- **算子名称**：`inner`
- **对应函数**：[`numpy.inner`](https://numpy.org/doc/stable/reference/generated/numpy.inner.html)
- **开发人员**：梁杨琳@liang-yanglin，3431470978@qq.com
- **难易度（预估）**：中
- **开发状态**：开发中
- **更新时间**：2025.11.4

## 相关内容
- **dtype**：aclTensor，**输出数据类型**为 `int32`、`int64` 和浮点数类型；建议兼容复数类型。  
- **shape**：各维长度常为 1 ~ 4096，极限通常不超过 65536；总元素常为 10^2 ~ 10^6，极限通常不超过 10^8。  
- **维度**：常为 1 ~ 3，极限通常不超过 4。  
- **功能**：计算两个数组的**内积**。对一维数组相当于向量内积；对更高维，返回 `sum(a[..., i] * b[..., i])`，在 `a` 的最后一维与 `b` 的最后一维上求和，输出形状为 `a.shape[:-1] + b.shape[:-1]`。

## np 示例
```python
import numpy as np

a = np.arange(24).reshape((2, 3, 4))
b = np.arange(4)
c = np.inner(a, b)
c.shape
# (2, 3)

c
# array([[ 14,  38,  62],
#        [ 86, 110, 134]])
```

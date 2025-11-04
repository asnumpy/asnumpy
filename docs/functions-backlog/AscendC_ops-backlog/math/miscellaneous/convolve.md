# convolve

- **算子名称**：`convolve`
- **对应函数**：[`numpy.convolve`](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
- **开发人员**：刘骏@kbryantttt，2305258805@qq.com
- **难易度（预估）**：中
- **开发状态**：开发中
- **更新时间**：2025.11.4

## 相关内容
- **dtype**：aclTensor，数据类型为所有 `int`、浮点型，最好可以支持复数类型。  
- **shape**：序列长度常不超过 10^5；卷积核长度通常不超过 128；总元素通常不超过 10^7。  
- **维度**：1 维。  
- **功能**：返回两个一维序列的**离散线性卷积**（更多细节参见 `numpy.convolve`）。

## np 示例
```python
import numpy as np

np.convolve([1, 2, 3], [0, 1, 0.5])
# array([0. , 1. , 2.5, 4. , 1.5])
```

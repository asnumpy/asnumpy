# tensordot

- **算子名称**：`tensordot`
- **对应函数**：[`numpy.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html)
- **开发人员**：刘骏@kbryantttt，2305258805@qq.com
- **难易度（预估）**：难
- **开发状态**：开发中
- **更新时间**：2025.11.4

## 相关内容
- **dtype**：aclTensor，输出数据类型为 `int32`、`int64` 和浮点数类型；建议兼容复数类型。  
- **shape**：各维长度常为 1 ~ 4096，极限通常不超过 65536；总元素常为 10^2 ~ 10^6，极限通常不超过 10^8。  
- **维度**：常为 1 ~ 6。  
- **功能**：沿**指定轴**计算张量点积。`axes=0` 等价于外积；`axes=1` 等价于在 `a` 的最后一维与 `b` 的倒数第二维做内积；`axes=([a_axes], [b_axes])` 支持多轴配对求和，输出形状为未参与求和的剩余轴拼接。

## np 示例
```python
import numpy as np

a_0 = np.array([[1, 2], [3, 4]])
b_0 = np.array([[5, 6], [7, 8]])
c_0 = np.tensordot(a_0, b_0, axes=0)
c_0.shape
# (2, 2, 2, 2)
c_0
# array([[[[ 5,  6],
#          [ 7,  8]],
#         [[10, 12],
#          [14, 16]]],
#        [[[15, 18],
#          [21, 24]],
#         [[20, 24],
#          [28, 32]]]])

a = np.arange(60.).reshape(3, 4, 5)
b = np.arange(24.).reshape(4, 3, 2)
c = np.tensordot(a, b, axes=([1, 0], [0, 1]))
c.shape
# (5, 2)
c
# array([[4400., 4730.],
#        [4532., 4874.],
#        [4664., 5018.],
#        [4796., 5162.],
#        [4928., 5306.]])
```

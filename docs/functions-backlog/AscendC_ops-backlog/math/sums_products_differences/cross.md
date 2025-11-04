# cross

- **算子名称**：`cross`
- **对应函数**：[`numpy.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)
- **开发人员**：梁杨琳@liang-yanglin，3431470978@qq.com
- **难易度（预估）**：中
- **开发状态**：开发中
- **更新时间**：2025.11.4

## 相关内容
- **dtype**：aclTensor，数据类型为所有 `int` 和所有浮点数类型，最好支持复数类型。  
- **shape**：各维长度常为 10^2 ~ 10^4，极限通常不超过 10^6；**运算轴大小恒为 3**。  
- **维度**：常为 2 ~ 3。  
- **功能**：返回两个（数组的）向量的**叉积**。需要能够分别选择两个数组向量所处的维度（如 `axisa`、`axisb`），以及结果向量所在维度（`axis`），细节参考 `numpy.cross`。

## np 示例
```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])

np.cross(x, y)
# array([[-6, 12, -6],
#        [ 0,  0,  0],
#        [ 6, -12,  6]])

np.cross(x, y, axisa=0, axisb=0)
# array([[-24,  48, -24],
#        [-30,  60, -30],
#        [-36,  72, -36]])
```

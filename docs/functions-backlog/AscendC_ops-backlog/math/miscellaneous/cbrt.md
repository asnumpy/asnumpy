# cbrt

- **算子名称**：`cbrt`
- **对应函数**：[`numpy.cbrt`](https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html)
- **开发人员**：曹晓娟@c15503545287，1793805374@qq.com
- **难易度（预估）**：易
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **贡献地址**：https://gitee.com/sutonghua/ascendbasicops/tree/main/AscendVectorMath

## 相关内容
- **dtype**：aclTensor，数据类型为所有浮点数类型，最好可以支持复数类型。  
- **shape**：各维长度常为 16 ~ 512，极限通常不超过 4096；总元素常为 10^2 ~ 10^6，极限通常不超过 10^7。  
- **维度**：常为 1 ~ 4，极限通常不超过 5。  
- **功能**：逐元素返回数组的**立方根**。

## np 示例
```python
import numpy as np

np.cbrt([1, 8, 27])
# array([1., 2., 3.])
```

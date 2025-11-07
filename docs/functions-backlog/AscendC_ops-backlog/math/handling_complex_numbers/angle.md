# angle

- **算子名称**：`angle`
- **对应函数**：[`numpy.angle`](https://numpy.org/doc/stable/reference/generated/numpy.angle.html)
- **开发人员**：曹晓娟@c15503545287，1793805374@qq.com
- **难易度（预估）**：易
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/1

## 相关内容
- **dtype**：aclTensor，**输出数据类型为 float64**。  
- **shape**：各维长度常为 16 ~ 2048，极限通常不超过 8192；总元素常为 10^2 ~ 10^6，极限通常不超过 10^8。  
- **维度**：常为 1 ~ 3，极限通常不超过 4。  
- **功能**：逐元素返回复数参数的**相位角**（弧度）。对实数输入返回其极角（正数为 0，负数为 π）。

## np 示例
```python
import numpy as np

np.angle([1.0, 1.0j, 1+1j])  # in radians
# array([0.        , 1.57079633, 0.78539816])
```

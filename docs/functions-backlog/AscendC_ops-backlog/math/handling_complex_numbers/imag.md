# imag

- **算子名称**：`imag`
- **对应函数**：[`numpy.imag`](https://numpy.org/doc/stable/reference/generated/numpy.imag.html)
- **开发人员**：涂远航@TuYHAAAAAA，2476630290@qq.com
- **难易度（预估）**：易
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/12

## 相关内容
- **dtype**：aclTensor，数据类型为所有 `int` 和所有浮点数类型（随输入 dtype 而定；对复数输入返回其对应的实数浮点类型）。  
- **shape**：各维长度常为 16 ~ 2048，极限通常不超过 8192；总元素常为 10^2 ~ 10^6，极限通常不超过 10^8。  
- **维度**：常为 1 ~ 3，极限通常不超过 4。  
- **功能**：逐元素返回复数参数的**虚部**；对实数/整数输入返回全 0。

## np 示例
```python
import numpy as np

np.imag([1+2j, 3+4j, 5+6j])
# array([2., 4., 6.])
```

# absolute

- **算子名称**：`absolute`
- **对应函数**：[`numpy.absolute`](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html)
- **开发人员**：周建华@LePenseur，dawner000@163.com
- **难易度（预估）**：易
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/2

## 相关内容
- **dtype**：aclTensor，数据类型为 `bool`、所有 `int` 与浮点型（随输入而定）；**建议兼容复数类型**，对复数返回其模（输出对应的实数浮点类型）。  
- **shape**：各维长度常为 16 ~ 512，极限通常不超过 4096；总元素常为 10^2 ~ 10^6，极限通常不超过 10^7。  
- **维度**：常为 1 ~ 4，极限通常不超过 5。  
- **功能**：逐元素计算**绝对值**；若输入为复数 `a + bi`，返回 `sqrt(a^2 + b^2)`。

## np 示例
```python
import numpy as np

x = np.array([-1.2, 1.2])
np.absolute(x)
# array([1.2, 1.2])

np.absolute(1.2 + 1j)
# 1.5620499351813308
```

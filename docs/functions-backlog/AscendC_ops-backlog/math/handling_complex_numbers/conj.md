# conj

- **算子名称**：`conj`
- **对应函数**：[`numpy.conj`](https://numpy.org/doc/stable/reference/generated/numpy.conj.html)
- **开发人员**：裴浩博@xiaopei-1，1550747408@qq.com
- **难易度（预估）**：易
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/5

## 相关内容
- **dtype**：aclTensor，数据类型为所有 `int`、浮点型、复数类型（输出 dtype 与输入相同；对复数返回其共轭，对实数/整数保持不变）。  
- **shape**：各维长度常为 16 ~ 2048，极限通常不超过 8192；总元素常为 10^2 ~ 10^6，极限通常不超过 10^8。  
- **维度**：常为 1 ~ 3，极限通常不超过 4。  
- **功能**：逐元素返回**复共轭**。

## np 示例
```python
import numpy as np

x = np.eye(2) + 1j * np.eye(2)
np.conjugate(x)
# array([[1.-1.j, 0.-0.j],
#        [0.-0.j, 1.-1.j]])
```

# nextafter

- **算子名称**：`nextafter`
- **对应函数**：[`numpy.nextafter`](https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html)
- **开发人员**：石向阳@shi-xiangyang225，2678490361@qq.com
- **难易度（预估）**：中
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/15

## 相关内容
- **dtype**：aclTensor，数据类型为所有浮点数类型（与第一个输入参数 dtype 相同）。  
- **shape**：各维长度常为 100 ~ 1000，极限通常不超过 10^4；总元素常为 10^3 ~ 10^5，极限通常不超过 10^7。  
- **维度**：常为 1 ~ 2，极限通常不超过 4。  
- **功能**：逐元素返回从 `x1` 朝 `x2` 方向的下一个可表示浮点数。若 `x1 == x2` 则返回 `x1`。

## np 示例
```python
import numpy as np

eps = np.finfo(np.float64).eps
np.nextafter(1, 2) == eps + 1
# True

np.nextafter([1, 2], [2, 1]) == [eps + 1, 2 - eps]
# array([ True,  True])
```

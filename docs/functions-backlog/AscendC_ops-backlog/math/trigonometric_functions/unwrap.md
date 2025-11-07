# unwrap

- **算子名称**：`unwrap`
- **对应函数**：[`numpy.unwrap`](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html#numpy.unwrap)
- **开发人员**：石向阳@shi-xiangyang225，2678490361@qq.com
- **难易度（预估）**：中
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/19

## 相关内容

- **dtype**：aclTensor，数据类型为所有浮点数类型
- **shape**：各维长度常为 10^2 ~ 10^4，极限通常不超过 10^6；总元素常为 10^4 ~ 10^6，极限通常不超过 10^7
- **维度**：常为 1 ~ 3，极限通常不超过 4
- **功能**：通过对相位序列进行“解包裹”（解跳变），当相位从 π 跳到 -π（或相反）产生不连续时，按给定周期（默认 2π）自动加减整周期，使相位曲线连续、平滑。

## np 示例

```python
import numpy as np

np.unwrap([0, 1, 2, -1, 0], period=4)
# array([0, 1, 2, 3, 4])
```
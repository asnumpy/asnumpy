# interp

- **算子名称**：`interp`
- **对应函数**：[`numpy.interp`](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)
- **开发人员**：石向阳@shi-xiangyang225，2678490361@qq.com
- **难易度（预估）**：中
- **开发状态**：开发中
- **更新时间**：2025.11.4
- **对应issue**：https://gitcode.com/HIT1920/OpenBOAT/issues/14

## 相关内容
- **dtype**：aclTensor，数据类型为所有浮点数类型。  
- **shape**：长度常为 10 ~ 1000，极限通常不超过 10^4；总元素常为 10^3 ~ 10^5。  
- **维度**：1 维。  
- **功能**：对**单调递增**的采样点 `xp` 进行一维线性插值，返回离散数据点 `(xp, fp)` 的**分段线性插值**在 `x` 处的取值；支持越界时的 `left` / `right` 常数填充，以及按 `period` 进行周期化处理（细节参考 NumPy 文档）。

## np 示例
```python
import numpy as np

xp = [1, 2, 3]
fp = [3, 2, 0]

np.interp(2.5, xp, fp)
# 1.0

np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
# array([3.  , 3.  , 2.5 , 0.56, 0.  ])

UNDEF = -99.0
np.interp(3.14, xp, fp, right=UNDEF)
# -99.0
```

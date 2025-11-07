# AsNumpy 文档总览（`docs/`）

本目录聚焦三件事：

1. **功能覆盖现状**（已支持与缺口）；
2. **贡献与认领指南**（外部贡献者如何补齐算子/函数）；
3. **统一的文档与图片资源**。

---

## 目录结构

```text
docs/
├─ contributing/                  # 缺口功能算子贡献相关
│  ├─example/                     # 贡献算子样例模版
│  └─ README.md                   # 缺口功能算子贡献流程介绍
├─ functions-backlog/             # 待补齐清单
│  ├─ AOL-backlog/                # 可用 CANN AOL 组合补齐的条目（按模块组织，示例：array、dtypes...）
│  │  ├─array/
│  │  ├─dtypes/
│  │  ├─...
│  │  └─ README.md                # AOL路线的划分规则与开发计划总览
│  ├─ AscendC_ops-backlog/        # 可认领贡献，需要 Ascend C 自定义开发算子的条目（按模块组织，示例：array、dtypes...）
│  │  ├─array/
│  │  ├─dtypes/
│  │  ├─...
│  │  └─ README.md                # Ascend C自定义算子路线的划分规则与开发计划总览
│  └─ README.md                   # backlog 维护规则、状态与需求分类介绍
├─ functions-supported/           # 已支持/已发布功能（按模块组织，示例：array、dtypes...）
│  ├─array/
│  ├─dtypes/
│  ├─...
│  └─ README.md                # 针对已支持/已发布功能的介绍，划分规则与开发计划总览
├─ images/                        # 文档统一图片资源
└─ README.md                      # 本目录总览与导航
```

> 术语约定：**AOL** 指基于 CANN 的算子库组合实现；**Ascend C** 指使用Ascend C手写自定义核实现。

## AsNumpy API 模块划分

> 下列“模块”用于组织 `functions-backlog/` 与 `functions-supported/` 的条目，也用于贡献指南中的导航。

```text
asnumpy-module/
├─ array/           # 数组结构与创建
├─ math/            # 通用数学运算
├─ logic/           # 比较与逻辑
├─ linalg/          # 线性代数
├─ statistics/      # 聚合与统计
├─ random/          # 随机数
├─ fft/             # 傅立叶变换
├─ polynomial/      # 多项式与插值
├─ dtypes/          # 数据类型管理
└─ utils/           # 工具与调试
```

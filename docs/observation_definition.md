# Observation Definition

1v1 Actor 保持固定 11 维，Critic 保持固定 10 维，并分别由显式特征名称/类型表驱动归一化。`info` 返回特征名称、raw、normalized、饱和维数和饱和比例。

2v2 每个红方智能体使用一个 6 维友机块和两个 11 维敌机块，总计 28 维；两个红方视角整体 shape 为 `(2, 28)`。敌机按距离和 ID 排序，死亡实体保留零槽位并提供 ally/enemy/own alive mask。

集中式全局状态固定 40 维：两个红机失效标志、按 `red_0-blue_0`、`red_0-blue_1`、`red_1-blue_0`、`red_1-blue_1` 排列的四个 9 维配对块，以及两个红机上一动作。失效标志为 `epsilon*(-1)^dam`。paper 模式保留动作索引，training 模式映射到 `[-1,1]`。

2024 年论文正式规模为 3v2；这里的一个友机块加两个敌机块是相同实体块定义的 2v2 适配。

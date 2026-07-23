# Homogeneous 2v2 Environment

双方 pursuit 使用同一个 unique-first 最近目标分配：先选尚未分配的最近目标，全部目标已有分配后才允许重复，距离并列按 UAV ID。规则随机流与物理/伤害 RNG 分离。算法适配器保留 truncated 前的 terminal observation/global state 供 value bootstrap，同时向采样器提供自动 reset 后的下一回合输入。

`CombatMultiEnv` 当前固定支持两个红方外部智能体和两个蓝方规则智能体。动作空间为 `MultiDiscrete([15,15])`，局部观测为 `(2,28)`，全局状态为 `(40,)`。死亡红机行和实体槽位补零，动作被替换为 LEVEL_HOLD，available action mask 只开放该动作。

蓝方先按 ID 顺序分配目标，优先选择尚未分配的最近存活红机；红机不足时允许复用目标。直飞、随机和安全几何追踪策略针对分配目标选动作。所有飞机随后完成同步物理推进和自动攻击。

达到 400 步或 200 秒时，存活数量较多的一方获胜，数量相同为平局。轨迹保存每个决策步及内部物理子步，不在环境核心打印。

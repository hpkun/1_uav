# Environment Design

MAPPO 位于 `uav_env.algorithms.mappo`，只通过 `MAPPOEnvAdapter` 消费既有 Gymnasium 环境，不反向修改动力学、战斗或奖励内核。依赖方向为：runner → adapter/algorithm → env API → observations/rewards/combat → dynamics/core；环境模块不依赖算法模块。

依赖方向保持为：`core/actions → dynamics/entities → combat/observations/rewards/opponents → envs → scripts`。1v1 与 2v2 共用状态、平台、RK4、攻击几何、伤害区间和事件定义，不复制动力学。

每个物理子步都从同一子步起始快照计算全部存活飞机的候选状态，计算结束后统一写回。1v1 保留原同步接口；2v2 中单机越界不会停止其余飞机，只有一方全部失效才可提前结束剩余子步。

多机攻击遵循：每架存活攻击者最多选一个最近可攻击目标；全部攻击先采样名义伤害，再按目标聚合和同步更新。多人攻击同一目标时按名义伤害比例分配有效伤害，击毁归属按有效伤害、攻击距离、攻击者 ID 依次消除并列。

2v2 环境返回两个 28 维局部观测、40 维全局状态、固定实体掩码、每个红方智能体的奖励分解，以及个体奖励均值形式的 Gymnasium 标量 team reward。

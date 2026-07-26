# Experimental Assumptions

- 终局 `win_weights/lose_weights/draw_reward` 是项目假设，不是论文公开参数。
- `paper_2024_exact` 的公式结构来自根目录原论文 PDF 式（21）—（25）；权重数值仍为项目假设。
- 同构 2v2 是固定规模项目实验，不等同于论文 3v2，也不推断异构或任意规模性能。
- 50k/20k 探针及单种子奖励消融只验证训练链路和敏感性，不作为收敛证据。
- `PursuitOpponent` 是固定几何规则空战对手，用于提供主动机动和攻击压力；它不是学习目标，也不是论文预测式对手的完整复现。
- `tail_chase` 仅用于 smoke、基础可学习性和动作/奖励链路检查，其胜率不作为正式空战能力。

| 类别 | 内容 | 状态 |
|---|---|---|
| 论文明确 | 2023 距离奖励的接近项、三段二次函数和端点 | 精确实现并公式测试 |
| 论文明确 | 2023 高度奖励四段函数与 `H_min/H_att/H_adv/H_max` | 精确实现并公式测试 |
| 论文明确 | 2023 角度、速度、稠密与终局奖励 | 正式 `zheng_series_1v1` 模式 |
| 论文明确 | 2024 事件奖励、态势组合、Algorithm 2 已公开分支 | 多机环境使用 |
| 项目约定 | 同步物理和同步伤害 | 消除程序遍历顺序偏置 |
| 项目约定 | 多人击毁归属按有效伤害、距离、ID 排序 | 论文未明确并列归属 |
| 项目约定 | Algorithm 2 中存活且 `raw_dense <= -0.01` 保持原值 | 论文伪代码缺失该分支赋值 |
| 项目约定 | 多机终局 win/lose 权重各约为三等分 | 权重未在附件公开，置于 `project_assumptions` |
| 项目约定 | 2v2 三种初始化范围 | 2v2 不是论文正式 3v2 规模 |
| 项目约定 | 最大高度 5000 米、碰撞距离默认 0 | 论文参数表未明确 |
| 调试模式 | `debug_linear` 距离和高度 | 仅回归调试，不称为论文奖励 |
| 已实现 | 前馈 CTDE MAPPO、GAE、PPO clipping、ValueNormalizer、active/action mask | 已有公式、损失、边界、确定性和 checkpoint 测试 |
| 未实现 | 预测威胁四个子函数、雷达、异构、循环/注意力网络、自博弈和迁移 | 明确保留 |

`damaged` 只表示失效并退出战斗，始终与 `alive` 相反；非致命命中只设置 `ever_hit`。战果统计同时保存名义、有效和过杀伤害，默认报告有效伤害。
# Fixed homogeneous 3v3 audit assumptions

- The current audited target is fixed homogeneous 3v3, not heterogeneous 3v2 and not a transfer experiment.
- The 3v3 Actor input is 45D local observation; the Critic input is 87D global state plus explicit agent identity.
- The baseline is shared feed-forward CTDE MAPPO. It does not implement attention, GRU, self-play, network migration, or the 2024 paper's three-subnetwork transfer Actor.
- The destroyed-agent reward lifecycle is fixed for fixed red slots: pre-destruction normal rewards, one current-step event/damaged dense branch, post-destruction zero dense/event rewards, and terminal allocation retained for every slot.
- The damaged dense branch negative cap is project-defined. It resolves the conflict where the literal Algorithm 2 branch can become positive, while the paper text states damaged UAVs receive negative reward.
- Timeout survivor-count wins are project outcome semantics and must be reported separately from elimination wins.
- The 3v3 audit currently leaves health omission, blue alive/action omission in global state, distance-ranked slot swaps, and timeout terminal semantics as unresolved issues.

# Fixed homogeneous 3v3 V2 assumptions

- V2 is still fixed homogeneous 3v3 only. It is not heterogeneous 3v2, transfer learning, attention, GRU, self-play, radar, or continuous action control.
- The formal V2 Actor input is a 63D project-defined fixed-ID body-frame observation designed to remove the legacy health omission, distance-ranked slot-swap risks, and finite-horizon time aliasing. The earlier 62D form is a development smoke schema only.
- The formal V2 Critic input is a 61D project-defined full-entity global state ordered `red_0..red_2, blue_0..blue_2`, followed by `episode_progress`. The earlier 60D form is a development smoke schema only.
- The V2 train/validation/test scenario is mirrored head-on with bounded jitter. The symmetric no-jitter stress scene is diagnostic only and is not used for checkpoint selection.
- The V2 reward profile keeps the paper-derived situation/terminal structure where applicable, but splits geometry shaping from combat events before Algorithm 2. Combat events are added directly after dense assignment.
- V2 timeout terminal reward is the configured fixed penalty for each red slot; timeout survivor-count outcome statistics remain separate from elimination outcomes.
- Checkpoint schema metadata is a project safety mechanism to prevent full-resume mixing between legacy, old 62D/60D V2, and formal time-aware V2 definitions. Legacy v3 checkpoints without metadata may full-resume only into legacy environments and rely on network state-dict dimensions for final compatibility checks.

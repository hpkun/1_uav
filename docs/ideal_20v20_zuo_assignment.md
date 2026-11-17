# Zuo 负载均衡分配 + ThreePoint-PIP + 预测 CPA-kNN 友方避碰（20v20）

入口：`python -m algorithm.ideal_20v20_zuo_assignment --seed 17 --out <new_directory>`。

本实验在独立的 `ZuoAssignmentTwentyEnv` 中运行。目标分配使用 Zuo 负载均衡公式，制导仍调用项目原有的 ThreePoint-PIP 决策函数，友方避碰调用项目现有的 `predictive_cpa_knn` 层。场景、种子、运动、接触和终止规则不变。论文以蓝方为防御方；本项目录像约定红方为防御方、蓝方为入侵方，算法角色按职能对应。

目标分配采用 Zuo 等人《A Hierarchical Cooperative Interception Framework for Multi-UAV Defense Against Large-Scale Swarm Intrusions》（Drones 2026, 10, 418）第 2.2 节的公开公式：

- 分配：第 2.2 节，式 (4)–(9)。按防御方索引逐一计算距离与相对 +X 方向角度的威胁评分，减去 `0.006 × load^1.2`，选最大效用；平分时选低目标索引。论文将此称为 LB-Hungarian，但公开式 (9) 是顺序 `argmax`，没有给出 Hungarian 子问题；实现遵循公开公式，不声称实现了额外的匈牙利步骤。
- 预测制导继续使用项目的 `estimate_three_point_pip_from_history`；未使用论文的线性预测或合作制导公式。
- 友方避碰复用项目已实现的预测 CPA-kNN 层，参数为 `k=4`、CPA 预测时域 `3 s`、安全距离 `3 m`、预警距离 `8 m`、最大方向混合权重 `0.40`。其输出是方向后处理，不修改 ThreePoint-PIP 预测值。避碰不能在任意场景保证零友方接触，仍以连续几何事件统计为准。

在现有 20v20 环境中，分配只在初始化和已分配目标退出时对受影响的活动防御方重新计算；未受影响的分配保持锁定。该刷新规则属于本项目实验适配，并非论文明确给出的逐步重分配协议。接触事件仍按现有环境判定：只有接触当前分配目标才计为有效拦截。

本实验沿用项目的 20 m/s 双方固定速率、`dt=0.05 s`、Level-2 Bézier 场景及 `0.1 m` 接触阈值。因此结果不是论文使用的速度、步长、场景或成功率的复现。未做参数搜索或训练。

输出 `summary.json`、`episode.json`；后者保存完整逐步状态、动作、分配历史、轨迹和事件。录像可通过 `python -m tools.ideal_interception_replay <episode.json> <video.mp4> --fps 10` 生成。录像使用实际记录轨迹，末帧覆盖完整回合终点。

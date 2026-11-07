# 三维多无人机协同拦截与追逃

默认研究任务是 3v1 区域防护。Pursuer 使用共享策略的 MAPPO，Evader 使用固定行为，双方采用三维多旋翼平动模型。任务成功由非接触式虚拟捕获判定。环境版本为 `uav-interception-v1.0.0`，支持 `1v1`、`3v1`、`5v2`。

## 运行

按仓库运行约定，在 WSL Ubuntu 激活 `uav` Conda 环境。训练、Checkpoint 加载和模型评估要求 CUDA。参考控制与随机基线只运行仿真。依赖见 `requirements.txt`。

```bash
conda activate uav
python -m algorithm.interception validate
python -m algorithm.interception run --scale 1v1 --out outputs/interception/reference_1v1
python -m algorithm.interception run --scenario random --policy random --episodes 3 --out outputs/interception/random_3v1
python -m algorithm.interception train --training configs/interception_smoke.json --updates 1 --out outputs/interception/smoke
python -m algorithm.interception resume --updates 1 --out outputs/interception/smoke
python -m algorithm.interception evaluate outputs/interception/smoke/latest.pt --scenario random --episodes 3 --out outputs/interception/eval
python -m tools.interception_replay outputs/interception/eval/trajectory_000.json outputs/interception/eval/replay.html
python -m pytest -q
```

用浏览器打开生成的 HTML。轨迹显示双方速度方向、保护球、事件位置、步号、仿真时间和任务结果。新运行目录和 HTML 路径必须尚不存在。续跑读取同一运行目录的原子 `latest.pt`，`--updates` 为本次追加更新数。每次 PPO 更新提交 Checkpoint，再提交相应的 `update_*.json`，恢复时补写最后一条更新日志。恢复粒度为一次完整 PPO 更新。

`validate`、`run`、`train` 支持 `--env 参数.json` 和 `--scale`。JSON 按字段覆盖 `InterceptionConfig`，可改变初始距离、速度比、时间常数或目标策略做敏感性检查。不同难度实验使用独立输出目录。模型评估从存档读取物理参数。训练配置可通过 `--training 参数.json` 覆盖。

`validate`、`run`、`evaluate` 支持 `--scenario fixed|random|custom` 与 `--initial 初态.json`。固定与随机场景可用 `{"distance": 45}` 覆盖目标初始距离。自定义初态使用 `pursuers` 和 `evaders` 两个数组，行数须对应规模，列为 `[x,y,z,vx,vy,vz]`。参考控制器用于验证默认物理设置下的可解性，适用条件是 `goal_seeking`、双方时间常数相等且 Pursuer 比目标指令速度快。它按相对制动距离切换速度并保留捕获间隔。

## 环境协议

| 数据 | 形状及单位 |
| --- | --- |
| Pursuer 与 Evader 物理状态 | `[5,6]`、`[2,6]`，float64，位置 m，速度 m/s |
| Pursuer 动作 | `[5,3]`，有效槽每维 [0,1]。物理指令为三维期望速度 |
| Actor 观测 | `[5,52]`，float32，每个有效 Pursuer 一行 |
| Critic 输入与输出 | `[54]` 归一化集中状态，输出 5 个 Pursuer 价值 |
| 奖励和实体掩码 | `[5]` float32 团队奖励，Pursuer `[5]` 与 Evader `[2]` bool 掩码 |
| 连续捕获计时 | `[5,2]` 秒数，同时提供除以 dt 的等效步数 |

`reset(seed, options)` 返回 `(observations, info)`。`step(actions)` 返回 `(observations, rewards, terminated, truncated, info)`。`snapshot()` 保存完整配置、双方状态、掩码、捕获计时、目标机动相位、仿真步与时间、累计事件、诊断和 RNG。`restore(snapshot)` 检查版本与配置。

局部观测依次为自身位置与速度、保护球中心相对位置、剩余时间比例、2 个目标槽、4 个按稳定 ID 排序的其他 Pursuer 槽。每个实体槽为相对位置、相对速度与有效标志。全目标信息可观测。集中状态依次为 5 个 Pursuer 槽、2 个 Evader 槽及保护球中心、半径、剩余时间。实体槽含绝对位置、速度及有效标志。位置与半径除以 80 m，速度除以 8 m/s。补零和失效槽置零。事件 ID 0–4 对应 Pursuer，5–6 对应 Evader。

## 第一版工程参数

下表及配置类中的其他默认值均为自主仿真实验参数。

| 项目 | 默认值 |
| --- | --- |
| 控制周期、回合上限 | 0.1 s、600 步（60 s） |
| Pursuer / Evader 速度模长上限 | 8 / 6 m/s |
| Pursuer / Evader 速度时间常数 | 1.5 / 1.5 s |
| 初始化区域 | x,y ∈ [−80,80] m，z ∈ [3,40] m |
| 保护球 | 中心 (0,0,15) m，半径 5 m |
| 目标生成 | 半径 35–55 m，随机方位与小仰角，双目标分布在相对方向 |
| Pursuer 防护布置 | 沿目标方位距中心 14 m，横向槽间距 8 m，初速为零 |
| 捕获 | 同一机对距离 ≤2 m，相对速度 ≤1.5 m/s，连续 5×dt = 0.5 s |
| 碰撞 | 任一对有效实体距离 ≤0.6 m |
| 事件时间分辨率、最小间距精度 | 1e−7 s、1 mm |
| 初态采样 | 最多 128 次，拒绝越界、重叠、初始捕获区/突破与必要可达界不满足 |
| 目标速度比例 | 1.0 |
| 恒速方向 | (−1,0,0) |
| 有界机动幅值、周期 | 0.45、8 s |
| 反应规避范围、增益 | 15 m、1.2 |
| 团队奖励 | 捕获 +5/目标，全部捕获 +10，突破 −10，碰撞 −10 |
| 进度、时间代价 | 进度权重 1、参考距离 40 m，每完整步 −0.01 |

区域坐标用于初始化约束，运行中位置由解析运动推进。保护球与实体碰撞决定几何失败。可达检查使用整个时域的双边位移上界，固定参考控制轨迹独立验证基础场景的可解性。

三维动作先映射 `w=2a−1`，再将 `w` 投影到单位速度球并乘以各方速度上限。速度采用指数混合，位置采用同一常值期望速度下的解析积分。最大隐含加速度为 `2 vmax/tau`，默认 Pursuer 10.667 m/s²、Evader 8 m/s²。从静止发出满速指令时分别为 5.333 与 4 m/s²。

四种目标策略为 `constant_velocity`、`goal_seeking`（默认）、`bounded_maneuver` 与 `reactive_evasion`。有界机动用固定相位与当前时刻生成周期偏转。反应规避混合当前目标方向与感知范围内 Pursuer 的位置斥力，并投影到速度球。它只使用当前状态。

步内事件沿解析轨迹搜索。用二阶导数界约束曲线与弦的误差，再细分无法确定的区间，叶区间按弦和小数值包络结算。捕获要求距离与相对速度的阈值区间持续相交。两者任何一项离开阈值都会中断该机对计时。先处理最早事件。容差内同刻按碰撞、突破、捕获顺序处理，同类按时间与稳定 ID。捕获立即停止该目标运动并移除其活动槽。全部捕获、碰撞和突破产生 terminated。达到时限产生 truncated。数值故障显式抛出，训练停在上一提交点，评估单列失败。

进度按步开始时有效目标的最近 Pursuer 距离均值差计算，捕获目标使用捕获时刻的两方位置。团队奖励复制到每个有效 Pursuer，统计按团队值计一次。时间代价按终止步实际时长折算。进度是启发式距离塑形。评估捕获时间仅统计成功回合，报告样本数。Control Effort 是有效 Pursuer 的 `∫||u||² dt` 均值，单位 m²/s，属于指令强度指标。

## 训练与历史档案

Actor 和 Critic 使用 128–128 Tanh，Actor 输出三组 Beta 参数。PPO 只用有效 Pursuer 的样本，真实终止不 bootstrap，超时使用自动重置前的真实末状态 bootstrap，两类边界都切断跨回合的 GAE。训练默认 4 个 spawn worker、每次采样 128 步、4 轮 PPO、minibatch 256、seed 17。Actor/Critic 学习率为 3e−4/5e−4，γ=0.99、λ=0.95、clip=0.2、entropy=0.01、梯度范数上限 0.5。训练与评估环境使用独立 SeedSequence 命名空间。

新 Checkpoint 包含版本、维度、环境和训练配置、模型、优化器、RNG、采样 worker 及进度。加载前检查专用协议标头。请只加载可信本地存档。历史协议的模型、资料与实验输出见 `archive/README.md`。历史执行代码以 Git 提交定位。

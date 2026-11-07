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

## 科研记录、定期评估和动态回放

```bash
root=outputs/interception/research_new
python -m algorithm.interception run --scenario fixed --out "$root/reference" --gzip
python -m algorithm.interception run --policy random --scenario random --episodes 3 --out "$root/random"
python -m algorithm.interception train --training configs/interception_smoke.json --updates 1 \
  --evaluation-interval-updates 1 --evaluation-episodes 2 --evaluation-seed-base 1000 \
  --evaluation-scenario random --log-interval 1 --out "$root/train"
python -m algorithm.interception resume --updates 1 --out "$root/train"
python -m algorithm.interception evaluate "$root/train/best.pt" --scenario random \
  --seed 2000 --episodes 3 --out "$root/best_eval"
python -m algorithm.interception evaluate "$root/train/latest.pt" --scenario fixed \
  --out "$root/latest_eval"
python -m tools.interception_replay "$root/reference/trajectory_000.json.gz" "$root/replay.html"
python -m tools.interception_replay "$root/reference/trajectory_000.json.gz" "$root/replay.mp4" \
  --fps 15 --width 960 --height 540 --elevation 25 --azimuth -55
python -m tools.interception_report "$root/train" --out "$root/report"
# 两个相同实验协议、相同预算、不同训练 seed 的运行目录
python -m tools.interception_report outputs/interception/seed17 outputs/interception/seed18 \
  --out outputs/interception/multi_seed_report
```

- HTML 内嵌 Plotly，支持离线播放、暂停、重播、前后步进、拖动时间轴及 0.5×/1×/2×/4×。滑块定位控制步或步内事件关键帧，播放在物理时钟上使用额外的 30 Hz 解析采样。支持鼠标轨道旋转和滚轮缩放。面板分列归一化策略动作、期望速度指令和真实物理速度，累计诊断注明最近一次记录时间。旧轨迹缺失速度指令时，中间画面保持最近记录位置，当前速度和相对量显示 N/A。
- 两种格式均在被捕获 Evader 的真实事件位置显示一个红色 X。普通目标点由 X 接替，历史轨迹保留。碰撞使用橙色菱形，显示于双方记录位置的中点。突破使用黄色方形，显示于 Evader 的记录位置。标记从实际事件时刻起保留，MP4 在事件时刻或之后的首个输出采样帧出现标记。
- MP4 的 N 帧均匀覆盖物理区间 `[0,T]`，包含起止状态。非零 T 取 `N=max(2,ceil(T×fps))`，物理间隔为 `T/(N-1)`，固定帧率容器时长为 `N/fps`。零时长保留一个静帧。常规轨迹的容器时长误差小于一帧。极短轨迹优先保留两个不同时间的端点。需要 FFmpeg，可通过 `conda install -c conda-forge ffmpeg` 安装。导出器优先使用可用的 H.264 编码器，也支持 FFmpeg 的 MPEG-4 编码器。
- 轨迹 v2 保存归一化策略动作、实际执行期望速度和真实状态。动作及指令对应结束于该帧的控制区间。记录还包括双方掩码、速度模长、距离和相对速度矩阵、捕获计时、奖励、模型 SHA256 及真实事件坐标。无效机对为空。读取器兼容原始 JSON 轨迹与 gzip。旧轨迹缺失的诊断显示 N/A。
- `training_metrics.jsonl` 按 PPO 更新记录损失、熵、KL、clip fraction、有效样本、裁剪前梯度范数、学习率及采样和优化时间。速度为本次采样联合步数除以采样墙钟时间。更新耗时包含采样和优化。一个联合步指一个并行环境的一次联合转移。
- `episode_metrics.jsonl` 每个完成回合一条，包含 worker 及其回合序号、结局、实际时长、捕获/突破/碰撞数、最小间距、指令强度和奖励分量。回报累计实际 float32 团队奖励，分量求和允许浮点舍入误差。终端展示最近最多 100 个已完成回合，窗口为空时显示 N/A。
- 定期评估默认关闭。间隔设为正数后启用，同一运行的协议在启动时固定，默认随机场景和 Beta 均值动作。可选 `--evaluation-scenario fixed` 或 `--no-evaluation-deterministic`。单独评估的随机动作选项为 `--stochastic`。评估使用独立环境和固定种子，并恢复 CPU/CUDA Torch RNG。
- `evaluation_*.json` 保存每局及聚合结果，`evaluation_history.csv` 是派生摘要。结局率分母为全部评估回合，单目标捕获率分母为该 ID 初始有效的回合数，部分捕获表示捕获至少一个且少于全部目标。捕获时长均值和中位数仅统计成功局，另给出全局最小间距、每局最小间距均值和全部回合时长中位数。Control Effort 仍为指令强度积分。
- `best.pt` 是胜出评估提交对应的 Checkpoint 副本，`best.json` 记录来源、更新数、联合步数、协议、分数和 SHA256。排序依次最大化成功率、最小化突破率、碰撞率、成功平均捕获时长，最后保留更早更新。缺失捕获时长按最差排序。
- 恢复核对运行身份、配置、连续更新记录及提交链。训练提交后可补写末条记录，评估未提交可补跑，已提交评估可补发 Best，JSONL 和 CSV 可重建。遇到超前日志、记录缺口或缺失且无法重建的历史 Best 会明确报错。已有拦截 v1 Checkpoint 可独立评估。续训要求本版科研提交元数据和对应运行目录。
- 离线报告读取 JSON 记录与存档 ZIP 的轻量元数据，输出 `report.md`、`summary.json` 和 `training_curves.png`。多种子报告按 Update 对齐，输出均值、样本标准差和有效种子数，检查预算和配置一致性。无需 CUDA。

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

# 三维多无人机协同拦截与追逃

环境协议为 `uav-interception-v3.0.0`。默认研究任务仍为 S2，3v1 三维区域防护。Pursuer 使用共享策略，Evader 使用规则策略，双方采用三维多旋翼平动模型。只有安全捕获全部目标才记为捕获成功。

## 地面设施防护（新增、推荐的独立实验场景）

旧球形空中保护区的代码默认值保持不变，供历史基线复核；新的**贴地柱状防护空域**通过独立配置启用，不覆盖任何历史训练结果。详细假设及来源见 [地面设施场景说明](docs/ground_facility_v3.md)。

```bash
python -m algorithm.interception validate --env configs/interception_ground_facility_v3.json --scenario random --seed 42
python -m algorithm.interception run --env configs/interception_ground_facility_v3.json --scenario random --policy random --episodes 2 --out outputs/ground_facility_v3/random_check
python -m pytest -q tests/test_ground_facility.py tests/test_interception.py tests/test_interception_replay.py
```

新的地面设施配置采用：半径 12 m、从地面向上 22 m 的防护柱体；初始入侵航迹距飞行导航点 85–115 m；防御平台水平部署半径 25–40 m；双方最大速度模长 12/9 m/s、垂直指令上限 4/3 m/s；机体简化包络半径 0.25 m/架，**接触判据为 0.50 m 中心距**。这些是可审计的仿真假设，不是现实拦截成功范围。**接触仍被视为安全失败，成功捕获仍是非接触式抽象事件，绝不等价于自损撞击。** 新协议不能直接恢复历史 v2 模型，必须单独重训。


| 任务 | 规模 | 初始化与目标策略 |
| --- | --- | --- |
| S0 | 1v1 | 基础追踪，固定初态沿目标方向部署，默认目标点导航 |
| S1 | 3v1 | 保护区周围分散部署，目标点导航 |
| S2 | 3v1 | 与 S1 共用初态分布，目标使用三维反应规避 |
| S3 | 5v2 | 分散部署，两个目标从不同方位扇区进入并独立规避 |

## 运行

按仓库运行约定，在 WSL Ubuntu 激活 `uav` Conda 环境。训练、Checkpoint 加载和模型评估要求 CUDA。参考控制与随机基线只运行仿真。依赖见 `requirements.txt`。

```bash
conda activate uav
python -m algorithm.interception validate
python -m algorithm.interception run --task S0 --out outputs/interception/reference_1v1
python -m algorithm.interception run --task S1 --out outputs/interception/reference_S1
python -m algorithm.interception run --task S3 --out outputs/interception/reference_S3
python -m tools.interception_pair_audit --seed 1000 --episodes 20 --out outputs/interception/paired
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
- 两种格式均在被捕获 Evader 的真实事件位置显示一个红色 X。普通目标点由 X 接替，历史轨迹保留。碰撞使用橙色菱形并显示在双方记录位置的中点。突破用黄色方形、Pursuer 越界用紫色十字、Evader 撤离用青色空心圆。Near Miss 保存在轨迹和统计中。标记从实际事件时刻起保留，MP4 在事件时刻或之后的首个输出采样帧出现标记。
- MP4 的 N 帧均匀覆盖物理区间 `[0,T]`，包含起止状态。非零 T 取 `N=max(2,ceil(T×fps))`，物理间隔为 `T/(N-1)`，固定帧率容器时长为 `N/fps`。零时长保留一个静帧。常规轨迹的容器时长误差小于一帧。极短轨迹优先保留两个不同时间的端点。需要 FFmpeg，可通过 `conda install -c conda-forge ffmpeg` 安装。导出器优先使用可用的 H.264 编码器，也支持 FFmpeg 的 MPEG-4 编码器。
- 轨迹 v2 保存归一化策略动作、实际执行期望速度和真实状态。动作及指令对应结束于该帧的控制区间。记录还包括双方掩码、速度模长、距离和相对速度矩阵、捕获计时、奖励、模型 SHA256 及真实事件坐标。无效机对为空。读取器兼容原始 JSON 轨迹与 gzip。旧轨迹缺失的诊断显示 N/A。
- `training_metrics.jsonl` 按 PPO 更新记录损失、熵、KL、clip fraction、有效样本、裁剪前梯度范数、学习率及采样和优化时间。速度为本次采样联合步数除以采样墙钟时间。更新耗时包含采样和优化。一个联合步指一个并行环境的一次联合转移。
- `episode_metrics.jsonl` 每个完成回合一条，包含 worker 及其回合序号、结局、实际时长、捕获/突破/碰撞数、最小间距、指令强度和奖励分量。回报累计实际 float32 团队奖励，分量求和允许浮点舍入误差。终端展示最近最多 100 个已完成回合，窗口为空时显示 N/A。
- 定期评估默认关闭。间隔设为正数后启用，同一运行的协议在启动时固定，默认随机场景和 Beta 均值动作。可选 `--evaluation-scenario fixed` 或 `--no-evaluation-deterministic`。单独评估的随机动作选项为 `--stochastic`。评估使用独立环境和固定种子，并恢复 CPU/CUDA Torch RNG。
- `evaluation_*.json` 保存每局及聚合结果，`evaluation_history.csv` 是派生摘要。结局率分母为全部评估回合。捕获成功只统计全部捕获；防护成功统计全部捕获和完整坚持到时限的回合，撤离、突破和安全失败单列。部分捕获表示至少捕获一个且少于全部目标。单目标捕获率分母为该 ID 初始有效的回合数。全捕获平均时长只在成功局计算。每个 Pursuer 的参与量为在安全距离带且满足相对速度阈值的机对秒数，同一时刻多个合格目标可分别贡献。该量衡量直接捕获参与，未衡量牵制、封堵或覆盖。
- `best.pt` 是胜出评估提交对应的 Checkpoint 副本，`best.json` 记录来源、更新数、联合步数、协议、分数和 SHA256。排序依次最大化捕获成功率，最小化突破率、碰撞率、Pursuer 越界率、Evader 撤离率及成功局的平均捕获时长，最后保留更早更新。缺失捕获时长按最差排序。
- 恢复核对运行身份、配置、连续更新记录及提交链。训练提交后可补写末条记录，评估未提交可补跑，已提交评估可补发 Best，JSONL 和 CSV 可重建。遇到超前日志、记录缺口或缺失且无法重建的历史 Best 会明确报错。v3 拒绝 v1/v2 Checkpoint 的训练恢复和评估；v1/v2 轨迹仍可按历史记录回放。
- 离线报告读取 JSON 记录与存档 ZIP 的轻量元数据，输出 `report.md`、`summary.json` 和 `training_curves.png`。多种子报告按 Update 对齐，输出均值、样本标准差和有效种子数，检查预算和配置一致性。无需 CUDA。

`validate`、`run`、`train` 支持 `--task S0|S1|S2|S3`、`--env 参数.json` 和 `--scale`。仅指定规模时，1v1、3v1、5v2 分别映射为 S0、S2、S3。JSON 按字段覆盖 `InterceptionConfig`，敏感性实验使用独立输出目录。模型评估从存档读取环境配置，训练配置可通过 `--training 参数.json` 覆盖。

`validate`、`run`、`evaluate` 支持 `--scenario fixed|random|custom` 与 `--initial 初态.json`。固定与随机场景可用 `{"distance": 45}` 覆盖目标初始距离。自定义初态使用 `pursuers` 和 `evaders` 两个数组，行数须对应规模，列为 `[x,y,z,vx,vy,vz]`。S0 参考控制按制动距离调节速度。S1–S3 参考控制跟踪目标后方的安全间隔，使用当前位置分配追踪者。配对审计按同一 3v1 初态取嵌套的 1、2、3 机子集，保存实际部署 ID、初态与结果。反应式目标的轨迹随所见 Pursuer 改变。

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

## 历史球形基线参数（保留旧默认值）

下表及配置类中的其他默认值均为自主仿真实验参数。

| 项目 | 默认值 |
| --- | --- |
| 控制周期、回合上限 | 0.1 s、600 步（60 s） |
| Pursuer / Evader 速度模长上限 | 8 / 6 m/s |
| Pursuer / Evader 速度时间常数 | 1.5 / 1.5 s |
| 三维运行空域 | x,y ∈ [−80,80] m，z ∈ [3,40] m，整个实体包络须在内 |
| 保护球 | 中心 (0,0,15) m，半径 5 m |
| 目标生成 | 距中心 35–55 m，高度 9–25 m，方位分扇区，初速 0–2 m/s |
| Pursuer 防护布置 | 水平半径 12–20 m，高度 8–24 m，独立方位扇区，初速 0–1.5 m/s |
| 捕获 | 同一机对中心距 1.1–3.0 m、相对速度 ≤1.5 m/s，连续 0.5 s |
| 碰撞与 Near Miss | 两类实体半径均为 0.3 m；碰撞阈值为机对半径之和；Near Miss 表面间距阈值 0.3 m |
| 事件时间分辨率、最小间距精度 | 1e−7 s、1 mm |
| 初态采样 | 最多 128 次，随机与固定生成额外要求 1 m 初始表面净距，自定义检查实体不重叠 |
| 目标速度比例 | 1.0 |
| 恒速方向 | (−1,0,0) |
| 有界机动幅值、周期 | 0.45、8 s |
| 反应规避 | 感知 15 m，预测 2 s，危险距离 5 m，风险权重 2，方向连续权重 0.15，锥半角 55° |
| 团队奖励 | 捕获 +5/目标，全部捕获 +10，突破、碰撞和 Pursuer 越界各 −10 |
| 进度、时间代价 | 进度权重 1、参考距离 40 m，每完整步 −0.01 |

保护球突破使用 Evader 中心，空域越界使用整个碰撞球。空域首次越界通过解析轨迹的单轴转折点和根搜索检测，终止于实际事件时刻。可达检查使用双边位移上界排除数学上不可能的初态。

三维动作先映射 `w=2a−1`，再将 `w` 投影到单位速度球并乘以各方速度上限。速度采用指数混合，位置采用同一常值期望速度下的解析积分。最大隐含加速度为 `2 vmax/tau`，默认 Pursuer 10.667 m/s²、Evader 8 m/s²。从静止发出满速指令时分别为 5.333 与 4 m/s²。

目标策略为 `constant_velocity`、`goal_seeking`、`bounded_maneuver` 与 `reactive_evasion`。反应规避在导航方向和两个三维锥环上的 16 个方向中评分。目标端按自身解析响应预测，Pursuer 假设保持当前速度。评分使用导航方向投影、预测最小间距的风险和当前速度方向连续性，有预测不越界的候选时优先从中选。候选预测以 8 个时点采样，真实物理事件仍由连续检测判定。策略依赖当前状态和固定任务配置。

步内事件沿解析轨迹搜索。捕获将外球内、内球外及相对速度合格的时间窗口求交，任何一项中断即清零该机对时钟。先处理最早事件，容差内同刻按碰撞、突破、越界或撤离、捕获排序。捕获立即冻结目标并移除其活动槽，其余目标继续运行。Near Miss 统计一次完整危险接近或回合结束时尚未碰撞的危险接近。同一次接近若发展为碰撞，则只记录碰撞。全部捕获、碰撞、突破、Pursuer 越界、Evader 撤离产生 terminated，达到时限产生 truncated。数值故障显式抛出，训练停在上一提交点，评估单列失败。

进度按步开始时有效目标的最近 Pursuer 距离均值差计算，捕获目标使用捕获时刻的两方位置。团队奖励复制到每个有效 Pursuer，统计按团队值计一次。时间代价按终止步实际时长折算。进度是启发式距离塑形。评估捕获时间仅统计成功回合，报告样本数。Control Effort 是有效 Pursuer 的 `∫||u||² dt` 均值，单位 m²/s，属于指令强度指标。

## 训练与历史档案

Actor 和 Critic 使用 128–128 Tanh，Actor 输出三组 Beta 参数。PPO 只用有效 Pursuer 的样本，真实终止不 bootstrap，超时使用自动重置前的真实末状态 bootstrap，两类边界都切断跨回合的 GAE。训练默认 4 个 spawn worker、每次采样 128 步、4 轮 PPO、minibatch 256、seed 17。Actor/Critic 学习率为 3e−4/5e−4，γ=0.99、λ=0.95、clip=0.2、entropy=0.01、梯度范数上限 0.5。训练与评估环境使用独立 SeedSequence 命名空间。

新 Checkpoint 包含版本、维度、环境和训练配置、模型、优化器、RNG、采样 worker 及进度。加载前检查专用协议标头。请只加载可信本地存档。历史协议的模型、资料与实验输出见 `archive/README.md`。历史执行代码以 Git 提交定位。

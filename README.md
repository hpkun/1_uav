# 三维多无人机协同拦截与追逃

环境协议为 `uav-interception-v4.0.0`。默认任务为 S2，3v1 地面设施防护。红方 Pursuer 使用 MAPPO 共享策略或参考控制器，蓝方 Evader 使用规则策略。全部蓝方通过异方几何接触退出才记为全拦截成功。

历史球形基线及 v3 独立地面设施配置保留在 Git 历史和 [v3 场景说明](docs/ground_facility_v3.md) 中。旧模型和实验输出原样保留；旧轨迹可只读回放。

| 任务 | 规模 | 初始化与目标策略 |
| --- | --- | --- |
| S0 | 1v1 | 基础追踪，固定初态沿目标方向部署，默认目标点导航 |
| S1 | 3v1 | 保护区周围分散部署，目标点导航 |
| S2 | 3v1 | 与 S1 共用初态分布，目标使用三维反应规避 |
| S3 | 5v2 | 分散部署，两个目标从不同方位扇区进入并独立规避 |

## 独立高速场景

`configs/interception_high_speed.json` 保持 0.1 s 控制周期，使用 300 步上限、650 m 水平半宽、25 m 保护半径、35 m 柱高和 18 m 导航点高度。红方最大速度 25 m/s、初速 15–20 m/s，蓝方默认最大速度 28 m/s、初速 18–23 m/s。红方阵型整体随机旋转，每机扰动不超过 0.15 rad，采用切向巡航并保留 20% 向设施的径向速度分量。蓝方初速指向导航点。双方共享 2.5 s 响应时间、6 m/s² 水平加速度、5 m/s 上升和 4 m/s 下降上限。观测位置和速度尺度为 650 m 与 30 m/s。1 m 接触及奖励权重沿用 v4，35 m 友机预警仅供统计。

```bash
root=outputs/interception_high_speed_new
python -m algorithm.interception run --env configs/interception_high_speed.json --task S2 \
  --policy predictive --scenario fixed --seed 1200 --gzip --out "$root/reference"
python -m tools.interception_high_speed_checks --out "$root/physics.json"
python -m tools.interception_high_speed_audit --seed 1200 --episodes 50 --out "$root/audit"
python -m tools.interception_replay "$root/reference/trajectory_000.json.gz" "$root/reference/replay.html"
python -m tools.interception_replay "$root/reference/trajectory_000.json.gz" "$root/reference/replay.mp4"
```

高速审计分别改变蓝方速度、双方水平加速度、预测时域及红方参战数量。逐目标最近点沿连续轨迹求得，距离误差上限为 1 mm。进入 10 m、重复接近和接近时加速度饱和是观察指标。反应式蓝方会随红方轨迹改变，配对比较共享初态和蓝方规则。

### 三档速度配对

在相同高速场景几何中比较红/蓝上限 6/7、12/14、18/21 m/s，初速度使用统一速度比例，时限 90 s。命令逐种子核验初始位置、掩码和成比例速度。主实验采用 10 s 预测时域，S2 各档与低速其余场景另测 20/40 s，S2 另测同种子单机子集。距离阶梯是诊断，1 m 接触事件仍由环境独立判定。

```bash
conda activate uav
root=outputs/interception_speed_comparison_new
for tier in low medium upper; do
  for task in S0 S1 S2 S3; do
    python -m tools.interception_speed_tiers --tier "$tier" --task "$task" \
      --seed 1200 --episodes 50 --out "$root/audit/$tier/$task"
  done
done
python -m tools.interception_speed_report --audit "$root/audit" --out "$root/report"
```

## 运行

按仓库运行约定，在 WSL Ubuntu 激活 `uav` Conda 环境。训练、Checkpoint 加载和模型评估要求 CUDA。参考控制与随机基线只运行仿真。依赖见 `requirements.txt`。

```bash
conda activate uav
python -m algorithm.interception validate
python -m algorithm.interception run --task S0 --policy predictive --out outputs/interception/reference_1v1
python -m algorithm.interception run --task S1 --policy predictive --out outputs/interception/reference_S1
python -m algorithm.interception run --task S3 --policy predictive --out outputs/interception/reference_S3
python -m algorithm.interception run --scale 3v3 --scenario fixed --gzip --out outputs/interception/reference_3v3
python -m tools.interception_difficulty_audit --seed 1000 --episodes 10 --out outputs/interception/v4_audit
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
python -m algorithm.interception run --scenario fixed --policy predictive --out "$root/reference" --gzip
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

可以给 HTML 或 MP4 叠加蓝方预设参考航迹。在同一份记录上使用同一参考种子，两种模式共享每架蓝机的记录起点与场景导航点。旧保护球记录没有导航偏移时，使用保护球中心。蓝色虚线为参考，实线为真实运动。曲线在导出时生成一次，回放时保持固定。HTML 的 **Blue reference paths** 面板和 MP4 的 `comment` 元数据记录种子、起终点、控制点、长度和参考时长。省略参数时按原样导出。

```bash
python -m tools.interception_replay "$source" "$out/straight.html" --blue-reference straight --reference-seed 17
python -m tools.interception_replay "$source" "$out/preset_bezier.mp4" --blue-reference preset_bezier --reference-seed 17
ffprobe -v error -show_entries format_tags=comment -of json "$out/preset_bezier.mp4"
```

- 两种格式均画出地面设施和保护柱。红色为 Pursuer，蓝色为 Evader。每次接触拦截在双方事件位置的中点显示一个绿色 X，双方历史轨迹停在各自真实端点。友机及目标方同侧接触使用橙色菱形，突破用黄色方形，Pursuer 越界用紫色十字，Evader 撤离用青色空心圆。友机预警只记录数据。旧版捕获仍按旧轨迹在 Evader 事件位置显示 X。
- MP4 的 N 帧均匀覆盖物理区间 `[0,T]`，包含起止状态。非零 T 取 `N=max(2,ceil(T×fps))`，物理间隔为 `T/(N-1)`，固定帧率容器时长为 `N/fps`。零时长保留一个静帧。常规轨迹的容器时长误差小于一帧。极短轨迹优先保留两个不同时间的端点。需要 FFmpeg，可通过 `conda install -c conda-forge ffmpeg` 安装。导出器优先使用可用的 H.264 编码器，也支持 FFmpeg 的 MPEG-4 编码器。
- 轨迹记录原始动作、速度请求、执行指令、实际状态、步初加速度、有效时长、掩码、相对几何、奖励和事件位置。速度与瞬时加速度由同一解析轨迹确定。读取器兼容 JSON 和 gzip。
- `training_metrics.jsonl` 按 PPO 更新记录损失、熵、KL、clip fraction、有效样本、裁剪前梯度范数、学习率及采样和优化时间。速度为本次采样联合步数除以采样墙钟时间。更新耗时包含采样和优化。一个联合步指一个并行环境的一次联合转移。
- `episode_metrics.jsonl` 每个完成回合一条，包含 worker 及其回合序号、结局、实际时长、捕获/突破/碰撞数、最小间距、指令强度和奖励分量。回报累计实际 float32 团队奖励，分量求和允许浮点舍入误差。终端展示最近最多 100 个已完成回合，窗口为空时显示 N/A。
- 定期评估默认关闭。间隔设为正数后启用，同一运行的协议在启动时固定，默认随机场景和 Beta 均值动作。可选 `--evaluation-scenario fixed` 或 `--no-evaluation-deterministic`。单独评估的随机动作选项为 `--stochastic`。评估使用独立环境和固定种子，并恢复 CPU/CUDA Torch RNG。
- `evaluation_*.json` 保存每局及聚合结果，`evaluation_history.csv` 是派生摘要。结局率分母为全部评估回合。捕获成功只统计全部捕获；防护成功统计全部捕获和完整坚持到时限的回合，撤离、突破和安全失败单列。部分捕获表示至少捕获一个且少于全部目标。单目标捕获率分母为该 ID 初始有效的回合数。全捕获平均时长只在成功局计算。同时记录异方接触导致的红方退出数量、同侧接触、友机预警和逐目标接触时刻。
- `best.pt` 是胜出评估提交对应的 Checkpoint 副本，`best.json` 记录来源、更新数、联合步数、协议、分数和 SHA256。排序依次最大化捕获成功率，最小化突破率、友机碰撞率、Evader 同侧碰撞率、Pursuer 越界率、Evader 撤离率及成功局的平均捕获时长，最后保留更早更新。缺失捕获时长按最差排序。
- 恢复核对运行身份、配置、连续更新记录及提交链。训练提交后可补写末条记录，评估未提交可补跑，已提交评估可补发 Best，JSONL 和 CSV 可重建。遇到超前日志、记录缺口或缺失且无法重建的历史 Best 会明确报错。v4 拒绝 v1–v3 Checkpoint 的训练恢复和评估。历史轨迹仍可按原协议回放。
- 离线报告读取 JSON 记录与存档 ZIP 的轻量元数据，输出 `report.md`、`summary.json` 和 `training_curves.png`。多种子报告按 Update 对齐，输出均值、样本标准差和有效种子数，检查预算和配置一致性。无需 CUDA。

`validate`、`run`、`train` 支持 `--task S0|S1|S2|S3`、`--env 参数.json` 和 `--scale`。仅指定规模时，1v1、3v1、5v2 分别映射为 S0、S2、S3。JSON 按字段覆盖 `InterceptionConfig`，敏感性实验使用独立输出目录。模型评估从存档读取环境配置，训练配置可通过 `--training 参数.json` 覆盖。

`validate` 和 `run` 另支持 `--scale 3v3`（任务 `reference_3v3`）。该模式使用三个目标槽和现有拦截规则，用于参考控制或随机控制仿真及录像。正式 MAPPO 训练和 Checkpoint 保持两目标槽协议。

`validate`、`run`、`evaluate` 支持 `--scenario fixed|random|custom` 与 `--initial 初态.json`。固定与随机场景可用 `{"distance": 115}` 覆盖目标初始距离。自定义初态使用 `pursuers` 和 `evaders` 两个数组，行数须对应规模，列为 `[x,y,z,vx,vy,vz]`。控制器 `stationary` 发零速度指令，随机初速仍按原动力学衰减。`reference` 直追当前位置，`predictive` 使用当前状态和目标点导航假设搜索最早可达相遇点。配对难度审计对同一规模复用完整初态，只切换控制器、目标速度上限或 S1/S2 行为。

## 环境协议

| 数据 | 形状及单位 |
| --- | --- |
| Pursuer 与 Evader 物理状态 | `[5,6]`、`[2,6]`，float64，位置 m，速度 m/s |
| Pursuer 动作 | `[5,3]`，有效槽每维 [0,1]。物理指令为三维期望速度 |
| Actor 观测 | `[5,52]`，float32，每个有效 Pursuer 一行 |
| Critic 输入与输出 | `[54]` 归一化集中状态，输出 5 个 Pursuer 价值 |
| 奖励和实体掩码 | `[5]` float32 团队奖励，Pursuer `[5]` 与 Evader `[2]` bool 掩码 |

`reset(seed, options)` 返回 `(observations, info)`。`step(actions)` 返回 `(observations, rewards, terminated, truncated, info)`。`snapshot()` 保存完整配置、双方状态、掩码、目标参考轨迹起点及机动相位、仿真步与时间、累计事件、诊断和 RNG。`restore(snapshot)` 检查版本与配置。

局部观测依次为自身位置与速度、地面设施中心相对位置、剩余时间比例、2 个目标槽、4 个按稳定 ID 排序的其他 Pursuer 槽。每个实体槽为相对位置、相对速度与有效标志。全目标信息可观测。集中状态依次为 5 个 Pursuer 槽、2 个 Evader 槽及地面设施中心、半径、剩余时间。实体槽含绝对位置、速度及有效标志。位置与半径除以 160 m，速度除以 13 m/s。补零和失效槽置零。事件 ID 0–4 对应 Pursuer，5–6 对应 Evader。

## v4 默认工程参数

| 项目 | 默认值 |
| --- | --- |
| 控制周期、时限 | 0.1 s、600 步 |
| 速度模长、响应时间 | Pursuer 12 m/s，Evader 13 m/s；均为 2.5 s |
| 垂直速度、水平加速度 | 下降不超过 2 m/s，上升不超过 3 m/s；水平加速度不超过 4 m/s² |
| 空域 | x、y ∈ [−160,160] m，z ∈ [1,70] m，约束完整实体包络 |
| 设施、保护空域、导航点 | (0,0,0) m；半径 12 m、高度 0–22 m 的柱体；(0,0,12) m |
| Pursuer 部署 | 距设施水平 28–42 m、高 12–28 m、初速 0–1 m/s |
| Evader 部署 | 距导航点 100–130 m、高 16–32 m、初速 8–10 m/s |
| 接触与友机预警 | 等效半径各 0.5 m，默认接触 1.0 m 中心距；P-P 预警 3 m |

先将请求投影到速度球并限制垂直分量，得到可行速度 q。执行速度 u = v + α(q−v)，其中 α = min(1, 4τ / ‖qₕ−vₕ‖)。执行速度及实际 v(t) 都是合法速度集合中的凸组合。控制步内水平加速度为 (uₕ−vₕ) exp(−t/τ)/τ，最大值在步初。位置按同一常值 u 解析积分。

反应规避从导航方向及两个三维锥环共 17 个方向评分，目标用相同的执行约束预测，Pursuer 按当前速度外推。可用 JSON `{"evader_mode":"bezier"}` 启用独立参考模式，通过曲线切向速度和位置误差形成请求速度。

以连续解析轨迹确定首个事件。时间容差内按同侧接触、突破、越界或撤离、异方拦截排序。同类事件再按时间和稳定 ID 排序。异方接触使双方退出，终局成功要求所有 Evader 通过异方接触退出。同侧接触结束回合并独立记录。突破按目标中心，越界按实体包络，均独立于接触。三个概念分别为地面设施中心、柱状保护空域和相对设施配置的导航点。

团队奖励为每次异方拦截 +5、全拦截 +10、突破/撤离/同侧接触/Pursuer 越界 −10，另保留距离进度及每步 −0.01 时间成本。失败结算扣除剩余时间预算，提前失败不会节省时间成本。失活机保留失活当步的团队奖励，后续步的观测和奖励为零。GAE 使用真实接触后的掩码切断该机的 bootstrap。

## 预设三维参考轨迹

独立协议 `uav-preset-trajectory-v1.0.0` 提供 `straight` 与 `bezier` 两种参考轨迹。起点、实际初速度、有效掩码和静态终点按种子配对。Bézier 各机控制点使用独立的 `[seed,1,slot]` 随机流，初态使用 `[seed,0]`。控制点和飞行包络以 JSON 存档。旧拦截任务的 `goal_seeking` 按当前实际位置指向目标，旧 `bezier` 保留其固定时长，两者继续使用原协议。

几何曲线使用直线插值或三次 Bézier，统一采用 `u = 10r³ − 15r⁴ + 6r⁵`、`r = t/T` 的时间律。端点位置、速度和加速度连续，终点之后保持静止参考。弧长采用解析直线长度或 Gauss–Legendre 积分。时长以弧长除参考速度为下限，再用导数控制多边形上界约束参考速度、水平加速度和理想指令 `v_ref + τ a_ref`。参考速度为上限，实际匀速巡航并非时间律的假设。整个参考曲线由合法控制点的凸包保证在空域内。

实际跟踪使用参考前馈和位置、速度误差反馈，经现有速度与加速度限制后调用解析积分。飞行边界沿实际连续轨迹检测。参考完成、实际末端误差和稳定到点分别报告。实际航程用控制时刻间的三维折线估计，逐步加速度记录对应该步开始，末帧没有新指令。

```bash
conda activate uav
root=outputs/preset_trajectory_new
python -m tools.preset_trajectory_audit --out "$root" --episodes 50 --seed 1200 --limits 4 7 12
# 单独生成 B 模式，也可通过 --config 加载独立 JSON 参数
python -m tools.preset_trajectory_audit --out outputs/preset_bezier_new --mode bezier --limits 12 --episodes 1
```

### 预设轨迹 MAPPO

协议 `uav-preset-tracking-mappo-v1.0.0` 单独训练 `straight` 和 `bezier`。默认两架非接触式 UAV，最多五槽，无移动目标及接触事件。Actor 和 Critic 均为 91 维。每槽含位置、速度、相对参考位置、参考速度、参考加速度、参考剩余时间、成员截止倒计时和有效标志。Actor 自身槽在首位，Critic 按实体 ID 排列。末维为全局预算剩余比例。位置除以 50 m，速度除以物理速度上限，加速度除以水平加速度上限，两类时间均除以回合最大时长。失活或空槽为零。

位置和速度误差按活跃时间及初始成员数聚合。每完整控制步的位置误差系数为 `1 / 10 m`，速度误差系数为 `0.25 / vmax`，时间代价为 0.002。到期达到位置和速度容差的成员贡献 `+5/N`，精度失败成员贡献 `−2/N`，越界整局扣 10。以上参数可用 `--env` 配置。成员在参考时长加保持期截止时退出，越界真终止，预算到达而仍活动则截断。数值异常显式停止训练并保留最新有效提交。此协议不启用方向势函数。

```bash
conda activate uav
common=(--total-steps 1048576 --device cuda:0 --seed 17 --workers 4 --horizon 128
        --epochs 4 --minibatch 256 --gamma 0.99 --gae-lambda 0.95
        --actor-lr 0.0003 --critic-lr 0.0005 --eval-interval 16
        --eval-seed 2200 --eval-episodes 10)
python -m algorithm.trajectory_tracking train --mode straight "${common[@]}" --out outputs/tracking_straight
python -m algorithm.trajectory_tracking train --mode bezier "${common[@]}" --out outputs/tracking_bezier
python -m tools.trajectory_tracking_report --straight outputs/tracking_straight \
  --bezier outputs/tracking_bezier --out outputs/tracking_comparison
```

`resume --out 运行目录 --mode 原模式` 恢复已保存预算。训练和恢复可加 `--until-steps` 在预算内暂停，步数必须对齐 `workers × horizon`。`evaluate latest.pt --out 新目录` 使用固定种子的确定性策略。评估逐局报告参考末点误差、保持末端误差、位置误差分位数、实际航程和限幅次数。时间归一化误差仅在参考运动阶段按实体秒数加权，路径归一化再对各成员的误差除以其参考弧长。

## 静态航路方向奖励消融

独立协议 `uav-static-navigation-v1.0.0` 使用可见的静态航路点、原解析动力学和 52/54 维网络。独立导航基础奖励为距离进度、到达、越界与时间代价。无移动目标及接触判定。配置字段 `heading_enabled=false`、`heading_weight=0.3`、`heading_velocity_reference=1.0`、`heading_distance_reference=5.0`、`heading_discount=0.99`。启用时检查折扣与训练 gamma 严格相同。拦截任务保持原协议和奖励。

方向质量为实际速度与航路点方向的余弦，乘以速度和距离门控。团队势为有效槽质量之和乘以权重，再除以回合初始成员数。每步只加入一次 `gamma * next_potential - potential`。真正终止的后继势为零，超时保留真实后继势与 GAE bootstrap。方向未定义时质量为零，角度统计缺失。折扣小于 1 时，固定负势会有一个正的折扣差分，整段折扣回报仍由端点决定。多机提前退出会切断该机的 GAE，团队恒等式不能自动推广为每个提前退出个体的策略不变性保证。

```bash
conda activate uav
root=outputs/navigation_heading_new
python -m algorithm.navigation --out "$root/ablation" --updates 12 --eval-interval 4 \
  --seeds 17 18 --evaluation-seed 2200 --evaluation-episodes 12
python -m tools.navigation_heading_audit --study "$root/ablation" --out "$root/report"
```

训练逐回合日志和独立评估逐局记录保存方向角、改善率、横向误差、速度、速度变化率、转向率和越界次数。最终评估每个模型的首局另存逐步势函数与原始方向诊断。默认短程对照包含关闭、0.1、0.3、1.0 四组。
可信的历史导航 Checkpoint 可通过 `algorithm.navigation.read_checkpoint` 加载，该入口识别原 CLI 存档中的旧统计类路径。

## 训练与历史档案

Actor 和 Critic 使用 128–128 Tanh，Actor 输出三组 Beta 参数。PPO 只用有效 Pursuer 的样本，真实终止不 bootstrap，超时使用自动重置前的真实末状态 bootstrap，两类边界都切断跨回合的 GAE。训练默认 4 个 spawn worker、每次采样 128 步、4 轮 PPO、minibatch 256、seed 17。Actor/Critic 学习率为 3e−4/5e−4，γ=0.99、λ=0.95、clip=0.2、entropy=0.01、梯度范数上限 0.5。训练与评估环境使用独立 SeedSequence 命名空间。

新 Checkpoint 包含版本、维度、环境和训练配置、模型、优化器、RNG、采样 worker 及进度。加载前检查专用协议标头。请只加载可信本地存档。历史协议的模型、资料与实验输出见 `archive/README.md`。历史执行代码以 Git 提交定位。

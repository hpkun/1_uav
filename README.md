# Xu (2026) APH-MAPPO 三维空战与历史巡检基准

新增受控奖励消融 `posture_approach_gated_3v3`：只在Reward结算时门控2000米外的正Posture，默认`main_3v3`仍为原始`paper`模式，APH原始姿态优势不变。公式、严格隔离、466项测试及CUDA实测见 [Approach-Gated Posture报告](docs/approach_gated_posture.md)。该实验不进入论文37项计划或论文表格，本轮未运行正式训练。

当前空战默认协议已升级为 **xu-aircombat-v2.0.0**：连续标架积分处理垂直方向，真正仿真故障显式中断采样。极区延续是论文未公开的工程假设；旧v1 checkpoint不能恢复v2训练。数学说明见 [动力学v2](docs/dynamics_v2.md)，408项测试和CUDA实测见 [修复验收报告](docs/dynamics_v2_validation.md)。新正式实验从头写入 `outputs/xu_iv_v2`，不混用旧统计。

运行遵循 `AGENTS.md`：优先WSL Ubuntu的 `uav` Conda环境，CUDA训练/评估必须可用；下文PowerShell命令保留为WSL不可用时的替代示例。

当前空战入口为 `python -m algorithm.aircombat`，独立环境位于 `env/aircombat/`：支持 **3v3、5v5、3v5**，Red 共享 Beta Actor 学习，Blue 使用论文 Table I 的 15 动作、aggregate-sum 80/20 规则。Actor/Critic 为 **12D/60D**，与历史巡检 12D/90D 存档互斥。

默认环境参数、训练表 III 参数、论文外假设、逐文件变更及验证结果见 [前三阶段实施报告](docs/xu2026_aircombat_implementation.md)。`shootdown_r0=10` 是**论文未报告的实验占位值**，可以通过 `--r0` 修改。未进行正式 5000 回合 × 5 种子实验，也未声称论文性能复现。

在 PowerShell 的 `uav` 环境运行（CUDA 必须可用）：

```powershell
$env:PYTHONIOENCODING='utf-8'
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate uav
python -m pytest -q
python -m tools.validate_aircombat --output outputs/my_aircombat_acceptance

# 以下正式预算命令仅为使用示例，本次没有执行。
python -m algorithm.aircombat train --scale 3v3 --output outputs/aircombat_3v3
python -m algorithm.aircombat train --scale 5v5 --output outputs/aircombat_5v5
python -m algorithm.aircombat train --scale 3v5 --output outputs/aircombat_3v5
python -m algorithm.aircombat evaluate outputs/aircombat_3v3/latest.pt --episodes 200
```

`train.jsonl` 记录完成回合数、联合环境步数、PPO 更新数、逐回合 return、五项奖励、Blue 击落数、Red 存活数、全消灭胜率及三类结束原因。第四节实验入口已支持**精确完成回合目标 checkpoint**，包括4950；中途存档保留半个rollout和预采样动作，恢复不提前触发PPO更新。独立评估保存在单独的 `final_evaluation.json`。

第四节全部主实验、消融、单因素和Blue敏感性配置及正式命令见 [第四节实验报告](docs/xu2026_section_iv_experiments.md)。`python -m algorithm.experiments plan` 仅导出37个独立训练任务，**不启动训练**。静态图表入口为 `python -m tools.report_aircombat`，典型场景轨迹入口为 `python -m tools.aircombat_trajectories`。

训练终端默认每5次PPO更新显示摘要，并在首次更新、50回合节点和训练结束时显示；checkpoint及最终独立评估另行显示。`--log-every-updates N`调整频率，`--quiet`关闭终端展示，完整JSONL/回合/PPO文件照常保存。最近100个已完成回合的统计为空时显示N/A。详见 [入口与日志说明](docs/aircombat_training_console.md)。

`algorithm.experiments train --episodes 2500` 只改变完成回合目标，标记为半预算诊断；默认输出根目录为 `outputs/xu_iv_2500`。默认5000回合仍输出到 `outputs/xu_iv`。可以通过`--root`显式指定不同目录；2500与5000保持独立训练，不能跨预算恢复。Ubuntu示例（本次未执行正式预算）：

```bash
conda activate uav
python -m algorithm.experiments train --experiment main_3v3 --seed 1 --episodes 2500 --root outputs/xu_iv_2500
python -m algorithm.experiments train --experiment main_3v3 --seed 1 --episodes 5000 --root outputs/xu_iv_5000
```

下面保留历史巡检任务的说明与入口；它们不属于新的论文空战训练路径。

## 历史巡检任务与早期数组协议

非武器化三维多UAV航路点巡检：3U3P、5U5P、3U5P。环境 `coop3d-v3.0.0`，巡检算法 `mappo-coop3d-v3.0.0`。环境在 `env/`，MAPPO 在 `algorithm/`，导航诊断及录像在 `tools/`。旧 `coop3d/` 包已移除，没有兼容转发入口。

本轮完成环境语义回归、90D集中状态、五个体价值Critic、Beta配平初始化、NaN安全GAE、严格采样预算、导航可达性审计和短程CUDA Smoke Test；没有运行正式长训练。

早期整改记录：巡检环境移除±85°限制并复用原式RHS/RK4；动力学及γ特征改变，**旧v2 checkpoint明确拒绝直接恢复**，历史数据未改写。早期独立数组协议的结果见 [历史整改报告](docs/xu2026_rectification.md)。本次已将 `ProtocolMAPPOTrainer` 连接真实空战；当前结论以上面的空战实施报告为准。

## 当前机器：PowerShell / uav

```powershell
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate uav
python -m pytest -q
python -m compileall -q env algorithm tools tests
python -m tools.audit_navigation --output outputs/my_navigation_audit
python -m tools.validate_coop3d --output outputs/my_v2_smoke
```

审计工具默认每规模每策略5个随机种子，加1个固定场景；不启动学习。validate工具只运行2环境、8步horizon、1 PPO epoch：总预算先17，再从断点继续至25个joint transitions。已有输出目录不会被新运行覆盖。

## Ubuntu 正式训练示例（本轮未执行）

在用户准备好的Ubuntu项目目录中：

```bash
conda activate uav
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name())'

python -m algorithm.train_mappo --env-config configs/coop3d_3U3P.yaml --algorithm-config configs/mappo_coop3d.yaml --device cuda --seed 1000 --num-envs 10 --total-sampled-steps 1000000 --output-dir outputs/mappo_3U3P_seed1000
python -m algorithm.train_mappo --env-config configs/coop3d_5U5P.yaml --algorithm-config configs/mappo_coop3d.yaml --device cuda --seed 1000 --num-envs 10 --total-sampled-steps 1000000 --output-dir outputs/mappo_5U5P_seed1000
python -m algorithm.train_mappo --env-config configs/coop3d_3U5P.yaml --algorithm-config configs/mappo_coop3d.yaml --device cuda --seed 1000 --num-envs 10 --total-sampled-steps 1000000 --output-dir outputs/mappo_3U5P_seed1000
```

以上只是用户自行执行的命令示例，不代表这个预算已经足以收敛。`sampled_steps`严格指全部环境的joint transitions之和。最后一次rollout和vector step可缩短，预算不会因batch对齐超出。

恢复时默认使用断点原配置；总预算可以提高，不能悄悄改变任务规则或优化参数：

```bash
python -m algorithm.train_mappo --device cuda --total-sampled-steps 2000000 --output-dir outputs/mappo_3U3P_seed1000 --resume outputs/mappo_3U3P_seed1000/latest.pt
```

## 独立评估与指定模型录像

```bash
python -m algorithm.evaluate_mappo --checkpoint outputs/mappo_3U3P_seed1000/best.pt --device cuda --seed-base 20000000 --episodes 5 --output outputs/eval_3U3P_best
python -m algorithm.evaluate_mappo --checkpoint outputs/mappo_3U3P_seed1000/latest.pt --device cuda --seed-base 20000000 --episodes 5 --output outputs/eval_3U3P_latest
python -m tools.record_episode --checkpoint outputs/mappo_3U3P_seed1000/best.pt --device cuda --seed 20000010 --output outputs/replay_3U3P_best
python -m tools.record_episode --policy navigation --env-config configs/coop3d_3U5P.yaml --seed 101 --output outputs/replay_navigation
python -m tools.render_replay --trace outputs/replay_navigation/trajectory.json --output outputs/replay_navigation/replay.html
```

训练、模型评估与checkpoint审计强制CUDA，缺少CUDA明确失败。旧60D checkpoint不能恢复为v2。模型保存包括两个网络/Adam、完整配置与维度协议、sampled_steps、所有RNG、worker状态及episode统计。只加载可信本地checkpoint。best与latest的评估分别记录，录像元数据标明实际模型来源。

## 环境接口

```python
import numpy as np
from env.config import EnvConfig
from env.factory import make_environment

env = make_environment(EnvConfig(scale='3U5P'))
observations, info = env.reset(seed=7)
actions = np.tile([.5, 1/3, 1/8], (3, 1))
observations, rewards, terminated, truncated, info = env.step(actions)
assert observations.shape == (3, 12)
assert info['centralized_state'].shape == (90,)
```

Actor只使用12D局部观测；Critic使用90D全局状态并输出5个UAV价值槽位，3UAV时只用前3项。后30维显式编码全部航路点，包含未分配的点。

保留五项巡检奖励、Hungarian独占分配、0.25s/RK4、100–400m/s、250步、300m完成半径及100m安全参考。动力学无γ裁剪；分母奇点/阶段跨越失败明确记录并失效，末态速度裁剪是工程选择。巡检γ特征改为γ/π，并非作者归一化。无任务配平、故障当步掩码和终止语义见 [环境规范](docs/environment_spec.md)。

默认Actor使用Beta `trim_centered` 初始化，动作均值接近 `[.5,1/3,1/8]`，浓度16，仍保留探索。优化参数与mask/GAE/预算/日志协议见 [MAPPO规范](docs/mappo_spec.md)。

## 诊断结果与回放

实际小样本审计：导航在3U3P、5U5P的固定场景和各5个随机种子上成功；3U5P随机成功1/5，固定场景完成3/5后超时。额外手工可达多任务场景79步完成5/5。它证明控制接口可用，不能据此声称所有随机场景都可完成。[完整导航审计](docs/navigation_audit.md)包含失败复核和两种Actor初始化的实际对照。

回放由真实JSON记录生成，HTML内嵌Plotly，可离线查看。支持播放/暂停、滑块、三维旋转、轨迹、速度方向、点完成与分配、控制、失效、限幅事件、五项奖励和逐UAV查询。渲染不执行环境。当前仍未完成浏览器内视觉交互验收，不能将脚本语法检查视为浏览器验证。

详细文件清单、测试和CUDA结果见 [重构报告](docs/refactor_report.md)。已有outputs、历史模型和原始PDF均保留。v1恢复副本在 `tmp/coop3d_v1_before_v2.zip`，迁移前参考轨迹在 `tests/fixtures/coop3d_v1_semantics.json.gz`。

## 训练日志维护与离线报告

训练终端默认输出紧凑的 `[TRAIN]`、独立 `[EVAL]` 和最终 best/latest 摘要，保留原有 JSONL/CSV。新记录同时提供 `policy_loss` 和实际用于 backward 的 `actor_total_loss`；兼容字段 `actor_loss` 现在是后者的别名。**修复前 v2 历史记录中的 actor_loss 实际是 policy_loss**，不能将新旧同名字段直接当作相同损失比较；历史实验文件不会被重写。

Ubuntu 离线查看已有训练结果（将 run-dir 替换为真实实验目录）：

```bash
conda activate uav
python -m tools.report_training --run-dir outputs/你的训练目录 --last-updates 10
```

也可用本仓库现有历史记录检查工具：

```bash
python -m tools.report_training --run-dir outputs/coop3d_v2/smoke --last-updates 10
```

该工具只读 `training_metrics.jsonl`、`evaluation_history.csv`、`run_summary.json`、`run_config.json`，不加载模型、不启动仿真，也不需要 CUDA 或 PyTorch。缺失字段显示 N/A，旧 actor_total_loss 不补造。诊断提示只供排查，不自动判定训练失败或收敛。

恢复训练前会只读检查日志、断点和 best/latest 来源。日志重复、比断点更新或存在更晚模型时明确拒绝追加，保留原文件；日志缺失或落后时提示历史缺口，不伪造数据。默认不自动裁剪、删除或覆盖冲突实验。网络、优化、版本、Checkpoint 协议和 TrainConfig 持久化字段均保持不变。

浮点轨迹回归采用 `rtol=1e-12, atol=1e-12`；离散状态与事件依旧精确比较，原始参考 fixture 不变。实际修复、测试与日志示例见 [日志维护报告](docs/training_log_maintenance.md)。

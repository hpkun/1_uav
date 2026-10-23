# 空战训练入口与终端摘要

后续动力学修复已升级至v2，当前默认根目录为 `outputs/xu_iv_v2_2500` / `outputs/xu_iv_v2`，且真正数值故障独立fail-fast。以下记录日志优化阶段的历史设计；当前运行与兼容性以 [动力学v2](dynamics_v2.md) 为准。

本轮只修改 `algorithm/aircombat.py`、`algorithm/experiments.py`，新增纯终端模块 `algorithm/common/aircombat_progress.py` 和针对性测试 `tests/test_aircombat_console.py`，更新README及本文档。环境、奖励、观测、APH、Blue、MAPPO、并行器、rollout收集器和checkpoint序列化实现均未改动。

## 入口与预算

`python -m algorithm.experiments train`支持`--episodes 2500`和`--episodes 5000`，省略仍是5000。只替换`ProtocolTrainConfig.training_episodes`；正式10环境、每环境horizon1500、单回合250步、PPO参数、种子、50回合checkpoint目标和最终200局独立评估均保持原值。

2500在experiment.json和final_evaluation.json中明确记录`training_budget_kind=half_budget_diagnostic`、`training_episode_target=2500`；训练config.json和checkpoint原有train_config保存实际2500目标。5000标记为paper_5000。统计工具原有“完成5000回合且独立评估200局”检查保留，半预算不会被标成formal_complete。

默认输出根目录分别是`outputs/xu_iv_2500`、`outputs/xu_iv`，也可分别传`--root`。同一目录已有训练记录时拒绝覆盖，不能用5000预算恢复2500checkpoint；原有完整配置校验和半rollout恢复语义保留。`--episodes`不能与`--smoke`组合，以免混淆诊断预算和小规模验收。

Ubuntu启动命令（使用CUDA的uav环境）：

```bash
conda activate uav
python -m algorithm.experiments train --experiment main_3v3 --seed 1 --episodes 2500 --root outputs/xu_iv_2500
python -m algorithm.experiments train --experiment main_3v3 --seed 1 --episodes 5000 --root outputs/xu_iv_5000
```

## 终端日志

- `[START]`：实际spawn worker数量、GPU设备和名称、种子、目标预算、horizon、max_steps、输出目录和恢复来源。
- `[TRAIN]`：首次更新、每5次更新、跨50回合节点和训练结束时打印。可用`--log-every-updates N`调整，`--quiet`关闭显示；这两个选择不写入PPO配置、不参与恢复兼容性。
- 四行摘要依次显示预算进度/联合步/PPO次数/实际耗时/吞吐、最近100个已完成回合的Return/训练Win Rate/存活/击落、相同窗口的三类结束计数与占比、真实PPO损失/熵/KL/裁剪比例/有效样本数。没有完成回合时显示N/A；后续空rollout继续使用已标明大小的历史窗口，不假装当前rollout产生了回合。
- 吞吐量为本次进程启动以来新增的联合环境步数除以实际墙钟耗时，包含启动、PPO、checkpoint和I/O，标记为session。恢复时不把过去的采样步计入本次吞吐量。本轮不显示ETA，避免将估计值当实际耗时。
- `[CKPT]`：精确目标及latest保存后打印实际完成回合、当时PPO更新数和文件名；半rollout存档的更新数可能小于最终latest，沿用既有语义。
- `[EVAL]`：最终独立评估开始时显示实际局数和模型绝对路径，结束时显示独立胜率、平均Red存活、Blue真实击落、三类结束计数/比例及模型来源。正式默认200局，小规模smoke仍明确使用2局。
- `[DONE]`：训练和实验完成提示；训练结束汇总显示已记录回合的累计结束计数，不把numerical_failure当成击落。

`train.jsonl`每次更新及其完整episodes数组、PPO指标、五项奖励、checkpoints.jsonl及最终评估统计仍完整写盘。纯展示模块没有NumPy/Torch/环境调用，不读写随机状态。对历史日志的读取只用于恢复后的显示窗口。

## 验证

针对性测试包含：两种预算只改一个字段、默认5000、CLI传参、元数据与目录隔离、严格拒绝跨预算恢复、真实指标格式、空窗口N/A、数值失败独立统计、最终200局参数传递、半预算不被标成正式完成、10个真实spawn worker及两种checkpoint预算。

CUDA日志无反馈测试分别运行关闭日志、每次更新输出、每5次更新输出，零容差比较完整checkpoint（Actor、Critic、优化器、worker快照、采样计数等）、Python/NumPy/Torch CPU/CUDA随机状态及train.jsonl原始字节，要求完全一致。原有并行和半rollout测试继续保留。

全量测试、编译检查和本次独立CLI smoke的最终实测结果见下方补充；未启动2500或5000回合训练。

## 本次实测结果

PowerShell/uav，解释器 `D:\conda_envs\envs_dirs\uav\python.exe`，GPU为NVIDIA GeForce RTX5060 Laptop GPU。

- 针对性测试11项通过；全量测试 **312 passed in 93.34s**，失败0、跳过0。记录为 `outputs/aircombat_console_targeted.xml` 和 `outputs/aircombat_console_full_tests.xml`。
- Python编译检查 `python -m compileall -q env algorithm tools tests` 退出码0。
- 实际启动10个不同PID的spawn worker并确认全部存活；分别保存并读取目标2500/5000的checkpoint配置，没有进行这两个预算的训练。
- CLI smoke使用既有`--smoke`配置：2个worker、horizon8、每次1个PPO epoch、2个完成回合、最终独立评估2局；环境max_steps仍为250。实际完成500个联合环境步、32次PPO更新，train.jsonl保留32条完整更新记录。正式10/1500/200参数没有被smoke配置替换。
- 两个训练回合均timeout，无训练数值失败；最终两个独立评估回合均numerical_failure，真实Blue击落为0。该现象保持原样，没有改环境、策略或算法。
- 实际终端记录：`outputs/aircombat_console_smoke.log`；完整数据和模型：`outputs/aircombat_console_smoke/main_3v3/seed_1/`。

真实输出摘录（计时/吞吐仅代表本次机器运行）：

```text
[START] scale=3v3 | seed=1 | Episodes 0/2 | Workers 2 (spawn) | GPU cuda | horizon=8 | max_steps=250 | log_every_updates=5 | output=outputs\aircombat_console_smoke\main_3v3\seed_1
[START] GPU name: NVIDIA GeForce RTX 5060 Laptop GPU
[TRAIN] Episodes 0/2 (0.0%) | Joint steps 16 | PPO updates 1 | Elapsed 00:00:04 | Throughput 4.0 joint steps/s (session)
        Recent n=0/100 | Return N/A | Train Win Rate N/A | Red alive N/A | Blue shot down N/A
        Ends(recent): Normal 0 (N/A) | Timeout 0 (N/A) | Numerical Failure 0 (N/A)
        PPO: Actor Total Loss 0.00070 | Policy Loss 0.00000 | Critic Loss 2.94278 | Entropy -0.69906 | Approx KL 0.000000 | Clip Fraction 0.00000 | Valid Samples 48
[CKPT] Episodes 2/2 | PPO updates 32 | File latest.pt
[DONE] training Episodes 2/2 | Joint steps 500 | PPO updates 32 | Elapsed 00:00:26 | Ends(recorded): Normal 0 (0.0%) | Timeout 2 (100.0%) | Numerical Failure 0 (0.0%)
```

最终EVAL行实际显示`independent n=2`、`Win Rate 0.0%`、`Red alive 3.00`、`Blue shot down 0.00`、`Timeout 0 (0.0%)`、`Numerical Failure 2 (100.0%)`，并附latest.pt的绝对来源路径。正式运行时该局数由未改变的配置取200，不使用训练回合胜率代替独立评估。

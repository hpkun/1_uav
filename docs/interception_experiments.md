# 三维拦截实验运行

在仓库目录激活 `uav` Conda 环境。训练、续训和模型评估要求 CUDA。以下命令在 Ubuntu 前台运行。

```bash
conda activate uav
python -m algorithm.interception train --config configs/experiment_s2_1000000.json --seed 17 --out outputs/s2_seed17
python -m algorithm.interception resume --out outputs/s2_seed17
python -m algorithm.interception resume --out outputs/s2_seed17 --extend-total-steps 2000000
python -m algorithm.interception evaluate outputs/s2_seed17/best.pt --scenario random --seed 10020000 --episodes 200 --output outputs/s2_test.json
python -m algorithm.interception stats --out outputs/s2_seed17 --json
python -m algorithm.interception report outputs/s2_seed17 --out outputs/s2_report
```

新运行、批次、独立评估和报告使用新路径。配置样例包含 [S2 常规 100 万步](../configs/experiment_s2_1000000.json)、[S2 直线 200 万步](../configs/experiment_s2_straight_2000000.json) 和 [S2 Bézier 200 万步](../configs/experiment_s2_preset_bezier_2000000.json)。命令行可显式指定 `--total-steps 1000000` 或 `--total-steps 2000000` 覆盖样例预算。

## 配置字段

| 字段 | 生效方式 |
|---|---|
| `version`, `name`, `algorithm` | 版本 1，实验名称，算法标识 `uav-interception-mappo-v4` |
| `environment` | `InterceptionConfig` 字段。任务 S0–S3，`evader_mode` 可使用已有规则、`straight` 或 `preset_bezier`。省略时沿用任务默认规则 |
| `training` | `TrainConfig` 字段，包括 `seed`、`workers`、`horizon`、优化器和 PPO 参数 |
| `total_joint_steps`, `log_interval` | 请求的联合环境步数，每隔多少个累计 Update 打印进度 |
| `evaluation` | 原有 `evaluation_episodes`、`evaluation_seed_base`、`evaluation_scenario`、`evaluation_deterministic`。间隔二选一，`evaluation_interval_updates` 或 `interval_joint_steps` |
| `checkpoint` | `latest_best`。只保留当前可恢复和当前最优验证模型 |
| `test` | `enabled`、`scenario`、`seed`、`episodes`、`deterministic`、`checkpoint`。默认训练结束用 latest、随机场景、起始种子 10010000、20 局、确定性动作。样例选择 Best 和 200 局 |
| `storage` | 自动最终测试的 `trajectories` 为 `none`、`selected` 或 `all`。选中模式必须给出明确的 `trajectory_seeds`。`gzip` 默认开启 |
| `report` | `on_complete` 默认关闭。`plots` 决定是否生成 PNG，显式报告也可用 `--no-plots` |

解析顺序为现有默认值、统一配置、显式 `--env` 和 `--training` 文件、显式 CLI 字段。最终生效的环境、训练、验证和实验设置写入 `config.json`，并进入现有 Checkpoint 运行清单。未知字段、无效任务组合、冲突预算及验证与测试种子重叠会报错。选择 Best 做最终测试时，原预算内必须安排至少一次定期验证。

## 步数、恢复和验证

每次 PPO 更新采样 `workers × horizon` 个联合步。请求步数只接受十进制正整数，预算向上对齐到完整 Update。默认 4 个环境、每次 128 步，1000000 步对齐到 1000448 步和 1954 次更新，2000000 步对齐到 2000384 步和 3907 次更新。按步数设置的验证间隔也向上对齐，例如请求 513 步时每 1024 步验证一次。

`resume --out RUN` 从最后提交恢复至已保存的绝对预算。完成后重复执行不会追加更新。可以在 `train` 或 `resume` 上用 `--until-steps` 在对齐的步边界主动暂停。`--extend-total-steps` 指定新的绝对预算，须至少增加一个 Update，每次变更保存为原子的 `budget_*.json`。扩容后的最终测试使用新编号。

原有 `train --updates N` 和旧目录上的 `resume --updates N` 仍支持。旧目录省略 `--updates` 会完成上次提交所指向的预算。统一运行扩容使用 `--extend-total-steps`。同时传入 `--updates` 和总步数时，两者必须表示同一个对齐预算。

终端用 START、RESUME、TRAIN、EVAL、DONE、PAUSE 和 ERROR 标识阶段。TRAIN 显示累计步数、Update、完整回合、最近 100 局指标和 PPO 诊断。周期 EVAL 使用固定的验证种子范围并沿用原 Best 评分。最终测试使用独立种子，报告明确标为最终测试。训练、评估场景和评估动作的随机流沿用各自的种子命名空间。`--debug` 显示完整异常栈。

## 文件和回放

- `config.json` 是生效配置和运行清单。扩容生成 `budget_*.json`。
- `latest.pt`、`best.pt`、`best.json` 是当前恢复点和验证最优模型。Best 仅在启用且已执行验证后出现。两者指向同一代时，统一运行在支持硬链接的文件系统共享数据。下一次原子更新替换 latest，Best 继续指向原来的代。文件系统不支持链接时采用原有独立副本。
- `update_*.json` 是逐 Update 的链式提交。`training_metrics.jsonl`、`episode_metrics.jsonl` 和 `evaluation_history.csv` 是派生视图。`evaluation_*.json` 保存定期验证的逐局和汇总指标。
- `test_*.json` 保存最终独立测试的模型标识、场景、种子、逐局与汇总指标。启用轨迹时改为同编号目录下的 `summary.json` 和所选轨迹。
- `failure.json` 保留最近一次训练更新故障。报告按需写入单独目录，默认训练过程不生成图表或视频。

恢复依赖 `latest.pt` 内的末条更新及逐 Update 文件的摘要链。报告也读取同一提交记录。本版继续使用该格式和末条补写协议。逐 Update 小文件会随训练增长，历史记录压缩迁移留待独立验证。

`stats` 分别报告文件名数量、逻辑字节数和按文件标识去重后的字节数。去重值不包含文件系统分配块与目录开销。

`evaluate --output RESULT.json` 默认只生成一个 JSON，评估环境取自 Checkpoint。需要轨迹时加 `--trajectories selected --trajectory-seed SEED --gzip`，或显式使用 `--trajectories all`。轨迹写入 `RESULT_trajectories`，可用原有 `python -m tools.interception_replay TRACE.json.gz REPLAY.html` 导出。旧 `evaluate --out DIR` 继续默认生成全部逐帧轨迹。`--out` 和 `--output` 互斥，`--json` 仍控制原有的 stdout 机器可读摘要。

## 多种子

```bash
python -m algorithm.interception batch --config configs/experiment_s2_1000000.json --seeds 17 18 19 --out outputs/s2_batch
python -m algorithm.interception batch --config configs/experiment_s2_1000000.json --seeds 17 18 19 --out outputs/s2_batch --resume
python -m algorithm.interception report --batch outputs/s2_batch --out outputs/s2_batch_report --no-plots
```

批次顺序训练，每个种子使用 `seed_<seed>` 和独立 Checkpoint。`batch.json` 保存共同协议和种子列表。恢复批次使用原预算、相同的生效配置及种子列表，续接已有有效 Checkpoint 并启动尚未创建的成员。已有目录如果缺失初始 Checkpoint，会明确报错并保留现场。报告检查任务、环境、算法、训练预算和评估设置，按 Update 汇总均值、样本标准差和实际样本数。最终测试单独汇总。配置开启 `report.on_complete` 后，批次完成时可自动调用同一报告实现。

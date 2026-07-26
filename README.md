# UAV Env

面向可复现实验的同构无人机近距空战环境，支持完整 1v1、固定同构 2v2，以及可训练的前馈 CTDE MAPPO 基线。公共内核包括三自由度过载质点动力学、15 种离散动作、同步 RK4、同步概率伤害、固定实体槽位、观测/全局状态诊断和规则对手。

当前 MAPPO 是项目标准基线，不是 2024 论文图网络的逐层复刻；不包含 GRU、注意力、异构平台、3v2、网络迁移或零样本迁移。

## 安装与测试

建议在 WSL 的 `uav` Conda 环境运行：

```bash
conda run -n uav pip install -e .
conda run -n uav pytest
conda run -n uav python scripts/smoke_test.py
```

## 环境与公平性评估

```bash
python scripts/run_1v1_episode.py --scenario head_on --opponent pursuit --red-policy pursuit --seed 1
python scripts/run_2v2_episode.py --scenario head_on_formation --opponent pursuit --red-policy pursuit --seed 1
python scripts/evaluate_symmetry.py --episodes 100
```

2v2 的红蓝规则侧使用同一目标分配语义；随机规则拥有独立 RNG。对称性脚本使用镜像加阵营交换的配对种子，结果保存到 `outputs/metrics/symmetry_report.csv`。

## MAPPO

```bash
python scripts/train_mappo.py --config configs/mappo_smoke_1v1.yaml --run-name smoke_1v1
python scripts/train_mappo.py --config configs/mappo_smoke_2v2.yaml --run-name smoke_2v2
python scripts/train_mappo.py --config configs/mappo_1v1_head_on.yaml --total-env-steps 50000 --run-name probe_1v1_head_on
python scripts/evaluate_mappo.py --checkpoint <best.pt> --episodes 200 --seed-start 200000 --deterministic
```

输出位于 `outputs/mappo/<experiment>/<run-id>/`，包含配置、CSV、TensorBoard、initial/last/best checkpoint 和最终摘要。详见 [MAPPO 设计](docs/mappo_design.md)、[训练协议](docs/mappo_training_protocol.md)及[可学习性检查表](docs/environment_learnability_checklist.md)。

`tail_chase` 只用于 smoke、basic learnability 和动作/奖励链路检查，不作为正式空战性能。正式 1v1 使用 `head_on`、`balanced_random`，正式 2v2 使用 `head_on_formation`、`offset_formation`、`balanced_random`。`PursuitOpponent` 是固定几何规则空战对手，不是研究目标。正式报告区分击毁胜利、非超时胜利和按存活数量得到的超时胜利。

默认训练归一化为 `symmetric_training`；`paper_linear` 仅用于复核论文线性变量形式。1v1/2v2 是项目的同构实验环境，不应被描述为论文正式 3v2 复现。

多机终局奖励已根据根目录 2024 原论文 PDF 的式（21）—（25）逐项核验，正式默认是 `paper_2024_exact`；旧项目比例公式保留为 `project_balanced` 供消融比较。权重数值仍是项目配置，不声称为论文公开参数。
## Fixed homogeneous 3v3 audit

The current formal environment target is fixed homogeneous 3v3: 45-dimensional local observations, 87-dimensional global state, shared feed-forward Actor, centralized Critic, and 4-process parallel execution in the smoke configuration. It does not include 1v1 transfer, heterogeneous 3v2, attention, GRU, self-play, or network migration.

Run the reproducible audit with:

```bash
conda run -n uav python scripts/audit_3v3_environment.py
```

The audit writes `outputs/audit/3v3_environment_audit.json` and `.csv`; the human report is `docs/3v3_environment_audit.md`. The audit documents the fixed-slot destruction reward lifecycle fix, the project-defined negative cap for damaged dense rewards, and unresolved observation/global-state aliasing risks. Timeout survivor-count wins and elimination wins must be reported separately. The audit is not a claim that the current 3v3 environment has passed learnability validation.

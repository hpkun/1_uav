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
python scripts/run_1v1_episode.py --scenario tail_chase --opponent straight --red-policy pursuit --seed 1
python scripts/run_2v2_episode.py --scenario head_on_formation --opponent pursuit --red-policy pursuit --seed 1
python scripts/evaluate_symmetry.py --episodes 100
```

2v2 的红蓝规则侧使用同一目标分配语义；随机规则拥有独立 RNG。对称性脚本使用镜像加阵营交换的配对种子，结果保存到 `outputs/metrics/symmetry_report.csv`。

## MAPPO

```bash
python scripts/train_mappo.py --config configs/mappo_smoke_1v1.yaml --run-name smoke_1v1
python scripts/train_mappo.py --config configs/mappo_smoke_2v2.yaml --run-name smoke_2v2
python scripts/train_mappo.py --config configs/mappo_1v1_tail_chase.yaml --total-env-steps 50000 --run-name probe_1v1_tail_chase
python scripts/evaluate_mappo.py --checkpoint <best.pt> --episodes 100 --seed-start 100000 --deterministic
```

输出位于 `outputs/mappo/<experiment>/<run-id>/`，包含配置、CSV、TensorBoard、initial/last/best checkpoint 和最终摘要。详见 [MAPPO 设计](docs/mappo_design.md)、[训练协议](docs/mappo_training_protocol.md)及[可学习性检查表](docs/environment_learnability_checklist.md)。

默认训练归一化为 `symmetric_training`；`paper_linear` 仅用于复核论文线性变量形式。1v1/2v2 是项目的同构实验环境，不应被描述为论文正式 3v2 复现。

注意：多机终局 `paper_2024_exact` 候选实现尚未完成原文式（22）—（25）核验，因此安全默认暂为 `project_balanced`。在提供可读取 PDF 并通过逐式手算测试前，不应将候选 profile 用于“论文精确复现”结论。

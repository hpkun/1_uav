# UAV Env

面向学术实验的轻量级同构无人机 1v1 近距空战环境。环境使用三自由度过载质点模型、15 种离散机动、RK4 积分、同步状态更新和同步概率伤害结算。红方是外部 Gymnasium 智能体，蓝方可选择直飞、均匀随机或透明的几何追踪策略。

当前版本能够运行完整回合：场景初始化后，每个决策动作保持 0.5 秒，并执行 5 个 0.1 秒物理子步；随后检查边界、攻击几何、概率伤害、事件、奖励和终止条件。实现了 tail chase、head on 和 balanced random 三个 1v1 场景，以及 11 维 Actor 观测和 10 维 Critic 状态。

## 安装与测试

需要 Python 3.10 或更高版本。

```bash
python -m pip install -e .
pytest
python scripts/smoke_test.py
```

## 运行完整回合

```bash
python scripts/run_1v1_episode.py --scenario tail_chase --opponent straight --seed 1
python scripts/run_1v1_episode.py --scenario balanced_random --opponent pursuit --seed 2
python scripts/evaluate_rule_opponents.py --episodes 100
python scripts/validate_damage_distribution.py
python scripts/visualize_trajectory.py --scenario tail_chase --seed 1
```

程序接口：

```python
from uav_env.envs import make_1v1_env

env = make_1v1_env(scenario="tail_chase", opponent="straight", seed=1)
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step(0)
trajectory = env.get_trajectory()
```

论文给定参数、公式推导和本项目为消除歧义采用的约定分别记录在 [experimental_assumptions.md](docs/experimental_assumptions.md)。当前未实现同构 2v2、异构平台、有限雷达探测、论文预测威胁对手和强化学习算法。

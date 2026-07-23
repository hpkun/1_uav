# UAV Env

面向可复现实验的同构无人机近距空战环境，支持完整 1v1 和固定规模 2v2 回合。公共内核包括三自由度过载质点动力学、15 种离散动作、同步 RK4 推进、同步概率伤害、结构化事件、论文校准奖励和两种显式归一化模式。

## 安装与测试

```bash
python -m pip install -e .
pytest
python scripts/smoke_test.py
```

## 1v1

```bash
python scripts/run_1v1_episode.py --scenario tail_chase --opponent straight --red-policy pursuit --seed 1
python scripts/evaluate_1v1_matrix.py --episodes 100
```

## 2v2

```bash
python scripts/run_2v2_episode.py --scenario head_on_formation --opponent straight --red-policy pursuit --seed 1
python scripts/run_2v2_episode.py --scenario balanced_random --opponent pursuit --red-policy pursuit --seed 2
python scripts/evaluate_2v2_rules.py --episodes 100
python scripts/visualize_2v2_trajectory.py
```

```python
from uav_env.envs import make_1v1_env, make_2v2_env

env = make_2v2_env(scenario="offset_formation", opponent="pursuit", seed=7)
observations, info = env.reset(seed=7)
observations, team_reward, terminated, truncated, info = env.step([0, 0])
```

默认训练归一化为 `symmetric_training`；`paper_linear` 仅用于复核论文线性变量形式。2v2 是将 2024 年多机实体块定义适配到同构 2v2 的项目实验，不是论文正式的 3v2 场景。当前不包含 PPO/MAPPO、异构平台、雷达探测或未公开的预测威胁子函数。

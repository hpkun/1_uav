# Coop3D 三维多无人机协同巡检基准

唯一环境为 `coop3d-v1.0.0` / `coop3d.Cooperative3DEnv`。3U3P、5U5P、3U5P 仅通过场景配置改变数量。无人机协作访问静态三维航路点；算法仅保留标准 MAPPO。

## PowerShell 运行

所有项目命令使用 Windows `uav` Conda 环境。训练、模型审计和评估强制要求 CUDA。环境及网络数学测试可在 CPU 上运行。

```powershell
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate uav
python -m pytest -q
python -m coop3d train --env-config configs/coop3d_3U3P.yaml --train-config configs/coop3d_mappo.yaml --output outputs/my_3U3P --updates 100
python -m coop3d train --env-config configs/coop3d_3U3P.yaml --train-config configs/coop3d_mappo.yaml --output outputs/my_3U3P --updates 10 --resume outputs/my_3U3P/latest.pt
python -m coop3d evaluate --checkpoint outputs/my_3U3P/best.pt --output outputs/my_3U3P_holdout --seeds 20000000 20000001
python -m coop3d replay --trace outputs/my_3U3P/trajectory.json --output outputs/my_3U3P/replay.html
python tools/validate_coop3d.py --output outputs/new_acceptance
```

新训练和评估要求输出目录为空。`--updates` 表示本次额外执行的更新次数；恢复时配置必须完全一致。完整断点包含模型、两个 Adam 优化器、CPU/CUDA/Python/NumPy 随机状态及所有工作进程的环境和 RNG 状态。仅加载自己信任的本地 checkpoint。

默认每次更新采样 10 × 250 个环境转移，执行 10 轮 PPO；每次更新评估独立种子，以成功率、平均完成率、平均回报的字典序选择 `best.pt`，同时保存 `latest.pt`、配置快照和逐更新指标。三种规模分别训练，不承诺跨规模迁移。

## 环境接口

```python
import numpy as np
from coop3d import Cooperative3DEnv, EnvConfig
env = Cooperative3DEnv(EnvConfig(scale='3U5P'))
observations, info = env.reset(seed=7)
actions = np.tile([0.5, 1/3, 1/8], (3, 1))
observations, rewards, terminated, truncated, info = env.step(actions)
```

`reset(seed, options=...)` 支持显式 `states[N,6]`、`waypoints[M,3]`、`priorities[M]`、`active_mask[N]`。状态顺序是 `[x,y,z,v,gamma,psi]`，右手系 z 向上。固定内置场景设置 `random_scenario=False`。确定性故障示例：`EnvConfig(failure_events={20: [1]})`。

动作 `[N,3]` 取值 `[0,1]`，映射为滚转角 `[-π,π]`、切向过载 `[-1,2]`、法向过载 `[0,8]`。失效动作忽略；无任务有效 UAV 用同一动力学下的配平过载巡航。没有空间边界终止规则。每步信息包含状态、实际控制、前后掩码、前后分配、完成事件、五项奖励、友机距离和数值诊断。终止后必须 reset。

## 参数与结果

RK4：`g=9.81`、`dt=0.25 s`、速度 `100–400 m/s`、最多 250 步。±85° 航迹倾角限幅是数值稳定性假设，不是论文明确规定的飞行包线。Actor 固定 12 维；Critic 将局部观测补足五架并拼接成 60 维，输出一个集中状态价值。巡检球半径 300 m；步末位置判定，最小 UAV ID 唯一记账。Hungarian 独占匹配，提供 `nearest` 贪心最近分配基线。

完整归一化、奖励、失效时序、GAE 和自行补充的约定见 [技术规范](docs/coop3d_spec.md)。实际执行及结果见 [重构验收报告](docs/refactor_report.md)。短程训练只验证训练闭环，不代表策略已收敛。

`trajectory.json` 保存真实环境状态和事件，`replay.html` 是无需联网的独立页面。浏览器打开后支持播放/暂停、滑块、视角旋转、轨迹、完成标志、分配连线和逐 UAV 状态查询。页面只展示记录，不重新运行环境。

## 项目结构

```text
coop3d/
  config.py models.py dynamics.py integrator.py scenario.py
  assignment.py observation.py reward.py environment.py
  vector_env.py recording.py replay.py __main__.py
  mappo/                Beta Actor、集中 Critic、GAE、PPO、保存/恢复、评估
configs/coop3d_*.yaml    三种场景及一份 MAPPO 配置
tests/test_coop3d_*.py   自动化测试
tools/validate_coop3d.py 三规模 CUDA 验收
docs/                   技术规范、迁移清单、验收报告
outputs/                训练产物与回放（不纳入 Git）
```

原始 PDF、已有模型、JSON 回归数据保留，旧数据不构成兼容入口。旧依赖清单为 `docs/migration_inventory.json`；清理前源码恢复副本为 `tmp/pre_coop3d_source_backup.zip`。

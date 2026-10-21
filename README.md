# Coop3D v2.0 三维协同巡检基准

非武器化三维多UAV航路点巡检：3U3P、5U5P、3U5P。唯一环境 `coop3d-v2.0.0`，算法 `mappo-coop3d-v2.0.0`。环境在 `env/`，MAPPO 在 `algorithm/`，导航诊断及录像在 `tools/`。旧 `coop3d/` 包已移除，没有兼容转发入口。

本轮完成环境语义回归、90D集中状态、五个体价值Critic、Beta配平初始化、NaN安全GAE、严格采样预算、导航可达性审计和短程CUDA Smoke Test；没有运行正式长训练。

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

动力学、五项巡检奖励、Hungarian独占分配、0.25s/RK4、100–400m/s、±85°数值保护、250步、300m完成半径及100m安全参考保持不变。无任务配平、故障当步掩码和终止语义见 [环境规范](docs/environment_spec.md)。

默认Actor使用Beta `trim_centered` 初始化，动作均值接近 `[.5,1/3,1/8]`，浓度16，仍保留探索。优化参数与mask/GAE/预算/日志协议见 [MAPPO规范](docs/mappo_spec.md)。

## 诊断结果与回放

实际小样本审计：导航在3U3P、5U5P的固定场景和各5个随机种子上成功；3U5P随机成功1/5，固定场景完成3/5后超时。额外手工可达多任务场景79步完成5/5。它证明控制接口可用，不能据此声称所有随机场景都可完成。[完整导航审计](docs/navigation_audit.md)包含失败复核和两种Actor初始化的实际对照。

回放由真实JSON记录生成，HTML内嵌Plotly，可离线查看。支持播放/暂停、滑块、三维旋转、轨迹、速度方向、点完成与分配、控制、失效、限幅事件、五项奖励和逐UAV查询。渲染不执行环境。当前仍未完成浏览器内视觉交互验收，不能将脚本语法检查视为浏览器验证。

详细文件清单、测试和CUDA结果见 [重构报告](docs/refactor_report.md)。已有outputs、历史模型和原始PDF均保留。v1恢复副本在 `tmp/coop3d_v1_before_v2.zip`，迁移前参考轨迹在 `tests/fixtures/coop3d_v1_semantics.json.gz`。

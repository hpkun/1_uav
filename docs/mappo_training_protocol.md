# MAPPO Training Protocol

## 可复现运行

推荐在 WSL 的 `uav` Conda 环境执行：

```bash
conda run -n uav pip install -e .
conda run -n uav pytest
conda run -n uav python scripts/train_mappo.py --config configs/mappo_smoke_1v1.yaml --run-name smoke_1v1
conda run -n uav python scripts/train_mappo.py --config configs/mappo_smoke_2v2.yaml --run-name smoke_2v2
```

训练 seed 从配置的 `seed` 开始，各同步环境使用 `seed + env_index`。确定性评估默认从 100000 开始，绝不复用训练环境或其 RNG。训练步数定义为实际执行的环境决策步数乘以 `num_envs`。

## 输出与恢复

每次运行写入 `outputs/mappo/<run-name>/<run-id>/`，包含配置快照、训练/评估 CSV、TensorBoard、final summary，以及 `initial.pt`、`last.pt`、`best.pt` 和按间隔保存的 step checkpoint。完整 checkpoint 保存网络、两个 optimizer、ValueNormalizer、Python/NumPy/Torch RNG、并行环境的回合中状态和 runner 当前输入；actor-only 加载不会覆盖 Critic 或 optimizer。

```bash
python scripts/train_mappo.py --config configs/mappo_1v1_tail_chase.yaml --resume <last.pt> --run-name resumed
python scripts/evaluate_mappo.py --checkpoint <best.pt> --episodes 100 --seed-start 100000 --deterministic
python scripts/inspect_mappo_checkpoint.py --checkpoint <last.pt>
python scripts/plot_mappo_training.py --run-dir <run-directory>
```

best 的字典序是：较高 red win rate、较低 crash rate、较高 mean episode return。短探针只验证链路、有限数值和趋势，不证明收敛。正式结论至少使用多训练 seed、共享的独立评估 seed、均值、标准差和 95% 置信区间。

# MAPPO training protocol

Run from WSL with the `uav` conda environment. Training, validation, and final test roles are distinct. Validation seeds are used only during training and select `best.pt`; after training, initial/last/best are reevaluated on a non-overlapping test range. `final_summary.yaml` treats only test results as formal performance.

`checkpoint_selection: smoke` retains the simple overall-win/crash/return ordering and is limited to pipeline smoke such as `tail_chase`. Formal `head_on`, `balanced_random`, `head_on_formation`, `offset_formation`, and multi-aircraft balanced configurations use `checkpoint_selection: combat`: elimination win, decisive win, lower timeout, effective damage, overall red win, then team return.

Formal return semantics are:

```text
team_step_reward = mean(agent_rewards)
team_episode_return = sum_t team_step_reward
agent_sum_episode_return = sum_t sum(agent_rewards)
mean_per_agent_episode_return = agent_sum_episode_return / num_agents
```

Training still optimizes each agent's own rewards. `mean_episode_return` is retained only as an alias for team return. All rules and MAPPO comparisons use the same definition; in 1v1 all three returns coincide.

Use `scripts/run_mappo_multiseed.py` for sequential single-device seeds. Each `seed_summary.yaml` records validation provenance separately from final test evaluations. Use `scripts/aggregate_mappo_multiseed.py` to aggregate test results by default. Across independent training seeds it reports a Student-t 95% interval, sample standard deviation, median, and min/max. Per-episode binary rule comparisons continue to use Wilson intervals. A smoke test or one favorable seed demonstrates pipeline execution, not convergence.

Checkpoint v3 preserves networks, optimizers, ValueNormalizer, Python/NumPy/Torch RNG, vector environments, partial episode return accumulators, and the Trainer's independent minibatch generator state. A v2 checkpoint may initialize the Actor only; full v2 resume is rejected because its Critic value semantics are incompatible.

`overall_red_win_rate` includes timeout survivor-count wins. `elimination_win_rate` requires `blue_eliminated`; `timeout_survival_win_rate` is a red timeout win by survivor count; `decisive_win_rate` is any red non-timeout win. A timeout-survival win must not be reported as completing air-combat victory.

## Fixed homogeneous 3v3 reporting

For the audited 3v3 target, report fixed homogeneous 3v3 separately from all 1v1/2v2 probes. The 3v3 configuration uses 45D local observations, 87D global state, shared feed-forward Actor, centralized Critic, and the 4-process parallel smoke setup. Do not present 1v1 transfer, heterogeneous 3v2, attention, GRU, self-play, or network migration as implemented.

Before formal 3v3 training claims, run `scripts/audit_3v3_environment.py` and review `docs/3v3_environment_audit.md`. The audit currently records P1 unresolved items for health/state aliasing, distance-ranked entity slot swaps, and timeout survivor-count terminal semantics. Timeout survivor-count wins and elimination wins must remain separate in tables and checkpoint selection notes.

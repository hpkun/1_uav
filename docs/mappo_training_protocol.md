# MAPPO Training Protocol

Run commands from WSL with the `uav` conda environment. Training, validation, and final test roles are distinct: validation seeds select `best.pt`, and final test seeds are held out for reporting.

`checkpoint_selection: smoke` is only for pipeline smoke runs. Formal and learnability 3v3 runs use `checkpoint_selection: combat`, ranked by elimination win rate, overall red win rate, effective damage, survivor difference, hits, attack-area steps, team return, lower red crash rate, and finally lower timeout rate. Timeout rate is deliberately last so fast 0-win failures are not selected over checkpoints with survival, damage, or better return.

Return semantics are:

```text
team_step_reward = mean(agent_rewards)
team_episode_return = sum_t team_step_reward
agent_sum_episode_return = sum_t sum(agent_rewards)
mean_per_agent_episode_return = agent_sum_episode_return / num_agents
```

`overall_red_win_rate` includes timeout survivor-count wins. `elimination_win_rate` requires `blue_eliminated`. `timeout_survival_win_rate` is a red timeout win by survivor count. Timeout survivor-count wins, elimination wins, and draws must remain separate in reports.

## Stage 0: environment-only checks

Stage 0 verifies code and environment mechanics without learning. Run `pytest -q`; the learnability tests check schemas, shapes, deterministic reset, action controllability, rule-pursuit reachability against straight blue, timeout progress, and four-worker vector execution.

The optional helper `scripts/diagnose_3v3_reward_ordering.py` may be used to compare fixed policies, but it is not the main development path and should not be expanded into historical run analysis or bulk checkpoint ranking.

## Stage 1: basic 3v3 learnability

Use `configs/mappo_learnability_3v3.yaml`:

- scenario: `head_on_learnability_v1`
- opponent: `straight`
- seed: `1`
- total environment steps: `50000`
- validation/test episodes: `20`

This stage asks whether fixed homogeneous 3v3 MAPPO can learn approach, attack-area entry, attacks, hits, or damage in a lower-difficulty environment without changing rewards or MAPPO algorithms. A 4096-step smoke of this config is allowed to verify workers, shapes, GAE, PPO update, finite numerics, and checkpoint writing. It is not evidence of learned combat behavior.

Do not run 300k training, multi-seed training, or bulk checkpoint comparisons before Stage 1 passes.

## Stage 2: formal V2 strong opponent

The formal V2 setting remains:

```text
configs/scenario_3v3_v2.yaml
configs/mappo_3v3_v2.yaml
head_on_mirrored_jitter_v2
opponent: pursuit
homogeneous_3v3_v2_timeaware
fixed_id_body_time_63d
full_entity_time_61d
project_3v3_v2
```

Only after Stage 1 passes should training return to the formal pursuit-opponent environment and consider longer runs, additional seeds, checkpoint comparisons, or paper-style statistics.

# MAPPO training protocol

Run from WSL with the `uav` conda environment. Training seeds and evaluation seeds must be disjoint. Initial, last, and best checkpoints are all evaluated on the same independent seed set. Best checkpoint ranking is red win rate, then lower combined crash rate, then team return.

Formal return semantics are:

```text
team_step_reward = mean(agent_rewards)
team_episode_return = sum_t team_step_reward
agent_sum_episode_return = sum_t sum(agent_rewards)
mean_per_agent_episode_return = agent_sum_episode_return / num_agents
```

Training still optimizes each agent's own rewards. `mean_episode_return` is retained only as an alias for team return. All rules and MAPPO comparisons use the same definition; in 1v1 all three returns coincide.

Use `scripts/run_mappo_multiseed.py` for sequential single-device seeds. It supports per-seed directories, initial/last/best evaluation, matched evaluation seeds, completed-seed skipping, and last-checkpoint resume. Use `scripts/aggregate_mappo_multiseed.py` for mean, sample standard deviation, 95% normal CI, median, and min/max. A smoke test or one favorable seed demonstrates pipeline execution, not convergence.

Checkpoints preserve networks, optimizers, ValueNormalizer, Python/NumPy/Torch RNG, vector environments, partial episode return accumulators, and the Trainer's independent minibatch generator state.

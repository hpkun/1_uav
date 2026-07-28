# HAPPO implementation notes

This project implements HAPPO as an independent baseline beside MAPPO. It is intended for the current fixed homogeneous 3v3 setting: three red learning UAVs cooperate against three blue rule-controlled UAVs. It does not change the environment, reward function, observation schema, dynamics, combat geometry, damage model, timeout semantics, blue opponent, or existing MAPPO implementation.

## Paper basis

Main algorithmic source:

Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fanglei Sun, Jun Wang, Yaodong Yang, "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning," ICLR 2022, arXiv:2109.11251.

The implementation is aligned with the paper's cooperative HAPPO semantics:

- independent agent policies rather than a shared actor;
- sequential policy updates;
- cumulative probability-ratio factor from previously updated agents;
- scalar team reward and scalar centralized value function;
- PPO-style clipped surrogate for each actor update.

The code was also structured to be comparable to PKU-MARL/HARL's HAPPO organization, but it does not copy the HARL framework into this repository.

## Equation mapping

| Paper concept | Project implementation |
|---|---|
| Multi-Agent Advantage Decomposition Lemma | HAPPO uses one shared joint advantage from team reward and updates agents sequentially, so each later policy update is conditioned by the cumulative factor from earlier updated agents. |
| Eq. (9) / sequential decomposition idea | `src/uav_env/algorithms/happo/trainer.py::HAPPOTrainer.update` samples one update order and updates actors one by one. |
| Eq. (10) / cumulative probability ratio | `update_happo_factor()` multiplies `factor` by `exp(updated_log_prob_i - old_log_prob_i)` after actor `i` finishes all epochs/minibatches. |
| Eq. (11) / clipped HAPPO surrogate | `happo_policy_loss()` computes `factor * min(ratio*A, clip(ratio,1-eps,1+eps)*A)`. |
| Appendix D.4 HAPPO pseudocode | `HAPPOTrainer.update()` follows actor-order update, then critic update, using independent actor optimizers. |

The factor is detached. It is not clipped, is not the current actor's own ratio, is not recomputed per minibatch, and is reset to ones at the start of each HAPPO update.

## Independent actors

`IndependentActorSet` stores one actor per red UAV:

```text
actor_0
actor_1
actor_2
```

The three actors currently share architecture, but they are separate module objects with separate parameter storage, initialization, Adam optimizer state, checkpoint entries, and minibatch RNG states. The constructor accepts per-agent observation and action dimensions, but this should not be overstated as full heterogeneous training support. The full runner, adapter, and matrix rollout buffer currently assume fixed homogeneous 3v3 with uniform `obs_dim` and `action_dim`. A future heterogeneous runner would still need padding plus feature masks/action masks, or per-agent list-style buffers/tensors.

## Scalar centralized critic and joint team reward

HAPPO uses the environment scalar `team_reward` as the default training reward. The critic is:

```text
V(global_state) -> scalar
```

The current V2 global state is the existing 61D time-aware state. Per-agent environment rewards are kept in the HAPPO rollout buffer as diagnostics, but they are not used for default HAPPO GAE. This is deliberate: the original cooperative HAPPO theory assumes a shared team objective. This differs from the MAPPO path in this project, where rollout data can preserve per-agent reward semantics. A per-agent reward HAPPO mode would be a project extension and is intentionally not implemented here.

## Sequential update and factor timing

For each HAPPO update:

1. initialize `factor = ones([rollout_length, num_envs])`;
2. choose an agent order:
   - `[0, 1, 2]` when `fixed_agent_order: true`;
   - reproducible RNG permutation when `fixed_agent_order: false`;
3. update the selected actor with its own observations, actions, old log probabilities, action masks, active masks, the shared joint advantage, and current factor;
4. after the actor completes all PPO epochs and minibatches, recompute its new log probabilities over the full rollout and update:

```text
factor = factor * exp(updated_log_prob_i - rollout_old_log_prob_i)
```

Inactive samples use ratio 1 for factor updates, so dead/inactive agents do not alter later actors' factors. The factor is detached before it affects later actor losses.

## Critic update

The critic is updated after all actors finish. It uses:

- shared global state;
- scalar team reward returns;
- scalar GAE;
- optional ValueNorm;
- PPO clipped value loss;
- optional Huber loss;
- gradient clipping.

The time-aware V2 timeout semantics are preserved: Gymnasium still reports timeout as `truncated=True`, but HAPPO finite-horizon timeout does not bootstrap the critic.

## Optimizer settings

HAPPO actor and critic Adam optimizers use the HAPPO config fields:

```yaml
optimizer_eps: 1.0e-5
weight_decay: 0.0
```

These are local optimizer choices for the HAPPO baseline only. They do not change the MAPPO optimizer configuration.

## Reward diagnostics

Training still uses the scalar `team_reward` for HAPPO GAE and the scalar critic. Separately, the runner logs reward-component diagnostics from `agent_reward_breakdowns` so that training curves can be interpreted without changing the learning objective. The logged components include `situation_reward`, `geometry_event_reward`, `raw_shape_reward`, `assigned_shape_reward`, `combat_event_reward`, `dense_reward`, `terminal_reward`, `hit_event_reward`, `destroy_event_reward`, `attacked_event_penalty`, `destroyed_event_penalty`, and `boundary_collision_penalty`.

For each component, the runner records step mean, per-step joint sum, absolute per-step joint sum, and completed-episode accumulator mean. It also logs `agent_reward_sum_mean` beside the joint `team_reward_mean`. These diagnostics are saved/restored in full HAPPO checkpoints, but they are not used for advantage estimation or loss calculation.

## Files

- `src/uav_env/algorithms/happo/networks.py`
- `src/uav_env/algorithms/happo/rollout_buffer.py`
- `src/uav_env/algorithms/happo/trainer.py`
- `src/uav_env/algorithms/happo/checkpoint.py`
- `src/uav_env/algorithms/happo/config.py`
- `src/uav_env/algorithms/happo/runner.py`
- `scripts/train_happo.py`
- `configs/happo_base.yaml`
- `configs/happo_learnability_3v3.yaml`
- `configs/happo_3v3_v2.yaml`

## Checkpoints

HAPPO checkpoints save:

- all actor state dictionaries;
- all actor optimizer state dictionaries;
- critic state dictionary;
- critic optimizer state dictionary;
- ValueNormalizer state;
- environment steps and update index;
- best evaluation;
- agent-order RNG state;
- each actor minibatch RNG state;
- critic minibatch RNG state;
- vector environment state;
- current rollout interface state;
- episode accumulators;
- reward component episode accumulators;
- schema metadata.

MAPPO and HAPPO checkpoints are intentionally incompatible. This implementation does not try to convert a shared MAPPO actor into three HAPPO actors.

Full resume is intentionally strict about schema metadata: environment schema version, observation schema, global state schema, reward profile, scenario profile, observation dimension, state dimension, and agent count must match. Actor-only evaluation/loading is narrower but still guarded: it checks at least environment schema version, observation schema, observation dimension, agent count, and actor count. Actor-only loading may differ in scenario profile, reward profile, and global state schema because it does not restore critic, optimizer, buffer, or full runner state.

On `KeyboardInterrupt`, the runner saves `checkpoints/interrupted.pt` with actors, critic, optimizers, normalizer, vector/current state, accumulators, and RNG states, prints the checkpoint path, re-raises the interrupt, and closes the writer/vector environment. It does not run final test evaluation after an interrupt.

## Evaluation metrics

HAPPO evaluation runs each actor separately on its own local observation and chooses argmax in deterministic mode. It reports the existing combat metrics plus per-actor action frequencies, entropy, and top1-top2 logit margin. Timeout survivor-count outcomes are logged through the existing combat outcome metrics and are not relabeled as elimination wins.

For compatibility with the existing smoke/combat checkpoint-selection keys, HAPPO evaluation includes aliases: `mean_episode_return = mean_team_episode_return`, `mean_effective_damage = mean_red_effective_damage`, `mean_hits = mean_red_hits`, `mean_attack_area_steps = mean_red_attack_area_steps`, and `mean_survivor_difference = mean_red_survivors - mean_blue_survivors`. It also reports `red_crash_rate` and `blue_crash_rate`. Crash rate is defined as the episode proportion in which that side has at least one ground crash. Ceiling violations are reported separately and do not count as crashes.

## End-of-run validation and test evaluation

At the start of a fresh run, HAPPO saves `initial.pt`. At the end of training it always saves `last.pt`. If no validation occurred during the loop, or if the last validation was not at the final environment step, the runner performs exactly one final validation. It also guarantees that `best.pt` exists.

After final validation, the runner loads `initial.pt`, `last.pt`, and `best.pt` in actor-only mode and evaluates each checkpoint deterministically with `test_seed_start` and `test_episodes`. These held-out results are written to `final_summary.yaml` under `test_evaluations` together with total `wall_time`.

## What is paper-aligned vs project adaptation

Paper-aligned:

- independent actors;
- sequential actor update;
- cumulative factor in the actor surrogate;
- shared cooperative reward;
- scalar centralized critic.

Project adaptation:

- the environment is the existing fixed homogeneous 3v3 time-aware V2 UAV environment;
- action space is the existing 15-action discrete table;
- blue UAVs remain rule-controlled by the existing environment;
- finite-horizon timeout no-bootstrap follows the current V2 training semantics;
- network sizes and optimizer settings are local configuration choices.

HAPPO here is not self-play and does not make both red and blue learn simultaneously. Red-side cooperation is learned by HAPPO; environmental adversarial pressure comes from the fixed blue rule policy.

The implementation should not be described as guaranteeing monotonic improvement for every finite-sample neural-network update. The paper's improvement claims depend on theoretical assumptions; this code is a practical approximate implementation.

## Example command for later server training

Do not run this as part of quick validation. On an Ubuntu server with the `uav` conda environment:

```bash
cd /path/to/1_uav
conda run -n uav python scripts/train_happo.py \
  --config configs/happo_learnability_3v3.yaml \
  --run-name happo_learnability_3v3
```

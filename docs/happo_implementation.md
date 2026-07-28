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

The three actors currently share architecture, but they are separate module objects with separate parameter storage, initialization, Adam optimizer state, checkpoint entries, and minibatch RNG states. The constructor accepts per-agent observation and action dimensions so future heterogeneous UAV experiments can reuse the API.

## Scalar centralized critic and joint team reward

HAPPO uses the environment scalar `team_reward` as the default training reward. The critic is:

```text
V(global_state) -> scalar
```

The current V2 global state is the existing 61D time-aware state. Per-agent environment rewards are kept in the HAPPO rollout buffer as diagnostics, but they are not used for default HAPPO GAE. This is deliberate: the original cooperative HAPPO theory assumes a shared team objective. A per-agent reward HAPPO mode would be a project extension and is intentionally not implemented here.

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
- schema metadata.

MAPPO and HAPPO checkpoints are intentionally incompatible. This implementation does not try to convert a shared MAPPO actor into three HAPPO actors.

## Evaluation metrics

HAPPO evaluation runs each actor separately on its own local observation and chooses argmax in deterministic mode. It reports the existing combat metrics plus per-actor action frequencies, entropy, and top1-top2 logit margin. Timeout survivor-count outcomes are logged through the existing combat outcome metrics and are not relabeled as elimination wins.

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

# MAPPO design

This is a feed-forward homogeneous 1v1/2v2 CTDE baseline, not a layer-by-layer reproduction of the 2024 network. A shared Actor consumes local observations (11D or 28D). The centralized Critic consumes global state (10D or 40D) plus explicit agent identity. It does not implement attention, recurrence, self-play, heterogeneity, transfer, or 3v2.

Rollouts use `[T,num_envs,num_agents,...]`. Actor active masks remove dead agents from policy and entropy losses. Critic masks remain active while the team episode continues, so post-death assigned rewards can enter value targets. True termination uses zero bootstrap; truncation bootstraps the retained terminal state; either boundary breaks recursive GAE before an auto-reset state.

PPO policy and value calculations are exposed as pure functions. `use_huber_loss=false` selects `0.5*squared_error`; `use_clipped_value_loss=false` disables value clipping. An all-zero Actor mask skips the Actor optimizer step and produces finite zero policy diagnostics.

Value data flow is deliberately physical-scale. The Critic output layer, rollout values, terminal bootstrap values, GAE, and returns all use the environment reward scale. `ValueNormalizer` is used only inside the Critic loss: after updating its statistics, the same mean/variance snapshot transforms new physical predictions, old physical predictions, and physical return targets before value clipping. It is target normalization, not PopArt, and it never changes the Critic output layer's numerical meaning. Explained variance compares physical Critic predictions directly with physical returns.

## Fixed homogeneous 3v3 V2 status

The legacy fixed homogeneous 3v3 path remains available with 45D local observations and an 87D centralized state for backward compatibility and audit comparison. It should be treated as a legacy experimental interface with known observation/state sufficiency risks.

The formal V2 fixed homogeneous 3v3 path is selected by `environment_schema_version: homogeneous_3v3_v2_timeaware`. It keeps the same shared feed-forward Actor and centralized feed-forward Critic code, but changes the environment interface to a 63D fixed-ID body-frame local observation and a 61D full-entity global state. The extra field is `episode_progress`, normalized to `[-1, 1]`, which removes finite-horizon time aliasing without adding duplicate time variables. Checkpoints store this schema metadata and full resume rejects mismatched environment, observation, state, reward, scenario, dimension, or agent-count metadata. Actor-only loading still relies on model tensor shape compatibility. The earlier 62D/60D V2 smoke schema is not runnable and is not auto-migrated.

The implementation intentionally still does not include the 2024 paper's transfer network, three-subnetwork Actor, GRU, attention, self-play, heterogeneous 3v2, or network migration.

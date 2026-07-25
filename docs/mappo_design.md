# MAPPO design

This is a feed-forward homogeneous 1v1/2v2 CTDE baseline, not a layer-by-layer reproduction of the 2024 network. A shared Actor consumes local observations (11D or 28D). The centralized Critic consumes global state (10D or 40D) plus explicit agent identity. It does not implement attention, recurrence, self-play, heterogeneity, transfer, or 3v2.

Rollouts use `[T,num_envs,num_agents,...]`. Actor active masks remove dead agents from policy and entropy losses. Critic masks remain active while the team episode continues, so post-death assigned rewards can enter value targets. True termination uses zero bootstrap; truncation bootstraps the retained terminal state; either boundary breaks recursive GAE before an auto-reset state.

PPO policy and value calculations are exposed as pure functions. `use_huber_loss=false` selects `0.5*squared_error`; `use_clipped_value_loss=false` disables value clipping. An all-zero Actor mask skips the Actor optimizer step and produces finite zero policy diagnostics.

Value data flow is deliberately physical-scale. The Critic output layer, rollout values, terminal bootstrap values, GAE, and returns all use the environment reward scale. `ValueNormalizer` is used only inside the Critic loss: after updating its statistics, the same mean/variance snapshot transforms new physical predictions, old physical predictions, and physical return targets before value clipping. It is target normalization, not PopArt, and it never changes the Critic output layer's numerical meaning. Explained variance compares physical Critic predictions directly with physical returns.

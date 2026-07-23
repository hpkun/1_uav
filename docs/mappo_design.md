# MAPPO design

This is a feed-forward homogeneous 1v1/2v2 CTDE baseline, not a layer-by-layer reproduction of the 2024 network. A shared Actor consumes local observations (11D or 28D). The centralized Critic consumes global state (10D or 40D) plus explicit agent identity. It does not implement attention, recurrence, self-play, heterogeneity, transfer, or 3v2.

Rollouts use `[T,num_envs,num_agents,...]`. Actor active masks remove dead agents from policy and entropy losses. Critic masks remain active while the team episode continues, so post-death assigned rewards can enter value targets. True termination uses zero bootstrap; truncation bootstraps the retained terminal state; either boundary breaks recursive GAE before an auto-reset state.

PPO policy and value calculations are exposed as pure functions. `use_huber_loss=false` selects `0.5*squared_error`; `use_clipped_value_loss=false` disables value clipping. An all-zero Actor mask skips the Actor optimizer step and produces finite zero policy diagnostics.

Value data flow is deliberate: Critic output is normalized value; rollout storage and GAE use denormalized physical values; returns remain in raw reward scale; training normalizes targets and old values into the same space before value clipping. Explained variance is computed after denormalizing predictions, in physical-return space.

"""Disk retention only; never changes the collector's checkpoint boundaries."""


def checkpoint_policy(config, *, legacy=False):
    interval = config.checkpoint_interval_episodes
    return dict(interval_episodes=interval if legacy or interval != 50 else 250,
                extra_episodes=[] if legacy or interval != 50 else [4950])


def checkpoint_targets(config, policy=None):
    policy = checkpoint_policy(config) if policy is None else policy
    # Only select existing collector boundaries, including custom intervals.
    return [episode for episode in range(config.checkpoint_interval_episodes,
                                        config.training_episodes + 1,
                                        config.checkpoint_interval_episodes)
            if episode % policy['interval_episodes'] == 0 or episode in policy['extra_episodes']]

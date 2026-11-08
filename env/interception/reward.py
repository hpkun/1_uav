import numpy as np


def reward_parts(before, after, captures, outcome, duration, config, *, episode_time=None):
    progress = config.progress_weight * np.mean(np.asarray(before) - after)
    failed = outcome not in ("ongoing", "success", "timeout")
    if failed:
        progress = min(progress, 0.)
    # Charge unused time on failures so ending early cannot save the time cost.
    remaining = (max(0., config.max_steps*config.dt-episode_time)
                 if failed and episode_time is not None else 0.)
    return {
        "capture_event": config.capture_reward * captures,
        "terminal_success": config.success_reward if outcome == "success" else 0.,
        "terminal_failure": {"breach": -config.breach_cost,
                             "evader_escape": -config.escape_cost,
                             "evader_collision": -config.collision_cost}.get(outcome, 0.),
        "progress": float(progress / config.progress_reference),
        "collision_safety": {"friendly_collision": -config.collision_cost,
                             "pursuer_boundary_violation": -config.boundary_cost}.get(outcome, 0.),
        "time_penalty": -config.step_cost * (duration+remaining) / config.dt,
    }

import numpy as np


def reward_parts(before, after, captures, outcome, duration, config):
    progress = config.progress_weight * np.mean(np.asarray(before) - after)
    if outcome in ("collision", "pursuer_boundary_violation", "evader_escape"):
        progress = min(progress, 0.)
    return {
        "capture_event": config.capture_reward * captures,
        "terminal_success": config.success_reward if outcome == "success" else 0.,
        "terminal_failure": -config.breach_cost if outcome == "breach" else 0.,
        "progress": float(progress / config.progress_reference),
        "collision_safety": {"collision": -config.collision_cost,
                             "pursuer_boundary_violation": -config.boundary_cost}.get(outcome, 0.),
        "time_penalty": -config.step_cost * duration / config.dt,
    }

import numpy as np


def reward_parts(before, after, captures, outcome, duration, config):
    progress = config.progress_weight * np.mean(np.asarray(before) - after)
    return {
        "capture_event": config.capture_reward * captures,
        "terminal_success": config.success_reward if outcome == "success" else 0.,
        "terminal_failure": -config.breach_cost if outcome == "breach" else 0.,
        "progress": float(progress / config.progress_reference),
        "collision_safety": -config.collision_cost if outcome == "collision" else 0.,
        "time_penalty": -config.step_cost * duration / config.dt,
    }

"""Two-position constant-velocity extrapolation at one and two time steps."""
import math
import numpy as np


def predict_zuo_two_steps(history_positions, dt):
    positions=np.asarray(history_positions,dtype=np.float64)
    if positions.shape!=(2,3) or not np.isfinite(positions).all():
        raise ValueError("history_positions must be a finite (2, 3) array")
    if not math.isfinite(dt) or dt<=0:
        raise ValueError("dt must be finite and positive")
    previous,current=positions
    velocity=(current-previous)/dt
    result=np.stack((current+velocity*dt,current+velocity*(2*dt)))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite linear prediction")
    return result

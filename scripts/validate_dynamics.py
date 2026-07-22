"""Run short representative point-mass dynamics checks."""

from __future__ import annotations

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15, get_action_name, get_control
from uav_env.dynamics.point_mass_3d import point_mass_3d_derivative
from uav_env.dynamics.rk4 import rk4_step


CASES = (
    DiscreteAction15.LEVEL_HOLD,
    DiscreteAction15.LEVEL_ACCELERATE,
    DiscreteAction15.LEVEL_DECELERATE,
    DiscreteAction15.CLIMB_HOLD,
    DiscreteAction15.DIVE_HOLD,
    DiscreteAction15.LEFT_HOLD,
    DiscreteAction15.RIGHT_HOLD,
)


def main() -> None:
    """Integrate each representative action for one second and print deltas."""

    initial = np.asarray([0.0, 0.0, 1_000.0, 100.0, 0.0, 0.0], dtype=np.float64)
    for action in CASES:
        control = get_control(action)

        def derivative(time: float, state: np.ndarray, command: object) -> np.ndarray:
            del time, command
            return point_mass_3d_derivative(state, control)

        state = initial.copy()
        time = 0.0
        for _ in range(10):
            state = rk4_step(derivative, time, state, 0.1, control)
            time += 0.1
        delta = state - initial
        print(
            f"{get_action_name(action):<6} "
            f"dx={delta[0]:9.3f} dy={delta[1]:9.3f} dz={delta[2]:9.3f} "
            f"dv={delta[3]:8.3f} dtheta={delta[4]:8.5f} dpsi={delta[5]:8.5f}"
        )


if __name__ == "__main__":
    main()

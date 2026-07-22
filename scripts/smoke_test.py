"""Run a small action-table and dynamics smoke test."""

from __future__ import annotations

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15, get_action_name, get_control, validate_action_table
from uav_env.core.state import UAVState
from uav_env.dynamics.point_mass_3d import point_mass_3d_derivative
from uav_env.dynamics.rk4 import rk4_step


def main() -> None:
    """Print all actions and advance one state by a single RK4 step."""

    validate_action_table()
    print("15 discrete actions:")
    for action in DiscreteAction15:
        print(f"{int(action):2d} {get_action_name(action):<6} {get_control(action).to_vector()}")

    state = UAVState(
        x=0.0,
        y=0.0,
        z=1_000.0,
        speed=100.0,
        flight_path_angle=0.0,
        heading_angle=0.0,
        health=300.0,
        alive=True,
        team_id=0,
        type_id="homogeneous_baseline",
    )
    control = get_control(DiscreteAction15.LEVEL_HOLD)

    def derivative(time: float, vector: np.ndarray, command: object) -> np.ndarray:
        del time
        assert command is control
        return point_mass_3d_derivative(vector, control)

    before = state.to_kinematic_vector()
    after = rk4_step(derivative, 0.0, before, 0.1, control)
    print(f"Before RK4: {before.tolist()}")
    print(f"After RK4:  {after.tolist()}")


if __name__ == "__main__":
    main()

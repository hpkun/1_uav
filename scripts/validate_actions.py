"""Validate and summarize the fixed action table."""

from math import cos, degrees

from uav_env.actions.discrete_15 import DiscreteAction15, get_control, validate_action_table


def main() -> None:
    """Print turn-angle invariants after validating all action mappings."""

    validate_action_table()
    left = get_control(DiscreteAction15.LEFT_HOLD)
    right = get_control(DiscreteAction15.RIGHT_HOLD)
    print("Action table: valid")
    print(f"Left bank:  {degrees(left.bank_angle):.6f} deg")
    print(f"Right bank: {degrees(right.bank_angle):.6f} deg")
    print(f"Left vertical overload:  {left.normal_overload * cos(left.bank_angle):.6f}")
    print(f"Right vertical overload: {right.normal_overload * cos(right.bank_angle):.6f}")


if __name__ == "__main__":
    main()

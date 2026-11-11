"""Optimistic geometric necessary condition; never a success certificate."""
import numpy as np

from .config import IdealConfig
from .environment import IdealEnv


def _earliest_possible(start, speed, radius, path, deadline):
    for left, right, target, velocity in path.segments(0., deadline):
        relative = target-np.asarray(start)
        allowance = speed*left+radius
        a = float(velocity@velocity-speed*speed)
        b = float(2*(relative@velocity-allowance*speed))
        c = float(relative@relative-allowance*allowance)
        span = right-left
        if c <= 1e-10:
            return left
        polynomial = [a,b,c] if abs(a) > 1e-12 else [b,c] if abs(b) > 1e-12 else []
        roots = np.roots(polynomial) if polynomial else []
        possible = sorted(float(root.real) for root in roots
                          if abs(root.imag) < 1e-8 and -1e-9 <= root.real <= span+1e-9)
        if possible:
            return left+max(0.,possible[0])
        if a*span*span+b*span+c <= 0:
            # Conservative fallback for a numerically degenerate root.
            return left
    return None


def analyze(config: IdealConfig):
    if not config.target_count:
        raise ValueError("feasibility bound applies to IK2/IK3 tracking")
    env = IdealEnv(config)
    _, initial = env.reset(config.seed)
    deadline = config.dt*config.max_steps
    allowed_invalid = (1-config.tracking_success_fraction)*deadline
    members = []
    for i in range(config.active_count):
        assignment = env.assignment[i]
        path = env.references[assignment]
        start = env.states[i,:3]
        earliest = _earliest_possible(start, config.speeds[i],
                    config.tracking_distance, path, deadline)
        upper = max(0., 1-earliest/deadline) if earliest is not None else 0.
        members.append(dict(uav=i, target=assignment,
            initial_distance=float(np.linalg.norm(start-path.sample(0.)[0])),
            uav_speed=config.speeds[i], target_speed=config.target_speed,
            earliest_optimistic_entry=earliest, latest_possible_fraction=upper,
            necessary_reachable=earliest is not None and earliest <= allowed_invalid+1e-10))
    return dict(task=config.task, scenario_id=initial["scenario_id"],
        deadline=deadline, tracking_distance=config.tracking_distance,
        contact_distance=config.contact_distance, success_fraction=config.tracking_success_fraction,
        members=members, necessary_reachable=all(row["necessary_reachable"] for row in members),
        interpretation="necessary_only; ignores turn and safety constraints")


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("IK2", "IK3"), default="IK2")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    config = (IdealConfig(**json.loads(args.config.read_text(encoding="utf-8")))
              if args.config else IdealConfig(task=args.task))
    print(json.dumps(analyze(config), indent=2))


if __name__ == "__main__":
    main()

"""Generate paired A/B 20v20 episodes with repository Zuo assignment, no avoidance."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from algorithm.common.serialization import serializable
from algorithm.ideal_collision_avoidance import AvoidanceConfig
from algorithm.ideal_20v20_zuo_assignment import load_balanced_assignment
import kinematic_interception.evaluation as evaluation
from kinematic_interception.environment import KinematicInterceptionEnv, twenty_scene
from kinematic_interception.replay import render_mp4


class ZuoKinematicEnv(KinematicInterceptionEnv):
    """Reuse the Zuo allocation formula, retaining unaffected assignments."""

    def reset(self):
        super().reset()
        self.assignment = load_balanced_assignment(
            self.defender_positions(), self.intruder_positions(),
            self.defender_mask, self.intruder_mask)
        return self.snapshot()

    def _resolve(self, kind, members, at, position_arrays):
        previous = self.assignment.copy()
        super()._resolve(kind, members, at, position_arrays)
        # The base class recalculates nearest-target matching; replace it with
        # the Zuo policy: preserve valid bindings and reassign only those whose
        # assigned target has exited, accounting for existing target loads.
        kept = {i: j for i, j in previous.items()
                if self.defender_mask[i] and self.intruder_mask[j]}
        affected = [i for i, j in sorted(previous.items())
                    if self.defender_mask[i] and not self.intruder_mask[j]]
        if affected and self.intruder_mask.any():
            affected_mask = np.zeros_like(self.defender_mask)
            affected_mask[affected] = True
            kept.update(load_balanced_assignment(
                position_arrays[0], position_arrays[1], affected_mask,
                self.intruder_mask, initial_loads=Counter(kept.values())))
        self.assignment = kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--method', choices=('G0','G1','G2','G3'), default='G2')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--outdir', type=Path, default=Path('outputs/kinematic_zuo_ab_noavoid'))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Replace only the environment constructor used in this experiment.
    evaluation.KinematicInterceptionEnv = ZuoKinematicEnv
    scene = twenty_scene(args.seed, count=20)
    initial_assignments = []
    for model in ('A', 'B'):
        stem = args.outdir / f'20v20_zuo_{model}_{args.method}_seed{args.seed}_noavoid'
        json_path = stem.with_suffix('.json')
        mp4_path = stem.with_suffix('.mp4')
        if json_path.exists() or mp4_path.exists():
            raise FileExistsError(f'Output already exists: {stem}.*')
        result, trace = evaluation.run_episode(
            scene, model, args.method,
            avoidance=AvoidanceConfig(mode='none'), substep=0.01, trace=True)
        result.update(seed=args.seed, assignment_policy='zuo_load_balanced_target_resolution')
        initial_assignments.append(trace['frames'][0]['assignment'])
        with json_path.open('x', encoding='utf-8') as f:
            json.dump(serializable({'summary': result, 'trace': trace}), f,
                      ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        render_mp4(json_path, mp4_path, fps=args.fps)
        print(f"{model}: intercepted={result['intercepted_count']}/20 "
              f"breached={result['breached_count']} "
              f"friendly_collisions={result['friendly_collision_count']} "
              f"outcome={result['outcome']}\n  {json_path}\n  {mp4_path}")
    assert initial_assignments[0] == initial_assignments[1], 'A/B initial allocations differ'
    print('Verified: same initial Zuo assignment, same reference scene, no avoidance.')


if __name__ == '__main__':
    main()

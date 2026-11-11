"""Short CPU step benchmark for the independent ideal environment."""
import argparse
from collections import defaultdict
from dataclasses import replace
from time import perf_counter
import numpy as np

import env.ideal_kinematic.step as stepping
from env.ideal_kinematic import IdealConfig, IdealEnv
from env.ideal_kinematic.reference import Reference


def benchmark(samples):
    times = defaultdict(float)
    counts = defaultdict(int)
    originals = {}
    for name in ("boundary", "closest", "proximity", "inside_intervals", "_error_integral"):
        original = getattr(stepping, name)
        originals[name] = original
        def timed(*args, _name=name, _original=original, **kwargs):
            start = perf_counter()
            result = _original(*args, **kwargs)
            times[_name] += perf_counter()-start
            counts[_name] += 1
            return result
        setattr(stepping, name, timed)
    original_segments = Reference.segments
    def timed_segments(self, *args):
        iterator = original_segments(self, *args)
        while True:
            start = perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                times["reference_segments"] += perf_counter()-start
                counts["reference_segments"] += 1
                return
            times["reference_segments"] += perf_counter()-start
            counts["reference_segments"] += 1
            yield item
    Reference.segments = timed_segments
    original_metrics = IdealEnv.tracking_metrics
    def timed_metrics(self):
        start = perf_counter()
        result = original_metrics(self)
        times["tracking_metrics"] += perf_counter()-start
        counts["tracking_metrics"] += 1
        return result
    IdealEnv.tracking_metrics = timed_metrics
    try:
        for task in ("IK0", "IK1", "IK2", "IK3"):
            for mode in ("straight", "bezier"):
                times.clear()
                counts.clear()
                config = replace(IdealConfig(task=task), target_mode=mode)
                env = IdealEnv(config)
                env.reset(7)
                action = np.full((5,3), .5)
                start = perf_counter()
                for _ in range(samples):
                    if env.outcome != "ongoing":
                        env.reset(7)
                    env.step(action)
                elapsed = perf_counter()-start
                events = sum(times[key] for key in
                    ("boundary", "closest", "proximity", "inside_intervals"))
                tracking = times["_error_integral"]+times["tracking_metrics"]
                print(f"{task} {mode}: {elapsed/samples*1000:.3f} ms/step, "
                      f"{samples/elapsed:.1f} steps/s, event_primitives={events/samples*1000:.3f} ms, "
                      f"reference_segments={times['reference_segments']/samples*1000:.3f} ms, "
                      f"tracking_metrics={tracking/samples*1000:.3f} ms")
    finally:
        for name, original in originals.items():
            setattr(stepping, name, original)
        Reference.segments = original_segments
        IdealEnv.tracking_metrics = original_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    benchmark(args.steps)

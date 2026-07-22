"""Compare the configured damage distribution with Monte Carlo frequencies."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from uav_env.combat.damage import DamageConfig, damage_for_random_value


def main() -> None:
    """Sample at least 100,000 values and print theoretical and empirical rates."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    if args.samples < 100_000:
        raise ValueError("--samples must be at least 100000")
    config = DamageConfig()
    random_values = np.random.default_rng(args.seed).random(args.samples)
    counts = Counter(damage_for_random_value(float(value), config) for value in random_values)
    theoretical = (0.1, 0.3, 0.4, 0.2)
    print(f"Samples: {args.samples}, seed: {args.seed}")
    for damage, probability in zip(config.damage_values, theoretical):
        empirical = counts[damage] / args.samples
        print(f"damage={damage:4.1f} theoretical={probability:.4f} empirical={empirical:.4f}")


if __name__ == "__main__":
    main()

# MAPPO UAV Combat Research Project

This repository contains the frozen Direct V2.3 4v4 UAV combat environment,
the Persistent-Wave V1/V2 mission variants, and the MAPPO baseline. The current
mainline is `persistent_wave_v2`; Direct V2.3 remains the single-round base
environment. Observation dimension (52), action dimension (3), environment
semantics, and MAPPO implementation version 2 are unchanged by the repository
layout.

## Project layout

```text
algorithm/  MAPPO implementation, shared RL utilities, train/evaluate entries
env/        Direct V2.3 and Persistent-Wave environments and all foundations
configs/    Flat environment and algorithm YAML files
outputs/    One direct child directory per experiment
tools/      Validation, audit, aggregation, and plotting utilities
tests/      Flat automated test suite
papers/     Reference paper PDFs
docs/       Active specifications and historical archives
```

## Install third-party dependencies

Use Python 3.10 or newer and install only the third-party dependencies:

```bash
pip install -r requirements.txt
```

The project itself does not need editable or wheel installation. `pytest` works
from the project root, while algorithm entries and tools bootstrap the project
root themselves and can be executed from any working directory.

```bash
pytest -q
python tools/validate_combat_environment.py
```

## Smoke test

```bash
python algorithm/train_mappo.py \
  --smoke \
  --device cpu \
  --num-envs 1 \
  --output-dir outputs/mappo_smoke
```

The default smoke and formal environment is Persistent-Wave V2, using
`configs/persistent_wave_v2_environment.yaml` and
`configs/mappo_persistent_wave.yaml`.

## Persistent-Wave V2 formal training

```bash
python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --num-envs 24 \
  --total-sampled-steps 8000000 \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/mappo_pw_v2_8m_seed2023
```

Persistent MAPPO uses `gamma=0.999`.

## Direct V2.3 training

```bash
python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --num-envs 24 \
  --total-sampled-steps 500000 \
  --env-config configs/combat_environment.yaml \
  --algorithm-config configs/mappo.yaml \
  --output-dir outputs/mappo_direct_v2_3_seed2023
```

Direct MAPPO uses `gamma=0.99`.

## Checkpoint evaluation

```bash
python algorithm/evaluate_mappo.py \
  --checkpoint outputs/mappo_pw_v2_8m_seed2023/best_eval.pt \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --seed-base 20000000 \
  --episodes 20 \
  --device cpu \
  --output outputs/mappo_pw_v2_8m_seed2023/holdout_evaluation.json
```

Checkpoint loading validates `environment_version`, `environment_variant`, and
MAPPO implementation compatibility before formal use.

## Validation and tools

Common direct commands include:

```bash
python tools/validate_combat_environment.py
python tools/validate_persistent_wave_environment.py
python tools/check_parallel_env.py --num-envs 2 --steps 10
python tools/stress_test_ground_guard.py --output outputs/ground_guard_check.json
```

All directly executable tools resolve project-relative configs, checkpoints,
inputs, and outputs against the repository root.

## Experiment output contract

One run equals one direct child directory under `outputs/`:

```text
outputs/
└── mappo_pw_v2_8m_seed2023/
    ├── algorithm_config.yaml
    ├── env_config.yaml
    ├── run_config.json
    ├── train.log
    ├── training_metrics.jsonl
    ├── optimization_metrics.jsonl
    ├── evaluation_history.csv
    ├── best_eval.pt
    ├── checkpoint_*.pt
    ├── latest.pt
    └── run_summary.json
```

The `--output-dir` value is the final run directory. No automatic
`run_seed_*` or algorithm/environment/seed hierarchy is added. If omitted, the
entry creates one timestamped directory directly under `outputs/`. Every run
stores snapshots of both YAML inputs plus a JSON record of effective command-line
settings.

## Environment contract

- Four homogeneous learned Red UAVs fight four deterministic Blue UAVs.
- NED 3DOF point-mass dynamics use RK4 with `dt=0.1 s`.
- Actions are relative heading, pitch, and speed commands.
- The observation remains exactly 52 floats per Red agent.
- The weapon uses the frozen V2.3 3-D firing cone and probabilistic hit model.
- Rewards remain the frozen R1-R4 terms.
- Persistent-Wave V2 preserves the multi-wave contract and uses the existing
  ground-aware nearest-target Blue policy.

The normative Direct formulas and update order are in
`docs/environment_v2_spec.md`. Persistent mission behavior and variant identity
are documented in `docs/persistent_wave_environment_design.md`. Files under
`docs/archive/` are historical records and are not rewritten as active runtime
instructions.


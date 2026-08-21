# MADSAC 4v4 UAV Combat Benchmark

This repository implements MADSAC against a public, lightweight 3D combat
environment. The active environment is **Paper-Constrained Direct 4v4 Combat
Environment V2.1**. It uses the motion equations, action increments, combat
geometry, attack equations, observation content and segmented reward structure
reported by Li et al. (2023), with every otherwise missing simulator choice
declared in `docs/environment_v2_spec.md`.

The benchmark is a direct 4v4 engagement: four learned Red agents share an actor
network and fight four Blue aircraft using a deterministic nearest-target pursuit
policy. The observation and action dimensions remain 52 and 3. MADSAC network,
optimizer and Algorithm-1 scheduler semantics are unchanged by V2.1.

## Install and test

Install the package in editable mode in a Python environment containing the
dependencies declared by `pyproject.toml`, then run:

```bash
pytest -q
python scripts/validate_combat_environment.py
```

The validation report covers the controller grid, Eq. (8) Monte Carlo behavior,
1000 randomized initializations and a 100-episode rule-based combat baseline.

## Train

```bash
python -u scripts/train_madsac.py \
  --device cuda \
  --seed 2023 \
  --total-sampled-steps 500000 \
  --num-envs 24 \
  --output-dir outputs/madsac_v2_1
```

Training uses one persistent spawned worker per environment. Console output stays
compact; `training_metrics.jsonl`, evaluation history, summaries and checkpoints
retain the complete diagnostic record.

V2.0 checkpoints are intentionally incompatible. V2.1 checkpoints contain an
`environment_version` field, and resume fails before loading model weights when
the field is absent or different. Replay contents are not checkpointed, so even a
compatible resume restarts replay collection and episodes.

## Environment contract

- NED 3DOF point-mass dynamics, RK4, `dt=0.1 s`.
- Relative action increments `[a_psi, a_theta, a_v]` with physical maxima
  `[pi rad, pi/3 rad, 50 m/s]`.
- Eq. (2) inverse controller with 2 s response constants and proportional
  `nz <= 8` projection.
- Random 8 km diameter initialization inside a hard 5 km arena.
- Eq. (7) fire window and Eq. (8) probabilistic, entry-triggered attacks.
- Paper R1-R4 only; ground and boundary semantics are explicitly separated.
- A 100 s timeout is a Red mission failure, not a draw.

See `docs/environment_v2_spec.md` for the normative formulas, indices, update
order, provenance and validation criteria.

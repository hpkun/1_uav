# MADSAC and MAPPO 4v4 UAV Combat Benchmark

This repository implements MADSAC and MAPPO against a public, lightweight 3D combat
environment. The active environment is **Paper-Constrained Direct 4v4 Combat
Environment V2.2**. It uses the motion equations, action increments, combat
geometry, attack equations, observation content and segmented reward structure
reported by Li et al. (2023), with every otherwise missing simulator choice
declared in `docs/environment_v2_spec.md`.

The benchmark is a direct 4v4 engagement: four learned Red agents share an actor
network and fight four Blue aircraft using a deterministic nearest-target pursuit
policy. The observation and action dimensions remain 52 and 3. MAPPO uses a
shared two-layer local actor and a centralized two-head attention value critic,
matching the MADSAC network width while retaining an on-policy PPO/GAE update.

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
  --output-dir outputs/madsac_v2_2
```

Training uses one persistent spawned worker per environment. Console output stays
compact; `training_metrics.jsonl`, evaluation history, summaries and checkpoints
retain the complete diagnostic record.

MAPPO uses the same environment, evaluation seeds and output schema:

```bash
python -u scripts/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --total-sampled-steps 8000000 \
  --num-envs 24 \
  --output-dir outputs/mappo_v2_2_8m_seed2023
```

Its formal hyperparameters are declared in `configs/mappo.yaml`. MAPPO
checkpoints contain the actor, centralized critic, both optimizers and training
counters; no partial on-policy rollout is serialized.

V2.0/V2.1 checkpoints are intentionally incompatible. V2.2 checkpoints contain an
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
- A 4 km, 30-degree true 3-D off-boresight fire cone and Eq. (8)
  probabilistic, entry-triggered attacks.
- Paper R1-R4 only; ground and boundary semantics are explicitly separated.
- A 100 s timeout is a Red mission failure, not a draw.

See `docs/environment_v2_spec.md` for the normative formulas, indices, update
order, provenance and validation criteria.

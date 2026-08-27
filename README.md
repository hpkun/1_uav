# MAPPO UAV Combat Benchmark

This repository implements MAPPO against public, lightweight 3D UAV combat
environments. The current primary environment is **persistent_wave_v2**, a
three-wave mission wrapper over **Paper-Constrained Direct 4v4 Combat
Environment V2.3** with a ground-aware nearest-target Blue policy. Direct V2.3
remains the base single-round environment, and `persistent_wave_v1` is retained
as a historical variant.

The benchmark is a direct 4v4 engagement: four learned Red agents share an actor
network and fight four Blue aircraft using a deterministic Blue policy. The
observation and action dimensions remain 52 and 3. MAPPO uses a shared
two-layer local actor and a centralized two-head attention value critic with an
on-policy PPO/GAE update.

## Install and test

Install the package in editable mode in a Python environment containing the
dependencies declared by `pyproject.toml`, then run:

```bash
pytest -q
python scripts/validate_combat_environment.py
```

The validation report covers the controller grid, Eq. (8) Monte Carlo behavior,
1000 randomized initializations and a 100-episode rule-based combat baseline.

## Train Current Mainline

```bash
python -u scripts/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --total-sampled-steps 8000000 \
  --num-envs 24 \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/mappo_persistent_wave_v2
```

Training uses one persistent spawned worker per environment. Console output stays
compact; `training_metrics.jsonl`, evaluation history, summaries and checkpoints
retain the complete diagnostic record.

To run the base Direct V2.3 environment explicitly:

```bash
python -u scripts/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --total-sampled-steps 500000 \
  --num-envs 24 \
  --env-config configs/combat_environment.yaml \
  --algorithm-config configs/mappo.yaml \
  --output-dir outputs/mappo_direct_v2_3
```

Its formal hyperparameters are declared in `configs/mappo.yaml`. MAPPO
checkpoints contain the actor, centralized critic, both optimizers and training
counters; no partial on-policy rollout is serialized. MAPPO implementation v2
stores both latent and squashed rollout actions, evaluates PPO ratios from the
stored latent action, and uses Monte-Carlo squashed-policy entropy with an exact
tanh Jacobian.

V2.0-V2.2 checkpoints are intentionally incompatible with V2.3. Checkpoints
contain an `environment_version` field, and resume fails before loading model
weights when the field is absent or different. MAPPO checkpoints also contain
`mappo_impl_version=2`; older MAPPO checkpoints are loadable only through the
explicit diagnostic path and cannot resume formal training. MAPPO saves
`best_eval.pt`, selected lexicographically by win rate, team return, then lower
Red loss for Direct V2.3; persistent missions use the persistent mission
selection key.

## Environment contract

- NED 3DOF point-mass dynamics, RK4, `dt=0.1 s`.
- Relative action increments `[a_psi, a_theta, a_v]` with physical maxima
  `[pi rad, pi/3 rad, 50 m/s]`.
- Eq. (2) inverse controller with 2 s response constants and proportional
  `nz <= 8` projection.
- Random 8 km diameter initialization inside a hard 5 km arena.
- A 4 km, 30-degree true 3-D off-boresight fire cone and velocity-frame Eq. (8)
  probabilistic, entry-triggered attacks.
- Paper R1-R4 only; ground and boundary semantics are explicitly separated.
- A 100 s timeout is a Red mission failure, not a draw.

See `docs/environment_v2_spec.md` for the normative formulas, indices, update
order, provenance and validation criteria.

## Persistent-Wave Variants

`PersistentWaveCombatEnv` keeps the frozen V2.3 observation, reward, dynamics,
weapon, and Red-side transition contract. When a non-final Blue formation is
eliminated, a fresh four-aircraft Blue wave is spawned immediately while Red
physical states and losses persist. Both sides receive fresh Boolean FireState
entry triggers at a wave boundary. The default configuration contains three
waves and does not add ammunition or new observation features.

The active persistent configuration is `configs/persistent_wave_v2_environment.yaml`.
Its MAPPO configuration is `configs/mappo_persistent_wave.yaml`.

Run a MAPPO smoke test with the current mainline using:

```bash
python scripts/train_mappo.py \
  --smoke \
  --device cpu \
  --num-envs 1 \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/mappo_persistent_wave_v2_smoke
```

Persistent algorithm configs use `gamma=0.999`; the original Direct configs
remain at `gamma=0.99`. Checkpoint resume also requires the environment variant
to match.

The implementation scope and deferred extensions are documented in
`docs/persistent_wave_environment_design.md`.

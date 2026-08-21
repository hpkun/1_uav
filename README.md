# MADSAC public 4v4 combat benchmark

This project combines the existing Multi-Agent Double Soft Actor-Critic
implementation with Enhanced Combat Environment V2, a lightweight,
paper-consistent 4v4 3D continuous-action environment. It is an academic
low-fidelity benchmark, not an engineering flight simulator. The complete active
definition is in [docs/environment_v2_spec.md](docs/environment_v2_spec.md).

The active runtime path contains one environment:

- `MultiUAVCombatEnv`, configured by `configs/combat_environment.yaml`
- six-state 3DOF dynamics with high-level heading/pitch/speed maneuver commands
- bounded 52-dimensional translation- and rotation-invariant local observations
- deterministic range/off-boresight/target-aspect firing windows and dwell
- nearest-target pursuit Blue policy through the same simple response layer
- shared MADSAC actor, double attention critics, replay, targets, entropy, and
  the existing Algorithm-1 update scheduler
- one persistent spawn subprocess per training environment; policy inference,
  replay, and optimization remain in the CUDA-capable main process

Run in the requested environment:

```bash
conda activate brmamappo
cd C:\Users\HPK\Desktop\1_uav

pytest -q
python scripts/validate_combat_environment.py
python scripts/check_parallel_env.py --num-envs 24 --steps 100
python scripts/train_madsac.py --smoke --num-envs 24 \
  --total-sampled-steps 24000 --seed 0
python scripts/evaluate_madsac.py \
  --checkpoint outputs/madsac/run_seed_0/latest.pt
```

Validation includes straight/head-on and maneuver/combat scripted baselines.
Combat diagnostics report Red and Blue attackable, completed-lock, and kill rates
separately. `--smoke` intentionally uses reduced hidden, batch, and replay sizes.

Historical Li et al. reconstruction notes are isolated under
`docs/archive/li2023/` and are not part of the active environment contract.
Training checkpoints intentionally omit replay contents, so resumed runs restart
with a fresh replay buffer and fresh episodes.

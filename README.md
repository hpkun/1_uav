# MADSAC public 4v4 combat benchmark

This project combines the existing Multi-Agent Double Soft Actor-Critic
implementation with an independent, lightweight, fully parameterized 4v4 3D
continuous-action environment. It is an academic low-fidelity benchmark, not an
exact reproduction of any published simulator. The complete active environment
definition is in [docs/environment_spec.md](docs/environment_spec.md).

The active runtime path contains one environment:

- `MultiUAVCombatEnv`, configured by `configs/combat_environment.yaml`
- six-state 3DOF point-mass dynamics with direct `[nx,nz,phi]` actions
- full-state 52-dimensional translation- and rotation-invariant local observations
- deterministic 45°/90° attack envelope, three-step locks, and simultaneous kills
- boundary-aware nearest-target pursuit Blue policy
- shared MADSAC actor, double attention critics, replay, targets, entropy, and
  the existing Algorithm-1 update scheduler

Run in the requested environment:

```bash
wsl -d Ubuntu
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/1_uav

pytest -q
python scripts/validate_combat_environment.py
python scripts/train_madsac.py --smoke --num-envs 24 \
  --total-sampled-steps 24000 --seed 0
python scripts/evaluate_madsac.py \
  --checkpoint outputs/madsac/run_seed_0/latest.pt
```

Historical Li et al. reconstruction notes are isolated under
`docs/archive/li2023/` and are not part of the active environment contract.
Training checkpoints intentionally omit replay contents, so resumed runs restart
with a fresh replay buffer and fresh episodes.

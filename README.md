# MADSAC Multi-UAV Cooperative Air Combat Paper Reproduction

This repository reproduces Li et al. (2023), *Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent Double-Soft Actor-Critic*, as a clean 4-red-vs-4-blue homogeneous UAV experiment.

It is a **paper-specification reproduction with explicitly documented assumptions for unpublished implementation details**, not a claim that the authors' unreleased source code has been recovered.

## Implemented mainline

- 4v4 random-diameter scenario in a 10 km engagement area
- Equations (1)-(2) point-mass dynamics in NED coordinates
- Equation (23) continuous actions: `delta_psi`, `delta_theta`, `delta_v`
- Equations (3)-(5) noisy sensor observations
- Figure 2 / Equation (6) ATA, AA, HA, and HCA geometry
- Equations (7)-(8) launch envelope and probabilistic hit model
- Section 2.5 nearest-alive-target fixed blue policy
- Equation (24) 45-dimensional decentralized observation
- Equation (25) segmented cooperative reward
- Shared squashed-Gaussian actor, independent double attention critics, target networks, replay, delayed updates, entropy, soft updates, and CTDE
- Checkpointing, vector environment interface, evaluation metrics, audit, and smoke training

The exact PAPER-SPECIFIED, PAPER-DERIVED, and PAPER-UNSPECIFIED split is in [docs/reproduction_spec.md](docs/reproduction_spec.md). Assumptions are centralized under `reproduction_assumptions` in [configs/paper_environment.yaml](configs/paper_environment.yaml) and [configs/madsac.yaml](configs/madsac.yaml).

## Verification

```powershell
pytest -q
python scripts/audit_paper_environment.py
python scripts/train_madsac.py --smoke --steps 2048
python scripts/evaluate_madsac.py --checkpoint outputs/madsac_smoke.pt
```

The smoke mode intentionally uses a small replay and batch for correctness checks. Formal defaults remain 24 environments, more than 8 million sampled steps, 20 disjoint evaluation seeds, and five independent runs; this repository does not start that long experiment automatically.

## Future fair baselines

The environment returns a stable multi-agent transition contract so MAAC, MAPPO, MADDPG, MASAC, and SAC can later be implemented against the same scenario, observations, action mapping, reward, seeds, and evaluation criteria. Historical HAPPO and heterogeneous experiment branches have intentionally been removed because they are not baselines in this paper.

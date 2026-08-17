# MADSAC Multi-UAV Cooperative Air Combat Paper Reproduction

Strict 4v4 reproduction of Li et al. (2023), *Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent Double-Soft Actor-Critic*. It is a paper-specification reproduction with documented assumptions for unpublished details, not a claim of recovering the authors' source.

## Implemented

- Equations (1)-(8), (23)-(25), `dt=0.1 s`, 4v4 random-diameter scenario, noisy formal sensors, probabilistic weapon model, and nearest-target fixed blue policy.
- Auditable per-aircraft `NONE/ATTACK/BOUNDARY` death ledger. A team wins when every opponent is dead and it retains a survivor; mutual elimination is a draw and timeout is not a red win. Attack kills never include boundary deaths.
- Pre-attack frozen geometry for R3/R41/R42 and simultaneous fire proposals; post-event R1/R2 is then added.
- Symmetric red/blue 45D observations and `env.step(red_actions, blue_actions=None)`. `None` retains the paper fixed-blue training protocol; supplied actions support future learned-vs-learned evaluation and Figure 17-style self-play.
- Shared squashed-Gaussian actor, independent double attention critics, target networks, mask-correct replay/loss/entropy/bootstrap, delayed updates, CTDE, checkpoint/resume, and isolated deterministic evaluation.
- The formal runner actually creates 24 environments. `sampled_env_steps` counts individual transitions; `vector_steps` counts synchronous batches. Update credit is defined per new transition, so update/data ratio is invariant to environment count.

See [the three-level specification](docs/reproduction_spec.md), [environment configuration](configs/paper_environment.yaml), and [MADSAC configuration](configs/madsac.yaml).

## Verification and smoke runs

```powershell
pytest -q
python scripts/audit_paper_environment.py
python scripts/audit_environment_statistics.py --smoke
python scripts/train_madsac.py --smoke --num-envs 1 --total-env-steps 2048
python scripts/train_madsac.py --smoke --num-envs 24 --total-env-steps 24000 --run-id 0
python scripts/evaluate_madsac.py --checkpoint outputs/madsac/run_0_seed_0/latest_evaluation.pt
```

Formal defaults remain 24 training environments, more than 8 million transitions, 20 disjoint evaluation seeds, five independent runs, and 95% confidence intervals. No formal long training starts automatically.

Smoke mode explicitly reduces batch/learning-starts and uses `0.05` updates per transition to bound validation time; these are printed runtime overrides and do not change the formal YAML defaults.

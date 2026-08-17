# MADSAC paper reproduction

Strict 4v4 reproduction target for Li et al. (2023), *Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent Double-Soft Actor-Critic*. The 2023 paper is normative; carefully bounded evidence from the same authors' 2022 predecessor and every unresolved value are listed in [the parameter provenance table](docs/parameter_provenance.md) and [the reproduction evidence table](docs/reproduction_spec.md).

Implemented scope is only MADSAC Red versus the paper's fixed nearest-target Blue strategy. MAAC, MAPPO, MADDPG, MASAC, SAC, learned-vs-learned APIs, and formal 8M training are intentionally out of this refactor.

Core protocol:

- Eq.(1)-(8), Table 1, Fig.4, Eq.(23)-(25)
- shared 2x256 actor, two independent two-head attention critics, target networks, min double-Q and entropy
- Algorithm 1 `T += M` scheduler with 24 synchronous training environments
- binary paper success: all Blue UAVs destroyed
- 20 disjoint evaluation seeds, five run seeds, and Figure 8/9 CSV aggregation with 95% CI

Run all commands in the requested WSL environment:

```bash
wsl -d Ubuntu
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/1_uav

pytest -q
python scripts/audit_paper_environment.py --weapon-samples 10000
python scripts/train_madsac.py --smoke --num-envs 24 --total-sampled-steps 24000 --seed 0
python scripts/evaluate_madsac.py --checkpoint outputs/madsac/run_seed_0/latest.pt
```

Run one candidate-only sensitivity profile at a time (the helper rejects more
than 200,000 sampled steps):

```bash
python scripts/run_reconstruction_sensitivity.py \
  --group weapon --profile weapon_weak --sampled-steps 24000 --seed 0
```

Available groups are `weapon`, `sensor`, `controller`, and `scheduler`. Profiles
are overlays from `configs/sensitivity_candidates.yaml`; they never modify the
canonical YAML files and are not paper values.

For five completed runs:

```bash
python scripts/aggregate_training_runs.py RUN0 RUN1 RUN2 RUN3 RUN4 --output-dir outputs/aggregate
```

Checkpoints contain actor, critics, target networks, optimizers, and counters. Replay is deliberately not saved, so resumed training does not preserve replay continuity. Formal defaults remain >8M sampled steps, but no formal training starts automatically.

# Environment Learnability Checklist

This checklist is for the fixed homogeneous 3v3 time-aware V2 environment. It separates environment reachability, basic MAPPO learnability, and formal strong-opponent experiments. Do not treat one failed short run as proof that the environment is impossible to learn.

## Stage 0: deterministic environment checks

Stage 0 does not run learning. It verifies that the environment and MAPPO interface are controllable and finite.

- `head_on_learnability_v1` must use `homogeneous_3v3_v2_timeaware`, `fixed_id_body_time_63d`, `full_entity_time_61d`, and `project_3v3_v2`.
- Local observations must be `(3, 63)` and global state must be `(61,)`.
- `episode_progress` must reset to `-1` in normalized observations/state and reach `+1` at the 400-step timeout.
- Basic action tests must show left/right, accelerate/decelerate, and climb/dive effects with finite outputs and distinguishable Actor observations.
- A fixed-seed rule-pursuit red team against straight blue in `head_on_learnability_v1` must produce red attack attempts, hits, and effective damage through the real environment step path.
- Four parallel workers must reset, step, report `(4, 3, 63)` and `(4, 61)` tensors, and close cleanly.

`scripts/diagnose_3v3_reward_ordering.py` is kept as an optional fixed-policy environment and reward-ordering helper. It is not the current development mainline and should not be expanded into checkpoint ranking, historical trend analysis, or new statistical machinery.

## Stage 1: single-seed basic learnability

Use only:

```bash
python scripts/train_mappo.py \
  --config configs/mappo_learnability_3v3.yaml \
  --run-name learnability_3v3_seed1
```

This configuration uses one training seed, four parallel workers, `head_on_learnability_v1`, and blue `StraightOpponent`. It is capped at 50k environment steps and is a basic learnability gate, not a formal paper experiment.

Stage 1 passes only if training or deterministic validation shows evidence that return improvement is coupled to combat behavior:

- red attack attempts remain nonzero rather than appearing as one-off noise;
- red hits or effective damage do not stay at zero;
- attack-area occupancy appears during training or validation;
- return improvement is accompanied by attack, hit, or damage improvement;
- improvement is not merely a rising timeout rate or survivor-count timeout outcome.

Stage 1 does not require formal win rate against the strong pursuit opponent.

## Stage 2: formal strong-opponent environment

Only after Stage 1 passes should experiments return to:

```text
head_on_mirrored_jitter_v2
opponent: pursuit
```

Then it is reasonable to consider a second training seed, longer runs, checkpoint comparisons, multi-seed statistics, and paper-style reporting. Before Stage 1 passes, do not run 300k training, multi-seed training, or bulk checkpoint comparisons.

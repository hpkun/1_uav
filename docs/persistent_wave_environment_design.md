# Persistent-Wave / Multi-Round environment audit

This document describes the implemented `persistent_wave_v1` variant. It is a
minimal wrapper around the frozen V2.3 direct-combat environment, not the
previous V3 redesign proposal.

## Frozen contracts

- Four homogeneous learned Red UAVs fight four Blue UAV slots.
- Actor observation remains 52 dimensions and action remains 3 dimensions.
- V2.3 point-mass dynamics, controller, probabilistic entry-triggered weapon,
  simultaneous-hit resolution, and R1-R4 rewards are unchanged.
- Blue uses only `NearestTargetPursuitPolicy`.
- There is no ammunition count, cooldown, Hungarian assignment, communication
  model, inter-wave countdown, explicit missile, or extra boundary reward.

## Mission configuration

`configs/persistent_wave_environment.yaml` selects
`environment_variant: persistent_wave_v1`. The default mission contains three
waves and has one global horizon of 3,000 physics steps (`dt=0.1`, 300 seconds).
Every replacement Blue wave has four live aircraft. A candidate formation is
centered at radius 4,400 m, must keep every Blue aircraft inside the 5,000 m
arena, keep every live Red/Blue three-dimensional distance at least 2,500 m,
and contain no immediate fire-window pair. Rejection sampling is bounded at
256 attempts and deliberately has no fallback.

## Exact boundary order

1. `MultiUAVCombatEnv.step` advances both teams, applies boundary/ground loss,
   snapshots post-motion state, computes R1-R4, evaluates both entry-triggered
   attacks, resolves simultaneous hits, and creates the old-wave reward/info.
2. If Red survives and every Blue slot is dead, the wrapper closes an immutable
   per-wave metric record and increments `waves_cleared`.
3. On a non-final wave before the global time limit, it samples a fresh Blue
   formation, resets both Red and Blue `FireState` objects, and zeros only the
   new Blue wave's last executed bank values.
4. Red physical state, live/dead slots, and last executed bank values persist.
   `wave_index` increments, the returned observation is rebuilt from the new
   Blue wave, and both terminal flags are false.
5. The clearing-step reward is returned unchanged. Final-wave clearance uses
   the original V2.3 success terminal. Global timeout prevents replacement.

The returned `red_alive_mask`, `blue_alive_mask`, survivor counts, loss counts,
observation, and wave fields are all rebuilt or overwritten after replacement
so they describe the same post-transition state.

## Construction and checkpoint contract

All formal vector training and deterministic evaluation create environments via
`make_combat_environment`. Each subprocess reports its actual class and variant
at startup for auditability. Training checkpoints store both
`environment_version` and `environment_variant`; Direct and Persistent
checkpoints are rejected on resume in either direction. Explicit diagnostic
weight loading remains possible in standalone evaluation tools.

The Direct algorithm files remain unchanged at `gamma: 0.99`. Persistent runs
must select `configs/mappo_persistent_wave.yaml` or
`configs/madsac_persistent_wave.yaml`, which use `gamma: 0.999`. With 0.1-second
steps, preserving a conventional per-second discount gives
`0.99 ** 0.1 = 0.998995...`, hence the rounded value.

Example:

```powershell
python scripts/train_mappo.py --device cuda --seed 2023 --num-envs 24 `
  --total-sampled-steps 8000000 `
  --env-config configs/persistent_wave_environment.yaml `
  --algorithm-config configs/mappo_persistent_wave.yaml `
  --output-dir outputs/mappo_persistent_wave
```

## Per-wave and mission metrics

Every completed wave records start/end/duration steps, Red survivors at both
ends, Blue survivors, each side's attempts/hits/combat kills/boundary exits/
ground losses, R1-R4 totals, and team return. Evaluation and final training
summaries add mean waves cleared, unconditional probability of clearing each
wave, conditional mean Red survivors after each cleared wave, total Blue/Red
losses, and a loss-denominator-clipped kill/loss ratio.

`scripts/validate_persistent_wave_environment.py` performs the pure-environment
10,000-case replacement stress audit across 1-4 Red survivors and center, edge,
dispersed, and varied-altitude/heading layouts. It reports failures without
changing parameters or introducing a fallback, and advances each successful
case for one second with zero Red action and the fixed Blue policy.

## Read-only Markov audit

The 52-dimensional actor observation does not contain `wave_index` or remaining
wave count. For a fixed physical 4v4 state, within-wave dynamics, Blue policy,
weapon behavior, and reward are independent of the index. The hidden variable
matters only when a wave is cleared: the same observed clearance state either
spawns another wave or terminates on the final wave. Therefore the actor-facing
process is formally non-Markov at mission-boundary/value-estimation level, even
though local flight and engagement control remain Markov with respect to the
unchanged physical observation. A feed-forward critic cannot distinguish these
two continuation values. This is a real experimental limitation to disclose,
but resolving it would require the explicitly forbidden 53D/mission-context
observation change and is outside `persistent_wave_v1`.

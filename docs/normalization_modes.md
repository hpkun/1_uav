# Normalization modes

`paper_linear` applies `a*value/reference-b` without clipping by default. Yaw is first mapped to `[0,2*pi)` and uses reference `2*pi`: `pi -> 0` and `7*pi/4 -> 0.75` for the default `a=2,b=1`.

`symmetric_training` preserves signed yaw in `[-pi,pi]` and uses a separate `pi` reference: `-pi, pi/2, pi -> -1, 0.5, 1`. Small opposite yaw angles remain opposites. Actor, 1v1 Critic, 2v2 local observations, and 2v2 global state use the same rule.

Signed features use `value/reference`; nonnegative features use `2*value/reference-1`; action indices map linearly to `[-1,1]`. Saturation is measured before optional clipping. Dead/padded entity blocks remain zero and are excluded from saturation counts.

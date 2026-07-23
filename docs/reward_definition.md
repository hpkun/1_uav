# Reward definition

## 1v1

The `zheng_series_1v1` profile keeps the named angle, distance, height, speed, dense, event, damage, boundary, and terminal components inspectable. Debug-linear alternatives are diagnostic only.

## 2024 multi-UAV terminal reward

These equations are transcribed from equations (21)--(25) on PDF pages 10--11 of the 2024 paper in the repository root. Let `n_r` be the number of red UAVs, `N_step` the decision limit, `n_step` the terminal decision step, `alive_count=|Omega'_r|`, `beta_i` cumulative contribution, `beta=sum_i beta_i`, `B_i` terminal health, `B_sum=sum_i B_i` over surviving red UAVs, and `B0` initial health.

```text
r_win_all = r_win0*n_r*(0.75 + 0.25*(N_step-n_step)/N_step)                (21)
r_end_i = r_win_all*(w_win1/n_r + 0.03*alive_count
                     + w_win2*beta_i/beta
                     + w_win3*(B_i/B_sum)*(B_i/B0))                       (22)
r_lose_all = r_lose0*n_r*(0.80 + 0.20*(N_step-n_step)/N_step)             (23)
beta'_i = max_k(beta_k)-beta_i+1; B'_i = B0-B_i+10                        (24)
r_end_i = r_lose_all*(w_lose1/n_r - 0.02*alive_count
                      + w_lose2*beta'_i/max_k(beta'_k)
                      + w_lose3*B'_i/B0)                                  (25)
```

The paper does not publish numerical values for win/lose weights or draw reward; they remain project configuration. The code follows equation (25)'s `w_lose3` despite a nearby prose typo. `paper_2024_exact` checks only three finite, nonnegative weights and does not renormalize them. `project_balanced` is a project-defined proportional ablation and requires weights to sum to one.

Project numerical conventions, not paper equations: on a win with `beta=0`, every contribution share is `1/n_r`; with `B_sum=0`, every health component is zero. A draw gives every red UAV the configured `draw_reward`. `base.yaml` safely defaults to `project_balanced`; all formal 2v2 configs explicitly select a profile.

`TerminalRewardAllocation` and environment info expose team base, allocation factor, base share, shared survival term, contribution, health, alive count, and both denominators. The allocation factor is their direct sum and reward equals `team_base*allocation_factor`.

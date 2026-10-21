# Recorded trajectory audit: trajectory.json

- Source: `C:\Users\HPK\Desktop\1_uav\outputs\coop3d_v2\best_recording\trajectory.json`
- SHA256: `42cb18f55a4b4c057ffdb61e6c0dfc29f471ec8c78b61c0729478c80bad2865e`
- Environment: `coop3d-v2.0.0`; seed=30000001; scale=3U3P.
- Provenance: `{"checkpoint": "C:\\Users\\HPK\\Desktop\\1_uav\\outputs\\coop3d_v2\\smoke\\best.pt", "checkpoint_role": "best", "update": 1, "sampled_steps": 16}`
- Steps=8; dt=0.25s; elapsed=2s; success=False; completion=0.
- State/previous-state max error=0 (8 checked transitions); repeated waypoint events=0.
- Geometry/lowest-active-owner event mismatches=0; completed-flag mismatches=0.
- Reward checks=24 UAV-transitions; max absolute residuals against current inspection rules: `{'progress': 1.3877787807814457e-17, 'completion': 0.0, 'separation': 0.0, 'efficiency': 0.0, 'terminal': 0.0, 'sum': 0.0}`.
- Reward residuals test current code rules, not either combat paper. No input was simulated or changed.
- Recorded physical controls are executed controls. Inverse mapping is NOT proof of a raw Beta sample; idle controls override policy actions.
- Raw policy action/Beta alpha/Beta beta/log_probability/entropy are absent in this recording. No distribution inference is made.
- Saturation means executed normalized control <=0.01 or >=0.99 (1% of action range); a state at a limit is distinct from a clipping event.
- Heading increments use wrap(delta psi); total turn sums absolute increments, net turn sums signed increments. Units below are degrees.

## UAV 0

- Active transitions=8; assigned=8; idle=0; path length=456.871m.
- Position start=[34.832, 474.856, 3029.686]; end=[443.52, 270.767, 3022.363].
- Speed min/mean/max=228.303 / 228.436 / 228.57 m/s; gamma=-0.92891 / -0.918354 / -0.907697 deg.
- Heading start/end=-26.522/-26.551; net/total turn=-0.029/0.029 deg.
- Assigned pre-action heading error min/mean/max=17.8931 / 20.6654 / 23.4928 deg; mean absolute=20.6654 deg.
- Same initial target P0: distance start/end/min=10239.982/9808.655/9808.655m; heading error start/end=17.893/18.746 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|-0.00596384 / -0.00592126 / -0.0058792|0.499051 / 0.499058 / 0.499064|0.000|
|tangential load|-0.00246242 / -0.0024485 / -0.00243479|0.332513 / 0.332517 / 0.332522|0.000|
|normal load|0.99553 / 0.995578 / 0.995626|0.124441 / 0.124447 / 0.124453|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 8 active transitions.
- speed_clips: 0 events / 8 active transitions.
- nonfinite_states: 0 events / 8 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|10239.982 / 9808.655 / 9808.655|False|
|P1|9827.482 / 9409.094 / 9409.094|False|
|P2|7864.297 / 7490.255 / 7490.255|False|

|Reward component|Sum over this UAV|
|---|---|
|progress|0.84939757|
|completion|0|
|separation|0|
|efficiency|-0.08|
|terminal|0|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|0 -> 0|10239.982m / 17.893deg|[85.898, 449.3688, 3028.7799, 228.336, -0.9104, -26.5255]|[-0.005964, -0.002462, 0.99553]|0.09803018209799848|[] []|
|4|0 -> 1|10078.019m / 18.204deg|[239.1297, 372.8668, 3026.0464, 228.4354, -0.9184, -26.5365]|[-0.005939, -0.002454, 0.99555]|0.09786355039739283|[] [{'uav_id': 1, 'reason': 'injected_failure'}]|
|8|1 -> 1|9461.219m / 23.493deg|[443.5199, 270.7669, 3022.3629, 228.5696, -0.9289, -26.5509]|[-0.005879, -0.002435, 0.995626]|0.0942492309504778|[] []|

## UAV 1

- Active transitions=4; assigned=4; idle=0; path length=222.045m.
- Position start=[122.307, 57.424, 3306.75]; end=[322.628, 152.354, 3293.955].
- Speed min/mean/max=221.775 / 222.165 / 222.316 m/s; gamma=-3.30669 / -3.30482 / -3.29996 deg.
- Heading start/end=25.363/25.348; net/total turn=-0.015/0.015 deg.
- Assigned pre-action heading error min/mean/max=-27.3626 / -27.1416 / -26.9222 deg; mean absolute=27.1416 deg.
- Same initial target P1: distance start/end/min=9702.810/9506.556/9506.556m; heading error start/end=-26.922/-27.513 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|-0.0059805 / -0.00596538 / -0.00595017|0.499048 / 0.499051 / 0.499053|0.000|
|tangential load|-0.00253233 / -0.00252755 / -0.00252277|0.332489 / 0.332491 / 0.332492|0.000|
|normal load|0.995689 / 0.995697 / 0.995705|0.124461 / 0.124462 / 0.124463|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 4 active transitions.
- speed_clips: 0 events / 4 active transitions.
- nonfinite_states: 0 events / 4 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|10078.920 / 9892.156 / 9892.156|False|
|P1|9702.810 / 9506.556 / 9506.556|False|
|P2|7866.288 / 7651.963 / 7651.963|False|

|Reward component|Sum over this UAV|
|---|---|
|progress|0.392507329|
|completion|0|
|separation|0|
|efficiency|-0.04|
|terminal|0|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|1 -> 1|9702.810m / -26.922deg|[172.3391, 81.1396, 3303.5562, 221.9102, -3.3016, 25.3595]|[-0.005981, -0.002532, 0.995689]|0.08824222043979171|[] []|
|4|1 -> -1|9555.561m / -27.363deg|[322.6281, 152.3537, 3293.9549, 222.3156, -3.3067, 25.3482]|[-0.00595, -0.002523, 0.995705]|0.0880091138978969|[] [{'uav_id': 1, 'reason': 'injected_failure'}]|
|8|-1 -> -1|idle|[322.6281, 152.3537, 3293.9549, 222.3156, -3.3067, 25.3482]|[0.0, 0.0, 0.0]|0.0|[] []|

## UAV 2

- Active transitions=8; assigned=8; idle=0; path length=447.361m.
- Position start=[87.083, 200.758, 2770.476]; end=[516.759, 324.237, 2786.672].
- Speed min/mean/max=223.302 / 223.68 / 224.06 m/s; gamma=2.06617 / 2.07481 / 2.08348 deg.
- Heading start/end=16.048/16.019; net/total turn=-0.028/0.028 deg.
- Assigned pre-action heading error min/mean/max=-6.87184 / -6.71033 / -6.55352 deg; mean absolute=6.71033 deg.
- Same initial target P2: distance start/end/min=7840.406/7397.239/7397.239m; heading error start/end=-6.554/-6.920 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|-0.005647 / -0.00560511 / -0.00556386|0.499101 / 0.499108 / 0.499114|0.000|
|tangential load|-0.00240171 / -0.00238838 / -0.00237525|0.332533 / 0.332537 / 0.332542|0.000|
|normal load|0.995893 / 0.995916 / 0.995938|0.124487 / 0.124489 / 0.124492|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 8 active transitions.
- speed_clips: 0 events / 8 active transitions.
- nonfinite_states: 0 events / 8 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|10176.762 / 9768.164 / 9768.164|False|
|P1|9783.537 / 9361.096 / 9361.096|False|
|P2|7840.406 / 7397.239 / 7397.239|False|

|Reward component|Sum over this UAV|
|---|---|
|progress|0.886335208|
|completion|0|
|separation|0|
|efficiency|-0.08|
|terminal|0|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|2 -> 2|7840.406m / -6.554deg|[140.8682, 216.227, 2772.5105, 223.9646, 2.0813, 16.044]|[-0.005647, -0.002402, 0.995893]|0.1010049187120967|[] []|
|4|2 -> 2|7673.989m / -6.686deg|[302.0947, 262.5759, 2778.5975, 223.6802, 2.0748, 16.0334]|[-0.005611, -0.00239, 0.995913]|0.1008230541736848|[] [{'uav_id': 1, 'reason': 'injected_failure'}]|
|8|2 -> 2|7452.527m / -6.872deg|[516.7594, 324.2372, 2786.672, 223.3024, 2.0662, 16.0195]|[-0.005564, -0.002375, 0.995938]|0.10057717830954789|[] []|

## Limits of inference

A persistent turn or large flight-path angle is a measured behavior, not proof of its learning cause. Different seeds, episode limits, failures or model versions cannot establish a controlled policy comparison. A recorded pose at the gamma/speed bound does not establish what an unconstrained integrator would have done. Missing fields remain missing.

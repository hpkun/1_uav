# Recorded trajectory audit: trajectory.json

- Source: `C:\Users\HPK\Desktop\1_uav\outputs\coop3d_v2\checkpoint_evaluation\trajectory.json`
- SHA256: `73217bc9a1bc45167ea0f93eae048e558245e1b4f04a8c64eb13e7d98bc1af95`
- Environment: `coop3d-v2.0.0`; seed=30000000; scale=3U3P.
- Provenance: `{"checkpoint": "C:\\Users\\HPK\\Desktop\\1_uav\\outputs\\coop3d_v2\\smoke\\latest.pt", "checkpoint_role": "latest", "sampled_steps": 25, "update": 3}`
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

- Active transitions=8; assigned=8; idle=0; path length=462.885m.
- Position start=[426.771, -434.071, 2917.617]; end=[858.888, -270.512, 2889.582].
- Speed min/mean/max=230.889 / 231.443 / 231.999 m/s; gamma=-3.47821 / -3.47217 / -3.466 deg.
- Heading start/end=20.744/20.721; net/total turn=-0.023/0.023 deg.
- Assigned pre-action heading error min/mean/max=-30.3466 / -18.6619 / -7.22289 deg; mean absolute=18.6619 deg.
- Same initial target P0: distance start/end/min=10382.326/9926.101/9926.101m; heading error start/end=-7.223/-7.535 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|-0.00506764 / -0.00474351 / -0.004419|0.499193 / 0.499245 / 0.499297|0.000|
|tangential load|-0.00413835 / -0.00398683 / -0.00383517|0.331954 / 0.332004 / 0.332055|0.000|
|normal load|0.995602 / 0.995661 / 0.995721|0.12445 / 0.124458 / 0.124465|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 8 active transitions.
- speed_clips: 0 events / 8 active transitions.
- nonfinite_states: 0 events / 8 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|10382.326 / 9926.101 / 9926.101|False|
|P1|8476.472 / 8074.496 / 8074.496|False|
|P2|6770.398 / 6325.753 / 6325.753|False|

|Reward component|Sum over this UAV|
|---|---|
|progress|0.856715499|
|completion|0|
|separation|0|
|efficiency|-0.08|
|terminal|0|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|0 -> 0|10382.326m / -7.223deg|[480.669, -413.6595, 2914.1251, 231.0268, -3.4676, 20.7408]|[-0.005068, -0.004138, 0.995602]|0.10388047445841948|[] []|
|4|0 -> 1|10211.429m / -7.336deg|[642.5629, -352.368, 2903.6286, 231.4418, -3.4722, 20.7316]|[-0.005037, -0.004122, 0.995616]|0.10403175167092923|[] [{'uav_id': 1, 'reason': 'injected_failure'}]|
|8|1 -> 1|8124.498m / -30.347deg|[858.888, -270.5123, 2889.5824, 231.9987, -3.4782, 20.7209]|[-0.004419, -0.003835, 0.995721]|0.09000513005430184|[] []|

## UAV 1

- Active transitions=4; assigned=4; idle=0; path length=242.092m.
- Position start=[-142.372, 293.161, 3209.789]; end=[74.485, 186.946, 3192.476].
- Speed min/mean/max=241.76 / 242.239 / 242.423 m/s; gamma=-4.10311 / -4.10188 / -4.09868 deg.
- Heading start/end=-26.090/-26.100; net/total turn=-0.010/0.010 deg.
- Assigned pre-action heading error min/mean/max=13.8453 / 13.9872 / 14.1304 deg; mean absolute=13.9872 deg.
- Same initial target P1: distance start/end/min=9193.271/8958.604/8958.604m; heading error start/end=13.845/14.228 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|-0.0044821 / -0.00447007 / -0.00445813|0.499287 / 0.499289 / 0.49929|0.000|
|tangential load|-0.00386047 / -0.00385388 / -0.00384733|0.332047 / 0.332049 / 0.332051|0.000|
|normal load|0.995531 / 0.995539 / 0.995548|0.124441 / 0.124442 / 0.124444|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 4 active transitions.
- speed_clips: 0 events / 4 active transitions.
- nonfinite_states: 0 events / 4 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|10788.512 / 10592.366 / 10592.366|False|
|P1|9193.271 / 8958.604 / 8958.604|False|
|P2|7340.453 / 7122.153 / 7122.153|False|

|Reward component|Sum over this UAV|
|---|---|
|progress|0.46933287|
|completion|0|
|separation|0|
|efficiency|-0.04|
|terminal|0|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|1 -> 1|9193.271m / 13.845deg|[-88.2119, 266.6378, 3205.4672, 241.9256, -4.0998, -26.0927]|[-0.004482, -0.00386, 0.995531]|0.10728699791222243|[] []|
|4|1 -> -1|9017.293m / 14.130deg|[74.4846, 186.9455, 3192.4765, 242.4235, -4.1031, -26.1005]|[-0.004458, -0.003847, 0.995548]|0.10737839716130111|[] [{'uav_id': 1, 'reason': 'injected_failure'}]|
|8|-1 -> -1|idle|[74.4846, 186.9455, 3192.4765, 242.4235, -4.1031, -26.1005]|[0.0, 0.0, 0.0]|0.0|[] []|

## UAV 2

- Active transitions=8; assigned=8; idle=0; path length=450.262m.
- Position start=[125.055, 383.926, 2716.615]; end=[574.46, 382.214, 2744.323].
- Speed min/mean/max=224.492 / 225.131 / 225.771 m/s; gamma=3.52258 / 3.52806 / 3.53355 deg.
- Heading start/end=-0.208/-0.229; net/total turn=-0.021/0.021 deg.
- Assigned pre-action heading error min/mean/max=-1.56772 / -1.5321 / -1.49775 deg; mean absolute=1.5321 deg.
- Same initial target P2: distance start/end/min=7032.626/6585.823/6585.823m; heading error start/end=-1.498/-1.578 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|-0.00424466 / -0.00420784 / -0.00417032|0.499324 / 0.49933 / 0.499336|0.000|
|tangential load|-0.00367907 / -0.00365957 / -0.00363964|0.332107 / 0.332113 / 0.33212|0.000|
|normal load|0.995897 / 0.995917 / 0.995937|0.124487 / 0.12449 / 0.124492|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 8 active transitions.
- speed_clips: 0 events / 8 active transitions.
- nonfinite_states: 0 events / 8 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|10531.396 / 10086.922 / 10086.922|False|
|P1|8907.966 / 8473.589 / 8473.589|False|
|P2|7032.626 / 6585.823 / 6585.823|False|

|Reward component|Sum over this UAV|
|---|---|
|progress|0.893605017|
|completion|0|
|separation|0|
|efficiency|-0.08|
|terminal|0|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|2 -> 2|7032.626m / -1.498deg|[181.3698, 383.7204, 2720.0917, 225.6106, 3.5322, -0.2104]|[-0.004245, -0.003679, 0.995897]|0.1020255836210963|[] []|
|4|2 -> 2|6864.726m / -1.527deg|[350.0758, 383.0892, 2730.4993, 225.1306, 3.5281, -0.2183]|[-0.004211, -0.003661, 0.995916]|0.10174786404492807|[] [{'uav_id': 1, 'reason': 'injected_failure'}]|
|8|2 -> 2|6641.510m / -1.568deg|[574.4596, 382.2138, 2744.3228, 224.4916, 3.5226, -0.2288]|[-0.00417, -0.00364, 0.995937]|0.10137374762865511|[] []|

## Limits of inference

A persistent turn or large flight-path angle is a measured behavior, not proof of its learning cause. Different seeds, episode limits, failures or model versions cannot establish a controlled policy comparison. A recorded pose at the gamma/speed bound does not establish what an unconstrained integrator would have done. Missing fields remain missing.

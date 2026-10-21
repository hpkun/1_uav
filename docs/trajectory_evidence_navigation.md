# Recorded trajectory audit: trajectory.json

- Source: `C:\Users\HPK\Desktop\1_uav\outputs\coop3d_v2\navigation_recording\trajectory.json`
- SHA256: `2943b94303ce0b7166c623cac609ac936d892a52859d24521ece0d303d83e8da`
- Environment: `coop3d-v2.0.0`; seed=104; scale=3U5P.
- Provenance: `{"policy": "navigation", "deterministic": false, "scenario": "standard"}`
- Steps=179; dt=0.25s; elapsed=44.75s; success=True; completion=1.
- State/previous-state max error=0 (179 checked transitions); repeated waypoint events=0.
- Geometry/lowest-active-owner event mismatches=0; completed-flag mismatches=0.
- Reward checks=537 UAV-transitions; max absolute residuals against current inspection rules: `{'progress': 1.3877787807814457e-17, 'completion': 0.0, 'separation': 0.0, 'efficiency': 0.0, 'terminal': 0.0, 'sum': 0.0}`.
- Reward residuals test current code rules, not either combat paper. No input was simulated or changed.
- Recorded physical controls are executed controls. Inverse mapping is NOT proof of a raw Beta sample; idle controls override policy actions.
- Raw policy action/Beta alpha/Beta beta/log_probability/entropy are absent in this recording. No distribution inference is made.
- Saturation means executed normalized control <=0.01 or >=0.99 (1% of action range); a state at a limit is distinct from a clipping event.
- Heading increments use wrap(delta psi); total turn sums absolute increments, net turn sums signed increments. Units below are degrees.

## UAV 0

- Active transitions=179; assigned=179; idle=0; path length=10058.830m.
- Position start=[-16.069, -133.882, 2765.349]; end=[9818.999, 1058.481, 2169.693].
- Speed min/mean/max=223.101 / 224.775 / 225.051 m/s; gamma=-7.1147 / -3.38503 / 3.69509 deg.
- Heading start/end=34.235/-8.303; net/total turn=-42.538/42.538 deg.
- Assigned pre-action heading error min/mean/max=-23.5451 / -2.998 / -5.2026e-09 deg; mean absolute=2.998 deg.
- Same initial target P4: distance start/end/min=7719.742/2284.509/21.346m; heading error start/end=-23.545/-177.728 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|-1.34893 / -0.327078 / -1.66606e-09|0.285312 / 0.447944 / 0.5|0.000|
|tangential load|-0.124129 / -0.0547998 / 0.0967033|0.291957 / 0.315067 / 0.365568|0.000|
|normal load|0.969631 / 1.12531 / 1.66658|0.121204 / 0.140664 / 0.208323|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 179 active transitions.
- speed_clips: 0 events / 179 active transitions.
- nonfinite_states: 0 events / 179 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|10212.814 / 298.552 / 298.552|True|
|P1|9389.775 / 1048.171 / 861.004|True|
|P2|6749.388 / 3685.191 / 1824.042|True|
|P3|6309.150 / 3948.895 / 1254.957|True|
|P4|7719.742 / 2284.509 / 21.346|True|

|Reward component|Sum over this UAV|
|---|---|
|progress|19.9938108|
|completion|20|
|separation|0|
|efficiency|-1.79|
|terminal|13|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|4 -> 4|7719.742m / -23.545deg|[30.1989, -102.9018, 2768.7355, 223.1897, 3.2653, 33.3762]|[-1.34301, 0.096703, 1.397805]|0.0919787459046911|[] []|
|25|4 -> 4|6428.061m / -5.648deg|[1259.0798, 420.151, 2751.9399, 224.4142, -2.6064, 12.7522]|[-0.970373, -0.034419, 1.661653]|0.10173034149598244|[] []|
|50|4 -> 4|5024.877m / -0.031deg|[2645.0703, 631.5957, 2683.941, 224.8001, -2.8297, 7.7284]|[-0.009821, -0.045809, 0.997475]|0.10239785254477647|[] []|
|75|4 -> 4|3619.430m / -0.000deg|[4036.1521, 819.8542, 2614.4601, 224.9311, -2.8349, 7.7023]|[-5.4e-05, -0.048235, 0.998743]|0.10246478441307864|[] []|
|100|4 -> 4|2213.456m / -0.000deg|[5427.747, 1008.0587, 2544.9211, 224.9762, -2.835, 7.7021]|[-0.0, -0.049038, 0.998775]|0.1024878486713269|[{'step': 100, 'waypoint_id': 3, 'uav_id': 1}] []|
|118|4 -> 4|1201.034m / -0.000deg|[6429.8104, 1143.581, 2494.8468, 224.9889, -2.835, 7.7021]|[-0.0, -0.049264, 0.998776]|0.10249435166730837|[{'step': 118, 'waypoint_id': 2, 'uav_id': 2}] []|
|125|4 -> 4|807.302m / -0.000deg|[6819.514, 1196.2858, 2475.3728, 224.9918, -2.835, 7.7021]|[-0.0, -0.049314, 0.998776]|0.10249580689128812|[] []|
|134|4 -> 0|301.068m / -0.000deg|[7320.5675, 1264.0499, 2450.3345, 224.9944, -2.835, 7.7021]|[-0.0, -0.049361, 0.998776]|10.102497141190756|[{'step': 134, 'waypoint_id': 4, 'uav_id': 0}] []|
|150|0 -> 0|1986.127m / -2.808deg|[8214.5455, 1277.5122, 2367.5922, 225.0501, -6.5668, -5.5397]|[-0.795273, -0.113953, 1.251676]|0.10240990657831799|[] []|
|158|0 -> 0|1536.208m / -0.629deg|[8658.2959, 1224.2718, 2314.5667, 225.0417, -6.9085, -7.6454]|[-0.207312, -0.120557, 0.971385]|0.10251502532064842|[{'step': 158, 'waypoint_id': 1, 'uav_id': 1}] []|
|175|0 -> 0|579.829m / -0.042deg|[9598.0394, 1090.6769, 2197.53, 225.0229, -7.0972, -8.2762]|[-0.01362, -0.123874, 0.984759]|0.10251160879176974|[] []|
|179|0 -> -1|354.807m / -0.030deg|[9818.9988, 1058.4809, 2169.6933, 225.0196, -7.1147, -8.303]|[-0.009646, -0.124129, 0.98553]|23.102509975915297|[{'step': 179, 'waypoint_id': 0, 'uav_id': 0}] []|

## UAV 1

- Active transitions=179; assigned=158; idle=21; path length=10134.152m.
- Position start=[298.33, 60.157, 3062.179]; end=[10246.034, 1049.745, 2877.613].
- Speed min/mean/max=225.15 / 226.484 / 235.701 m/s; gamma=-18.5871 / -1.16974 / 5.89605 deg.
- Heading start/end=4.780/6.619; net/total turn=1.839/1.839 deg.
- Assigned pre-action heading error min/mean/max=1.77831e-09 / 0.0581978 / 1.43693 deg; mean absolute=0.0581978 deg.
- Same initial target P3: distance start/end/min=5943.506/4155.938/24.882m; heading error start/end=0.259/179.990 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|0 / 0.0394917 / 0.981561|0.5 / 0.506285 / 0.65622|0.000|
|tangential load|-0.318746 / -0.0427306 / 0.10017|0.227085 / 0.31909 / 0.366723|0.000|
|normal load|0.268264 / 0.802026 / 1.72192|0.033533 / 0.100253 / 0.215241|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 179 active transitions.
- speed_clips: 0 events / 179 active transitions.
- nonfinite_states: 0 events / 179 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|9903.897 / 757.761 / 757.761|True|
|P1|9051.434 / 966.233 / 112.807|True|
|P2|6418.971 / 3874.685 / 991.519|True|
|P3|5943.506 / 4155.938 / 24.882|True|
|P4|7395.428 / 2730.095 / 1283.220|True|

|Reward component|Sum over this UAV|
|---|---|
|progress|17.7484768|
|completion|20|
|separation|0|
|efficiency|-1.79|
|terminal|13|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|3 -> 3|5943.506m / 0.259deg|[356.9184, 65.0837, 3065.1223, 235.2459, 3.0812, 4.8324]|[0.050513, -0.135553, 1.721924]|0.10759261574535049|[] []|
|25|3 -> 3|4553.760m / 0.002deg|[1734.3034, 185.8037, 3189.2733, 228.6609, 5.8183, 5.0512]|[0.000526, 0.036254, 1.014988]|0.10437022969235113|[] []|
|50|3 -> 3|3132.817m / 0.000deg|[3141.6838, 310.2301, 3334.6618, 226.2625, 5.8939, 5.0526]|[3e-06, 0.080299, 0.995227]|0.10314498997226657|[] []|
|75|3 -> 3|1721.498m / 0.000deg|[4539.8738, 433.8507, 3479.6014, 225.4356, 5.896, 5.0527]|[0.0, 0.094999, 0.994727]|0.10272255926799517|[] []|
|100|3 -> 1|313.499m / 0.000deg|[5934.9101, 557.1925, 3624.2285, 225.1503, 5.896, 5.0527]|[0.0, 0.100059, 0.994712]|10.102576799409082|[{'step': 100, 'waypoint_id': 3, 'uav_id': 1}] []|
|118|1 -> 1|2507.318m / 0.047deg|[6940.8336, 666.4205, 3660.0451, 225.1879, -1.8404, 6.5753]|[0.048322, -0.027788, 0.311417]|0.10013110954231888|[{'step': 118, 'waypoint_id': 2, 'uav_id': 2}] []|
|125|1 -> 1|2121.213m / 0.012deg|[7331.5982, 711.5986, 3637.0599, 225.1962, -4.8475, 6.6076]|[0.01243, -0.080349, 0.308382]|0.10054282794684605|[] []|
|134|1 -> 1|1622.964m / 0.002deg|[7831.3084, 769.5371, 3577.25, 225.2035, -8.7126, 6.617]|[0.002329, -0.147512, 0.300875]|0.10090620091673191|[{'step': 134, 'waypoint_id': 4, 'uav_id': 0}] []|
|150|1 -> 1|735.797m / 0.000deg|[8705.5696, 870.9796, 3387.8136, 225.2093, -15.5803, 6.6192]|[0.00018, -0.26492, 0.276527]|0.10038732820053542|[] []|
|158|1 -> -1|300.866m / 0.000deg|[9132.8669, 920.5655, 3254.4686, 225.2013, -18.5871, 6.6194]|[4.9e-05, -0.318106, 0.549736]|10.092514724109698|[{'step': 158, 'waypoint_id': 1, 'uav_id': 1}] []|
|175|-1 -> -1|idle|[10034.0025, 1025.1394, 2949.3949, 225.2013, -18.5871, 6.6194]|[0.0, -0.318746, 0.94784]|-0.01|[] []|
|179|-1 -> -1|idle|[10246.0344, 1049.745, 2877.6128, 225.2013, -18.5871, 6.6194]|[0.0, -0.318746, 0.94784]|12.99|[{'step': 179, 'waypoint_id': 0, 'uav_id': 0}] []|

## UAV 2

- Active transitions=179; assigned=134; idle=45; path length=10164.556m.
- Position start=[-234.104, 197.955, 2560.459]; end=[9789.898, 155.954, 3640.323].
- Speed min/mean/max=225.104 / 227.175 / 241.233 m/s; gamma=-1.21564 / 6.0866 / 8.86853 deg.
- Heading start/end=-15.105/9.948; net/total turn=25.053/25.053 deg.
- Assigned pre-action heading error min/mean/max=3.11954e-09 / 2.97564 / 23.7353 deg; mean absolute=2.97564 deg.
- Same initial target P2: distance start/end/min=7008.978/3144.787/13.939m; heading error start/end=10.592/179.289 deg.

|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|
|---|---|---|---|
|bank mu (rad)|0 / 0.175169 / 1.35398|0.5 / 0.527879 / 0.715492|0.000|
|tangential load|-0.296998 / 0.0698245 / 0.152395|0.234334 / 0.356608 / 0.384132|0.000|
|normal load|0.988046 / 1.15603 / 2.27921|0.123506 / 0.144503 / 0.284901|0.000|

- Executed bank with cos(mu)<0: 0.000%.
- gamma_clips: 0 events / 179 active transitions.
- speed_clips: 0 events / 179 active transitions.
- nonfinite_states: 0 events / 179 active transitions.

|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|
|---|---|---|
|P0|10387.282 / 1765.238 / 1765.238|True|
|P1|9583.698 / 1080.309 / 1018.936|True|
|P2|7008.978 / 3144.787 / 13.939|True|
|P3|6526.895 / 3625.002 / 920.786|True|
|P4|7874.987 / 2776.008 / 1886.033|True|

|Reward component|Sum over this UAV|
|---|---|
|progress|14.9498497|
|completion|10|
|separation|0|
|efficiency|-1.79|
|terminal|13|

Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.

|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|
|---|---|---|---|---|---|---|
|1|2 -> 2|7008.978m / 10.592deg|[-175.855, 182.7009, 2559.4073, 240.547, -0.7854, -14.2443]|[0.703903, -0.296998, 2.279206]|0.10712624743158268|[] []|
|25|2 -> 2|5609.049m / 0.141deg|[1214.6857, 15.4659, 2661.371, 230.4862, 8.0418, -3.9178]|[0.037926, 0.040262, 1.211896]|0.10529120992486969|[] []|
|50|2 -> 2|4180.848m / 0.001deg|[2622.6029, -78.555, 2876.0768, 226.8851, 8.8484, -3.7982]|[0.000238, 0.120345, 0.993218]|0.10346306914135858|[] []|
|75|2 -> 2|2767.031m / 0.000deg|[4016.1599, -171.0591, 3093.8604, 225.6503, 8.8679, -3.7976]|[1e-06, 0.142627, 0.988183]|0.10283222730909347|[] []|
|100|2 -> 2|1358.170m / 0.000deg|[5405.0144, -263.2481, 3311.0403, 225.2244, 8.8685, -3.7976]|[0.0, 0.150189, 0.98805]|0.1026146406027101|[{'step': 100, 'waypoint_id': 3, 'uav_id': 1}] []|
|118|2 -> 0|344.934m / 0.000deg|[6403.9087, -329.5523, 3467.2436, 225.1043, 8.8685, -3.7976]|[0.0, 0.152318, 0.988046]|10.102553288770498|[{'step': 118, 'waypoint_id': 2, 'uav_id': 2}] []|
|125|0 -> 0|3887.479m / 20.485deg|[6794.3557, -334.9198, 3517.7282, 225.1339, 5.8584, 2.2155]|[1.351177, 0.107321, 1.402342]|0.08383092105161268|[] []|
|134|0 -> -1|3458.262m / 15.374deg|[7296.4247, -281.398, 3552.3866, 225.1613, 1.9895, 9.9484]|[1.348808, 0.039514, 1.410502]|0.08712806750849814|[{'step': 134, 'waypoint_id': 4, 'uav_id': 0}] []|
|150|-1 -> -1|idle|[8182.9929, -125.895, 3583.653, 225.1613, 1.9895, 9.9484]|[0.0, 0.034715, 0.999397]|-0.01|[] []|
|158|-1 -> -1|idle|[8626.2771, -48.1436, 3599.2862, 225.1613, 1.9895, 9.9484]|[0.0, 0.034715, 0.999397]|-0.01|[{'step': 158, 'waypoint_id': 1, 'uav_id': 1}] []|
|175|-1 -> -1|idle|[9568.2559, 117.0783, 3632.5067, 225.1613, 1.9895, 9.9484]|[0.0, 0.034715, 0.999397]|-0.01|[] []|
|179|-1 -> -1|idle|[9789.898, 155.9541, 3640.3233, 225.1613, 1.9895, 9.9484]|[0.0, 0.034715, 0.999397]|12.99|[{'step': 179, 'waypoint_id': 0, 'uav_id': 0}] []|

## Limits of inference

A persistent turn or large flight-path angle is a measured behavior, not proof of its learning cause. Different seeds, episode limits, failures or model versions cannot establish a controlled policy comparison. A recorded pose at the gamma/speed bound does not establish what an unconstrained integrator would have done. Missing fields remain missing.

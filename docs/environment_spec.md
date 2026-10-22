# Coop3D v2.0 环境规范

## 身份与边界

唯一版本 `coop3d-v2.0.0`，唯一环境类 `env.cooperative_env.Cooperative3DEnv`。算法与工具通过 `env.factory.make_environment()` 构造环境；`env/` 不导入 algorithm、tools 或 torch。3U3P、5U5P、3U5P 只改变配置规模。原 v1 环境任务规则保持不变，12 个迁移前场景的 873 个转移以测试 fixture 独立保存并精确回归。

`reset(seed=None, options=None) -> observations, info`；`step(actions) -> observations, rewards, terminated, truncated, info`。N 架 UAV 的观测 `[N,12]`、奖励 `[N]`、掩码 `[N]`、状态 `[N,6]`，点坐标 `[M,3]`。info 中 `centralized_state` 固定 `[90]`，仅供 Critic。

## 动力学与任务规则（未改变）

状态 `[x,y,z,v,γ,ψ]`，右手系 z 向上，单位 m、m/s、rad。有效性独立维护。控制 `[μ,ζ,ζ̂]`：

```text
dx=v*cosγ*cosψ; dy=v*cosγ*sinψ; dz=v*sinγ
dv=g*(ζ-sinγ)
dγ=g/v*(ζ̂*cosμ-cosγ)
dψ=g*ζ̂*sinμ/(v*cosγ)
```

历史兼容实现采用经典 RK4、g=9.81、dt=.25 s、速度100–400 m/s、γ限幅±85°、ψ环绕[-π,π)。±85°是无论文依据的额外约束（C类），不是必要数值防护或论文飞行包线。导数使用epsilon=1e-6替代分母，中间阶段也裁剪γ；这会改变原方程，不能冒称严格复现。非有限输入报错，非有限积分保留前个有限状态、记录并使 UAV 失效。没有直接增量修改速度或姿态的运动模型。独立 `env/paper_dynamics.py` 仅核验原式，不做γ/速度/分母投影、不连接本环境，见 [逐公式审计](xu2026_strict_audit.md)。

动作 `[0,1]^3` 映射 `[2πa0−π,3a1−1,8a2]`。有效动作须有限合法，失效动作忽略。无任务有效 UAV 用 `[0,sinγ,cosγ]` 配平巡航，仍积分同一动力学。

未完成点与有效 UAV 每步独占 Hungarian：`d/10000+.2*|wrap(atan2(dy,dx)−ψ)|/π−.1*p`。优先级默认1，合法[0,1]。任务多于 UAV 时点等待，反之 UAV 闲置。nearest 对全部距离对排序后贪心独占，同距按 UAV/点 ID，作为非学习分配对照。

点以不可变数组索引为 ID。步末三维球半径≤300m即完成，无停留状态、无轨迹穿球检测、不在reset直接完成。任一有效 UAV 的实际到达可完成点，不要求是它分配的点；同点同一步最低有效 UAV ID 唯一记账。完成只改变任务进度，不移除 UAV。不同重叠任务球可分别完成。

step 顺序：存当前状态、mask、分配 → 积分 → 故障注入 → 下一有效 UAV 的完成判定 → 五项奖励和终止 → 重分配 → 下一观测。故障当步保留转移、进度和时间项，但不参与步末到达/安全；后续冻结，观测清零，分配排除。

五项 reward：

|项|公式|
|---|---|
|progress|`2*(d_before−d_after)/1000`，动作前同一个分配点，无任务为0|
|completion|每个首次唯一完成事件10分|
|separation|`−mean(max(0,1−distance/100))`，只对步末有效友机，单架为0|
|efficiency|动作前有效 UAV 每步−.01|
|terminal|仅成功当步，步末有效 UAV 获得 `20−floor(T/25)`|

仅这五项，没有额外奖励。纯函数 `env.reward.compute_rewards` 可从记录重算；总 reward 返回float32，分项保留float64。全部点完成则terminated/success；全部UAV失效且任务未完则terminated/all_inactive；达到250步尚未结束则truncated。真正终止优先于同一步超时。结束后必须reset。

## Actor 12D（未改变）

|索引|内容/公式|
|---|---|
|0|active 0/1|
|1|`(v-v_min)/(v_max-v_min)`|
|2|`γ/gamma_limit`|
|3|`wrap(ψ)/π`|
|4|`z/altitude_scale`，默认5000m|
|5–7|`(assigned_point-position)/distance_scale`，默认10000m|
|8|三维目标距离/distance_scale|
|9|`wrap(atan2(dy,dx)−ψ)/π`|
|10|是否有有效分配|
|11|最近有效友机距离/distance_scale，无友机为0|

无任务5–10项填0；失效整行0。归一化后不隐藏裁剪，远距离、高度可超过1。

## Critic 90D（明确的 v2 接口调整）

前60维：按ID顺序的最多5×12D局部观测，不存在UAV槽位补零。后30维：最多5×6D航路点记录：

`[x/distance_scale,y/distance_scale,z/altitude_scale,priority,completed,exists]`。

点按ID排序，完成点仍保留坐标和exists=1，不存在槽位六项全零。因此3U5P尚未分配的两个点也能进入Critic；Actor仍只看到原12D观测。三种规模的Critic状态均90D。reset提供当前状态，step提供真实后继状态；vector包装额外保存auto-reset状态，绝不拿新一局状态对上一局自举。

## 配置与诊断

统一 `EnvConfig.load(path)`。随机UAV水平±500m、高度2500–3500m、速度200–250m/s、γ±5°、朝向任务均值±30°；点x5500–11000m、y±2000m、z2000–4000m。初始友机三维距离≥200m，最多10000次采样，全部使用环境自己的RNG。

固定场景 `random_scenario=False`；reset options支持states、waypoints、priorities、active_mask。故障配置 `{20:[1]}`，默认不注入随机故障。配置新增严格类型、有限性和故障步号检查，只改变不合法输入的处理，不改变合法场景的运动或任务规则。

保留累计 `numerical_counters`，新增 `numerical_delta` 和逐UAV `numerical_events`。每步delta=累计前后差，事件数组求和等于delta。裁剪按每 UAV 完整积分结果计一次；统计分母为动作前真正有效 UAV-step，不能以累计计数最大值当本次事件数。

本基准没有地形、能耗、传感噪声或碰撞损伤模型；100m参考是奖励约束，不是严格避碰保证。

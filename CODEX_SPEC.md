请在当前固定同构3v3环境V2基础上，完成一次训练前的统一修正。

本轮不是重新设计环境，也不是继续扩展功能，而是修复当前V2代码审查确认的六类问题：

1. Actor和Critic缺少有限时域的回合进度信息；
2. 升级前保存的legacy checkpoint v3可能无法继续完整恢复；
3. 规则诊断和训练日志没有完整区分红蓝双方；
4. timeout终局奖励的profile标签仍错误显示为paper_2024_exact；
5. 死亡实体的归一化值和saturation统计不规范；
6. `_body_relative_kinematics`返回值类型标注错误。

请一次性完成这些修改，不要拆成多轮小补丁。

本轮完成后，正式固定同构3v3环境仍称为V2，但schema必须升级为time-aware版本。旧45D/87D legacy环境继续保留。之前仅用于4096步smoke的62D/60D checkpoint不需要自动迁移到新63D/61D网络，但必须产生明确的schema不兼容错误。

一、严格禁止修改的部分

不得修改：

- 三自由度质点动力学；
- RK4积分；
- 15种离散动作及其过载；
- physics_dt=0.1；
- decision_dt=0.5；
- physics_steps_per_action=5；
- max_decision_steps=400；
- max_episode_seconds=200；
- 攻击距离40至900米；
- 攻击角30度；
- 逃逸角60度；
- 优势区和攻击区阈值；
- 毁伤概率阈值；
- 毁伤值51、21、11、0；
- 初始血量300；
- PursuitOpponent动作选择逻辑；
- 最近存活目标分配逻辑；
- 镜像扰动范围；
- V2奖励拆分公式；
- 命中、击毁、被命中、被击毁、边界和碰撞事件值；
- timeout_reward=-4.0；
- elimination时的paper_2024_exact终局公式；
- MAPPO Actor和Critic隐藏层；
- PPO、GAE、学习率、clip、batch等训练超参数；
- 共享前馈Actor；
- 集中式Critic；
- 固定同构3v3规模。

不实现：

- 1v1迁移；
- 3v2；
- 异构；
- 注意力；
- GRU；
- 自博弈；
- 雷达；
- 连续动作；
- 课程学习；
- 新规则对手。

二、将正式V2 schema升级为time-aware版本

更新正式V2场景配置中的schema字段：

```yaml
environment_schema_version: homogeneous_3v3_v2_timeaware
observation_schema: fixed_id_body_time_63d
global_state_schema: full_entity_time_61d
reward_profile: project_3v3_v2
scenario_profile: head_on_mirrored_jitter_v2
```

完全对称压力测试配置应使用：

```yaml
environment_schema_version: homogeneous_3v3_v2_timeaware
observation_schema: fixed_id_body_time_63d
global_state_schema: full_entity_time_61d
reward_profile: project_3v3_v2
scenario_profile: symmetric_stress_test
```

同步更新：

* 正式V2训练配置；
* V2 smoke配置；
* V2场景配置；
* 完全对称压力测试配置；
* 文档；
* checkpoint metadata；
* 审计；
* 测试。

环境判断V2时不要只写：

```python
environment_schema_version == "homogeneous_3v3_v2"
```

应明确支持新的正式schema。建议实现为：

```python
self.is_v2 = self.environment_schema_version in {
    "homogeneous_3v3_v2",
    "homogeneous_3v3_v2_timeaware",
}
self.is_timeaware_v2 = (
    self.environment_schema_version == "homogeneous_3v3_v2_timeaware"
)
```

正式配置只能使用time-aware版本。

旧62D/60D schema可以继续被代码识别用于明确报错或旧smoke复核，但不要把它作为正式训练配置。

三、Actor局部观测由62D升级为63D

当前V2局部观测为：

* own block：7维；
* ally blocks：2×8维；
* enemy blocks：3×13维；
* 总计62维。

在own block最后增加一个：

```text
episode_progress
```

新的own block固定为8维，顺序必须为：

1. own_altitude
2. own_speed
3. own_flight_path_angle
4. own_heading_sin
5. own_heading_cos
6. own_health_ratio
7. own_last_action
8. episode_progress

新的总维度：

```text
8 + 2×8 + 3×13 = 63
```

不要改变友机和敌机块的现有字段、顺序和固定ID语义。

四、episode_progress的精确定义

在环境中统一计算：

```python
episode_progress = min(
    max(self.decision_step / float(self.config["max_decision_steps"]), 0.0),
    1.0,
)
```

语义：

* reset后decision_step=0，raw progress=0；
* 第200步，raw progress=0.5；
* 第400步，raw progress=1；
* 截断终端状态必须为1；
* 提前歼灭时保留当前真实进度，不强制设为1。

Actor和Critic必须使用完全相同的episode_progress。

归一化为：

```python
normalized_progress = 2.0 * episode_progress - 1.0
```

因此：

* 回合开始为-1；
* 回合中点为0；
* 超时终点为+1。

不得使用simulation_time和decision_step两个重复时间量，只加入一个episode_progress。

五、修改局部观测构建接口

修改：

```text
src/uav_env/observations/multi_observation.py
```

要求：

1. `V2_OWN_FEATURES`增加`episode_progress`；
2. `multi_observation_feature_names_v2()`返回63个名称；
3. `multi_observation_feature_names_v2_for_agent()`返回63个名称；
4. `build_multi_observations_v2()`增加显式参数：

```python
episode_progress: float
```

不要让该函数自行读取隐藏的环境对象。

建议签名：

```python
def build_multi_observations_v2(
    red_aircraft: Sequence[UAV],
    blue_aircraft: Sequence[UAV],
    config: dict[str, object],
    attack_config: object,
    episode_progress: float,
) -> MultiObservationResult:
```

5. own block末尾写入raw episode_progress；
6. `_normalize_v2()`增加：

```python
elif name == "episode_progress":
    transformed = 2.0 * np.clip(value, 0.0, 1.0) - 1.0
```

7. 所有写死的62改为63；
8. saturation ratio除以63；
9. dead red Actor row仍可保持全零，因为Actor会被alive mask屏蔽；
10. 修正所有测试中的shape。

六、Critic全局状态由60D升级为61D

当前60D由6架飞机×10维组成。

保持每架飞机10维实体块完全不变，在所有六架实体块之后追加一个全局字段：

```text
episode_progress
```

顺序必须是：

```text
red_0 10D
red_1 10D
red_2 10D
blue_0 10D
blue_1 10D
blue_2 10D
episode_progress 1D
```

总维度：

```text
6×10+1=61
```

修改：

```text
src/uav_env/observations/global_state.py
```

要求：

1. `global_state_feature_names_v2()`最后追加`episode_progress`；
2. `build_global_state_v2()`增加显式参数：

```python
episode_progress: float
```

3. raw state最后追加progress；
4. `_normalize_v2_global()`支持progress映射到[-1,1]；
5. saturation ratio除以61；
6. 不把progress放进某一架飞机实体块；
7. 不改变Critic的agent identity拼接方式。

建议签名：

```python
def build_global_state_v2(
    red_aircraft: Sequence[UAV],
    blue_aircraft: Sequence[UAV],
    config: dict[str, object],
    episode_progress: float,
) -> GlobalStateResult:
```

七、环境中统一传递episode_progress

修改：

```text
src/uav_env/envs/combat_multi_env.py
```

增加一个内部方法：

```python
def _episode_progress(self) -> float:
    return float(
        np.clip(
            self.decision_step / float(self.config["max_decision_steps"]),
            0.0,
            1.0,
        )
    )
```

然后：

```python
def _observations(self):
```

V2调用：

```python
build_multi_observations_v2(
    self.red_aircraft,
    self.blue_aircraft,
    self.config,
    self.attack_config,
    self._episode_progress(),
)
```

`_global_state()`同样传入该值。

正式time-aware V2：

```python
local_observation_dim = 63
global_state_dim = 61
```

旧legacy仍为45D/87D。

若保留旧62D/60D schema，则必须显式区分，禁止根据team size自动猜测。

八、增加有限时域Markov测试

必须新增测试，构造完全相同的飞机物理状态，但修改环境decision_step。

至少验证：

1. decision_step=0与decision_step=399：

   * Actor局部观测不同；
   * Critic全局状态不同；
   * 唯一变化字段是episode_progress。

2. reset时：

   * raw progress=0；
   * normalized progress=-1。

3. decision_step=200时：

   * raw progress=0.5；
   * normalized progress=0。

4. decision_step=400时：

   * raw progress=1；
   * normalized progress=1。

5. 相同飞机物理状态、相同动作、不同剩余时间时，Critic状态不能相同。

6. 终端截断状态传给GAE的global state必须包含progress=1，而不是auto-reset后的-1。

不要只测试feature name和shape，要检查具体数值。

九、修复死亡实体的归一化语义

当前raw死亡槽位已经使用：

```text
alive_flag=-1
health_ratio=0
其他物理量=0
```

但通用归一化会把：

* health_ratio=0映射为-1；
* distance=0映射为-1；
* attack_angle=0映射为-1；
* last_action=0映射为-1；
* speed=0映射到小于-1后clip；

导致所谓"死亡槽位其余字段置零"在normalized observation中不成立，并制造虚假的saturation。

需要统一规定：

局部观测中每个死亡友机或敌机块的normalized值必须是：

```text
alive_flag = -1
其余字段 = 0
```

全局状态中每个死亡飞机实体块的normalized值必须是：

```text
alive_flag = -1
其余9个字段 = 0
```

raw值可以继续保留现有规范，但normalized输出必须显式覆盖。

十、局部死亡槽位的具体修改

在`build_multi_observations_v2()`中：

1. 正常完成raw拼接和初始归一化；
2. 根据固定ID实体是否存活，显式覆盖normalized块；
3. 同时把对应saturated mask整块设为False。

固定索引必须根据结构计算，不要散落魔法数字。

新63D结构：

```text
own block: [0:8]

ally_0: [8:16]
ally_1: [16:24]

blue_0: [24:37]
blue_1: [37:50]
blue_2: [50:63]
```

建议通过常量计算：

```python
OWN_SIZE = len(V2_OWN_FEATURES)
ALLY_SIZE = len(V2_ALLY_FEATURES)
ENEMY_SIZE = len(V2_ENEMY_FEATURES)
```

对于死亡友机：

```python
normalized[start] = -1.0
normalized[start + 1:end] = 0.0
saturated[start:end] = False
```

对于死亡敌机同样处理。

不要把死亡实体的health归一化为-1；alive_flag已经表示死亡，其余字段统一0。

十一、全局死亡槽位的具体修改

在`build_global_state_v2()`或专用辅助函数中，对每个死亡飞机实体块执行：

```python
normalized[start] = -1.0
normalized[start + 1:start + 10] = 0.0
saturated[start:start + 10] = False
```

六个实体块索引：

```text
red_0: [0:10]
red_1: [10:20]
red_2: [20:30]
blue_0: [30:40]
blue_1: [40:50]
blue_2: [50:60]
episode_progress: [60]
```

episode_progress不能被死亡槽位覆盖。

增加测试：

* 死亡飞机raw位置分别为0和9999时，normalized实体块完全相同；
* 死亡飞机raw速度为0和150时，normalized实体块完全相同；
* 死亡飞机不会增加saturation_count；
* alive_flag=-1，其余字段均0；
* progress字段保持正确。

十二、修复legacy checkpoint v3完整恢复

修改：

```text
src/uav_env/algorithms/mappo/checkpoint.py
```

当前问题：

升级前的checkpoint v3没有`schema_metadata`。当前回退逻辑调用：

```python
schema_metadata(data.get("config", {}))
```

会得到：

```text
obs_dim=None
state_dim=None
num_agents=None
```

随后与legacy Runner实际expected metadata中的45、87、3比较，导致本来兼容的旧checkpoint被拒绝。

请按以下规则修改。

规则A：checkpoint显式包含schema_metadata

保持严格比较：

* environment_schema_version；
* observation_schema；
* global_state_schema；
* reward_profile；
* scenario_profile；
* obs_dim；
* state_dim；
* num_agents。

任意不一致都拒绝。

规则B：checkpoint没有schema_metadata

只允许在以下条件下使用legacy兼容回退：

```python
expected_metadata["environment_schema_version"] == "legacy"
expected_metadata["observation_schema"] == "legacy"
expected_metadata["global_state_schema"] == "legacy"
```

对于这种旧v3 checkpoint：

1. 将缺失schema解释为legacy；
2. 不因为obs_dim/state_dim/num_agents缺失而直接拒绝；
3. 依靠`actor.load_state_dict()`和`critic.load_state_dict()`检查真实网络维度；
4. 如果Actor或Critic维度不匹配，转换为明确的ValueError；
5. optimizer和normalizer继续按旧逻辑恢复；
6. 不允许无metadata的checkpoint恢复到time-aware V2；
7. 无metadata checkpoint恢复到V2时，必须给出明确错误：

```text
legacy checkpoint without schema metadata cannot resume into homogeneous_3v3_v2_timeaware
```

不要通过补零、截断、自动映射来兼容网络。

规则C：actor_only

保持当前actor_only允许加载，但实际维度不匹配时必须明确报错。

增加测试：

1. 构造一个version=3但删除`schema_metadata`的legacy 45D/87D/3-agent checkpoint；
2. 匹配的legacy Runner完整resume成功；
3. Actor、Critic、normalizer、environment_steps和update_index正确恢复；
4. 同一无metadata legacy checkpoint恢复到63D/61D V2时被拒绝；
5. 显式schema metadata的V2 checkpoint仍严格比较；
6. 旧62D/60D smoke checkpoint不能恢复到63D/61D正式V2；
7. 错误信息明确指出不匹配字段或网络维度。

十三、修复Critic load_state_dict错误包装

当前Actor维度错误被包装为ValueError，但Critic直接调用：

```python
critic.load_state_dict(...)
```

请同样捕获RuntimeError并转换：

```python
try:
    critic.load_state_dict(data["critic"])
except RuntimeError as error:
    raise ValueError(
        f"Critic dimensions are incompatible: {error}"
    ) from error
```

optimizer不匹配也可以保留PyTorch原始错误，但Actor和Critic主网络必须有明确错误。

十四、修正timeout terminal profile标签

修改：

```text
src/uav_env/rewards/multi_reward.py
```

当前timeout奖励虽然是项目定义的-4，但`TerminalRewardAllocation.profile`仍使用`paper_2024_exact`。

V2 timeout应返回：

```text
profile = project_3v3_v2_timeout
```

建议使用关键字构造，避免组件字段错位：

```python
TerminalRewardAllocation(
    reward=value,
    profile="project_3v3_v2_timeout",
    team_base=value,
    allocation_factor=1.0,
    base_share_component=0.0,
    survival_component=0.0,
    contribution_component=0.0,
    health_component=0.0,
    alive_count=sum(u.is_alive for u in red_aircraft),
    contribution_denominator=0.0,
    health_denominator=0.0,
)
```

V2 simultaneous elimination应返回：

```text
profile = project_3v3_v2_simultaneous_elimination
reward = 0
```

只有：

* blue_eliminated；
* red_eliminated；

才继续使用：

```text
paper_2024_exact
```

增加测试：

* timeout每个slot=-4；
* timeout profile为`project_3v3_v2_timeout`；
* simultaneous elimination reward=0；
* simultaneous profile正确；
* elimination profile仍为`paper_2024_exact`；
* elimination公式数值与修改前一致。

十五、完整拆分规则诊断中的红蓝指标

修改：

```text
scripts/audit_3v3_environment.py
```

当前单回合已经有部分red/blue指标，但聚合时重新合并。

`run_rule_episode()`必须分别输出：

```text
red_attack_attempts
blue_attack_attempts

red_hits
blue_hits

red_nominal_damage
blue_nominal_damage

red_effective_damage
blue_effective_damage

red_overkill_damage
blue_overkill_damage

red_attack_area_steps
blue_attack_area_steps

red_ground_crashes
blue_ground_crashes

red_ceiling_violations
blue_ceiling_violations

red_collisions
blue_collisions

red_survivors
blue_survivors
```

不要再通过trajectory计算红蓝合计attack attempts。直接从：

```python
statistics["aircraft"][uav_id]
```

按阵营聚合。

环境per-aircraft statistics当前没有collisions字段，因此需要同时修改环境统计。

十六、为每架飞机增加collision统计

在`CombatMultiEnv.reset()`初始化每架飞机统计时增加：

```python
"collisions": 0
```

发生碰撞后，根据`collision_pairs`统计每架飞机参与的碰撞事件数。

例如：

```python
for first, second in collision_pairs:
    self._statistics["aircraft"][first]["collisions"] += 1
    self._statistics["aircraft"][second]["collisions"] += 1
```

全局：

```python
self._statistics["collisions"]
```

继续保留，定义为碰撞pair数量。

避免只根据`collision_ids`加1，因为同一飞机可能同时出现在多个碰撞pair中。

十七、aggregate_rule必须输出红蓝分项

`aggregate_rule()`必须至少返回：

```text
mean_red_attack_attempts
mean_blue_attack_attempts

mean_red_hits
mean_blue_hits

mean_red_nominal_damage
mean_blue_nominal_damage

mean_red_effective_damage
mean_blue_effective_damage

mean_red_overkill_damage
mean_blue_overkill_damage

mean_red_attack_area_steps
mean_blue_attack_area_steps

mean_red_ground_crashes
mean_blue_ground_crashes

mean_red_ceiling_violations
mean_blue_ceiling_violations

mean_red_collisions
mean_blue_collisions

mean_red_survivors
mean_blue_survivors
```

可以额外保留合计字段，但不能只保留合计字段。

同时输出：

```text
red_timeout_survival_wins
blue_timeout_survival_wins
red_elimination_wins
blue_elimination_wins
draws
timeouts
```

V2审计必须明确检查：

* pursuit vs pursuit中红蓝双方是否都出现attack attempts；
* 若只有一方出现，标记warning；
* 若双方都为0，标记warning；
* 不要因此自动修改PursuitOpponent。

十八、训练rollout增加红蓝双方战斗统计

修改：

```text
src/uav_env/algorithms/mappo/runner.py
```

在`collect()`中，为每个完成回合分别统计：

红方：

```text
red_attack_attempts
red_hits
red_nominal_damage
red_effective_damage
red_overkill_damage
red_attack_area_steps
red_ground_crashes
red_ceiling_violations
red_collisions
red_survivors
```

蓝方：

```text
blue_attack_attempts
blue_hits
blue_nominal_damage
blue_effective_damage
blue_overkill_damage
blue_attack_area_steps
blue_ground_crashes
blue_ceiling_violations
blue_collisions
blue_survivors
```

写入metrics.csv时使用明确名称：

```text
rollout_red_attack_attempts_mean
rollout_blue_attack_attempts_mean
rollout_red_hits_mean
rollout_blue_hits_mean
...
```

原有：

```text
effective_damage
hit_count
ground_crash_rate
```

可以为了兼容保留，但正式诊断必须使用红蓝分项。

ground crash rate也应分别增加：

```text
red_ground_crash_rate
blue_ground_crash_rate
```

十九、评估指标补齐collision分项

`evaluate()`已有部分side_metrics。

增加：

```text
red_collisions
blue_collisions
```

并确保所有以下项目均按红蓝分开：

* attack_attempts；
* hits；
* nominal_damage；
* effective_damage；
* overkill_damage；
* attack_area_steps；
* ground_crashes；
* ceiling_violations；
* collisions；
* survivors。

结果键保持：

```text
mean_red_...
mean_blue_...
```

不要更改checkpoint选择的核心combat排序逻辑。

二十、奖励尺度增加四种归一化统计

当前训练日志主要记录逐agent-step均值，还不足以判断奖励尺度。

对以下reward components：

```text
situation_reward
geometry_event_reward
raw_shape_reward
assigned_shape_reward
combat_event_reward
dense_reward
terminal_reward
hit_event_reward
destroy_event_reward
attacked_event_penalty
destroyed_event_penalty
boundary_collision_penalty
```

至少增加：

1. 每个完成回合平均：

```text
<component>_per_episode
```

2. 每个环境决策步平均：

```text
<component>_per_decision_step
```

3. 每个存活红方智能体步平均：

```text
<component>_per_alive_agent_step
```

4. 每架红机每回合平均：

```text
<component>_per_agent_episode
```

定义必须清楚：

```text
per_episode =
    所有完成回合的该component总和 / 完成回合数

per_decision_step =
    该rollout所有环境中该component总和 /
    rollout内实际环境决策步数量

per_alive_agent_step =
    该component总和 /
    rollout内所有红方alive mask为1的agent-step数量

per_agent_episode =
    该component总和 /
    (完成回合数 × num_agents)
```

若rollout中没有完成回合：

* per_episode=0；
* per_agent_episode=0；
* 不得产生NaN。

已有`<component>_mean`可以保留。

二十一、训练奖励统计必须基于真实逐步breakdown

不要只在terminal_steps上读取reward breakdown。

检查当前ParallelCombatVectorEnv返回的数据结构。确保每一个环境step的：

```text
agent_reward_breakdowns
statistics
```

都能被训练日志读取。

如果当前vector结果只在terminal_steps中保留完整info，那么：

* 红蓝回合统计可以在terminal step读取；
* 但逐步reward component统计必须从每一步的step info读取，而不能只统计回合结束步。

必须确认当前实现是否已经统计每个step。

若当前只读取`result["terminal_steps"]`中的breakdown，应修正vector adapter或collect数据读取，使reward component统计覆盖rollout中的所有实际step。

不要改变RolloutBuffer奖励本身，只补齐diagnostics。

这一项必须有测试：

* 构造一个至少3步的受控episode；
* 前两步有非零geometry reward；
* 最后一步terminal；
* metrics中的component累计值必须包含前三步，而不是只包含最后一步。

二十二、修正 `_body_relative_kinematics` 类型标注

当前函数实际返回9个float，但类型标注只声明8个。

修改为：

```python
def _body_relative_kinematics(
    own: UAV,
    target: UAV,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
```

不要改变函数运行逻辑和返回顺序。

二十三、更新V2测试

重点更新：

```text
tests/test_3v3_v2_environment.py
tests/test_checkpoint_roundtrip.py
```

至少增加或修改以下测试：

1. local obs shape为(3,63)；
2. global state shape为(61,)；
3. parallel vector shape为：

   * `(4,3,63)`
   * `(4,61)`；
4. feature name数量准确；
5. episode_progress索引准确；
6. reset progress=-1；
7. 中点progress=0；
8. timeout terminal progress=1；
9. 同物理状态不同decision_step可区分；
10. fixed ID槽位仍不交换；
11. normalized死亡友机块为[-1,0,...,0]；
12. normalized死亡敌机块为[-1,0,...,0]；
13. normalized死亡global实体块为[-1,0,...,0]；
14. 死亡实体不增加saturation_count；
15. timeout profile标签正确；
16. elimination profile和公式不变；
17. 无metadata legacy v3 checkpoint可以恢复legacy环境；
18. 无metadata legacy checkpoint不能恢复time-aware V2；
19. 旧62D/60D checkpoint不能恢复63D/61D；
20. Critic维度错误被包装为ValueError；
21. 红蓝规则聚合字段存在；
22. 红蓝训练rollout字段存在；
23. 多步reward component统计不会漏掉非终端step；
24. collision分项统计正确；
25. 所有结果有限。

不要使用只检查"字段存在"的弱测试。关键数值必须有受控断言。

二十四、更新审计脚本

V2正式审计必须新增以下检查：

1. schema为time-aware版本；
2. 63D Actor观测；
3. 61D Critic状态；
4. progress字段raw和normalized值正确；
5. 同物理状态不同剩余时间可区分；
6. terminal state progress=1；
7. 死亡局部块规范化；
8. 死亡全局块规范化；
9. 死亡块不产生虚假saturation；
10. timeout profile标签正确；
11. elimination仍是paper_2024_exact；
12. legacy无metadata v3 checkpoint兼容逻辑；
13. V2 checkpoint严格schema检查；
14. 规则实验红蓝双方分项；
15. V2 pursuit vs pursuit双方攻击可达性；
16. 训练metrics所需字段；
17. 所有输出无NaN和Inf。

保留legacy 45D/87D的4个warning，不要把它们伪装成pass。

V2审计行只有在所有time-aware检查通过时才能是P0 pass。

二十五、更新文档

更新：

* README；
* 3v3环境审计文档；
* MAPPO设计文档；
* 实验假设；
* 奖励定义；
* 训练协议。

必须写明：

* 正式V2 Actor为63D；
* 正式V2 Critic状态为61D；
* 新增episode_progress解决有限时域状态混叠；
* 旧62D/60D只属于开发阶段smoke schema；
* 旧45D/87D继续作为legacy；
* 死亡实体normalized块使用alive=-1、其余0；
* timeout profile为project_3v3_v2_timeout；
* elimination终局才标记paper_2024_exact；
* checkpoint无metadata回退仅允许legacy；
* 训练和评估均按红蓝分开记录战斗统计；
* 当前time-aware V2尚未通过300k可学习性验证。

二十六、运行要求

完成代码后运行：

```text
pytest
```

运行：

```text
python scripts/audit_3v3_environment.py
```

运行一次从随机初始化开始的4096环境步time-aware V2 MAPPO smoke。

允许运行规则诊断：

* jitter pursuit vs pursuit 100回合；
* jitter straight vs pursuit 100回合；
* jitter random vs pursuit 100回合；
* symmetric stress pursuit vs pursuit 100回合。

不得运行：

* 100k；
* 300k；
* 5M；
* 任何其他长训练。

二十七、最终回复格式

最终只报告：

1. 修改文件；
2. 63D局部观测完整结构；
3. 61D全局状态完整结构；
4. episode_progress定义及归一化；
5. 死亡槽位normalized规范；
6. legacy checkpoint无metadata兼容逻辑；
7. V2 checkpoint严格检查逻辑；
8. timeout profile修正；
9. 红蓝规则统计；
10. 红蓝训练统计；
11. 奖励尺度四种归一化指标；
12. 多步reward component是否完整统计；
13. collision分项统计；
14. pytest结果；
15. audit结果；
16. 四组规则诊断具体数值；
17. 4096步smoke结果；
18. NaN、Inf、shape和worker检查；
19. 明确说明没有运行长训练；
20. 明确说明仍需300k验证实际可学习性。

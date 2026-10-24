# Approach-Gated Posture：受控奖励消融与验收报告

日期：2026-10-23。实验身份：`posture_approach_gated_3v3`，family=`reward_ablation`，`paper_experiment=false`。本轮只实现和验收，不执行正式2500/5000回合训练，不恢复历史模型继续训练。

## 公式与实际数据流

新增配置：

```python
posture_reward_mode = 'paper'       # 默认，或 'approach_gated'
posture_gate_speed = 100.0          # m/s，必须有限且>0
```

`paper`直接使用原始姿态优势P，不对其做乘法、重算、缩放或裁剪。原始`geometry.posture_advantage`、`pair_advantages`及APH矩阵计算完全未修改。

对动作前存活的Red i及动作前分配的有效Blue目标j：

```
d0 = ||p_blue_before[j] - p_red_before[i]||
d1 = ||p_blue_after[j]  - p_red_after[i]||
u_los = (p_blue_before[j] - p_red_before[i]) / d0
c_self = (p_red_after[i] - p_red_before[i]) · u_los / dt
c_rel = (d0-d1) / dt
G = clip(min(c_self,c_rel) / posture_gate_speed, 0, 1)
```

只有`approach_gated`且`d1>tau_max`：`P_new=min(P,0)+G*max(P,0)`；其余`P_new=P`。tau_max继续来自原配置，默认2000m。负值完全保留，没有额外距离/失败惩罚或新权重。

step开始时复制双方真实状态、Blue存活掩码和已有分配；积分、Blue决策、攻击判定照旧；Reward阶段读取这组前后状态，结算同一目标。不会把下一步APH新目标用于本步门控。双方本步阵亡仍使用真实运动端点和动作前有效性结算；此前已阵亡Red或无效/越界/此前已死的目标不参与门控。d0=0时G=0，不做除零；若动作后仍处于近/中距离，原P仍不变。

三个奖励区间严格沿用原式：`d1<tau_min`（默认100m）、`tau_min<=d1<=tau_max`、`d1>tau_max`。100m和2000m边界均不门控。

## 修改模块及隔离

| 文件 | 变化 |
|---|---|
| `env/aircombat/config.py` | 模式、速度参数校验及假设记录 |
| `env/aircombat/posture_reward.py`（新增） | 真实转移门控、纯统计与按样本累计的比例 |
| `env/aircombat/rewards.py` | 仅在结算时替换Posture；paper保留原赋值路径；其他四项原逻辑不变 |
| `env/aircombat/environment.py` | 捕获动作前状态并传给Reward，返回只读姿态诊断 |
| `algorithm/common/aircombat_vector.py` | 逐回合与独立评估累计姿态诊断 |
| `algorithm/mappo/aircombat_rollout.py` | 在既有pending字典中累计诊断，随精确回合checkpoint保存；不改rollout张量/采样顺序 |
| `algorithm/aircombat.py`、`common/aircombat_progress.py` | JSONL、评估和启动日志标记模式及速度 |
| `algorithm/experiments.py` | 新实验注册、种子/预算/smoke复用、独立默认根目录、从论文计划排除 |
| `tools/report_aircombat.py` | 排除实验family/非paper标志/非paper奖励模式，避免进入论文图表 |
| `tests/test_approach_posture.py`（新增） | 48个公式、边界、隔离、CUDA/PPO确定性及恢复用例 |
| `tools/validate_approach_posture.py`（新增） | 有界10worker CUDA验收和历史paper只读精确重放 |

APH/原姿态矩阵、其他四项奖励、双边武器、dwell、3DOF/标架积分、12D/60D观测、Beta、Actor/Critic、PPO目标、GAE、优势归一化及正式训练配置文件均未修改。

新旧配置的TrainConfig完全相同；环境唯一差异是`posture_reward_mode`，门控参数在两者均默认为100。全部五个seed、2500/5000预算和既有smoke均支持。默认结果根目录：`outputs/approach_gated_2500`、`outputs/approach_gated_5000`，其下仍按实验名/seed归档。显式指定同一root时，实验子目录也不同。已有日志/模型时拒绝新训练覆盖。

论文37项正式计划数量保持37，新family不进入export_plan；原计划文件和旧结果未重写。论文表格读取器主动跳过新消融，不能把它归入原论文表IV或其他原论文统计。预算标签仍描述预算，是否论文实验由独立family/paper_experiment标记区分。

## Checkpoint与旧模型

环境版本保持v2.0.0（动力学未变）；完整env_config以及reproduction_assumptions保存模式/速度，构成奖励转移协议身份。训练入口、checkpoint恢复及worker snapshot恢复复用原有完整配置相等校验，因此paper↔approach_gated、门控速度不同均拒绝恢复，且拒绝发生在加载网络/优化器前。

原v2检查点缺少新增字段时，仅按其历史语义解释为paper、100m/s；实际已验证历史模型仍可只读评估。没有放宽配置比较，没有自动把paper模型迁成gated训练。所有新增checkpoint都有显式字段。

## 只读诊断定义

训练JSONL的`rollout.posture_diagnostics`、逐回合`episodes[].posture_diagnostics`及独立评估结果中保存：

- `posture_close_total`、`posture_engagement_total`、`posture_far_total`：三个区间实际Posture累计。
- `far_raw_total`、`far_applied_total`：远区原始/实际值；paper下两者相同。
- `far_gate_sum`、`far_samples`、`far_gate_mean`：按实际有效分配样本加权；paper下G是只读反事实诊断，绝不施加到奖励。
- `far_approaching/maintaining/receding_samples`及fraction：按c_rel正/零/负精确分类，无隐藏速度死区。
- `far_self_approaching_samples`：c_self>0的远区样本，可区别“Blue追近、Red却在远离”的情形。
- `attack_range_entries`：同一动作前分配目标满足d0>2000、d1<=2000的次数；目标切换本身不算进入。
- `in_attack_range_samples/fraction`：动作后距离<=2000的有效分配样本及比例，只是距离区间，不等于武器条件成立。
- `assigned_samples`、`unassigned_active_samples`及`diagnostic_steps`：提供分母和实际诊断覆盖范围。

比例来自总和/总样本数，不平均各worker/episode的均值；无远区样本时G均值为null，不造0。分量total是**所有有效Red的和**，与原`reward_components_total`的每Red槽平均口径不同；比较原回合平均Return时应按Red槽数换算，比较行为比例时按assigned/far样本归一化。历史没有这些字段的训练记录不补写；继续分析旧模型时应在新目录做独立评估。若恢复的历史tracker没有诊断，覆盖步数只记录新观察到的部分。

## 实际测试与CUDA验收

运行环境：WSL Ubuntu，uav Conda，NVIDIA GeForce RTX 5060 Laptop GPU。CUDA强制可用，没有CPU训练回退。

最终完整测试：**466 passed，1 warning，121.02秒**。新增文件48个用例，加上原目录对新实验的参数化覆盖。Python编译检查通过，差异格式检查通过。

覆盖主动接近、平行保持、Red远离而Blue追近、远区负值、100/2000及两侧、零距离、无目标/非法目标、动作前目标时序及本步阵亡；同时验证APH/状态/Blue动作/武器/终止/RNG一致，其他四项奖励一致，实际Reward等于五项之和。

诊断计算开关在两种模式下不改变环境和奖励；诊断收集开关的实际CUDA测试比较rollout、Actor动作、PPO指标、Actor/Critic及两优化器、worker快照、Python/NumPy/Torch/CUDA RNG，均精确一致。gated半rollout检查点在cursor=1、已累计3个真实worker步时保存，含worker重置后的tracker；恢复后全部张量、诊断、模型/优化器/RNG精确一致，未放宽为近似比较。

实际10进程CUDA审计：保持max_steps=250、worker种子1000～1009；只为审计使用horizon8、PPO epoch1。共160 joint steps、2次更新、480个有效UAV样本，全部有限，无无效样本进入loss，无数值故障；checkpoint恢复后下一rollout及更新精确一致。

实际新CLI smoke（既有smoke预算，不是正式预算）：2个spawn、horizon8、PPO epoch1，完成2个训练回合、500 joint steps、32次更新。两个完成回合均timeout，累计1351个有效训练样本，进入loss的无效样本0，数值故障0。其独立评估2局：胜率0%、平均Red存活1、Blue击落0，正常结束1/timeout1，数值故障0。死亡后的槽仍由原mask正常排除，不把它们误称成有效个体。

这些短程结果只验收能否正确运行和恢复，不证明行为或胜率改善。

## 历史paper实际回归与固定动作对照

只读来源：`outputs/xu_iv_v2_2500/main_3v3/seed_1/latest.pt`（2500回合模型）；SHA256前后保持`8c2c368f11563a944f92167d823684607cb0f0289f943258a2b47fe48e92fe66`。使用修改前已归档的seed10010000录像，对250步重新采样Actor均值动作，逐步与历史记录核对：动作、双方物理状态、Blue控制、存活、APH、五项Reward和击落均精确一致。

另把相同动作送入gated环境，只做固定动作反事实结算，不把历史模型恢复成gated训练。250步双方物理轨迹、事件及其他四项奖励精确一致：

| 项目（全Red求和） | paper | approach_gated，同动作 |
|---|---:|---:|
| Posture总计 | 41.908953755188 | -0.9603126663744056 |
| 近区Posture | 0 | 0 |
| 中区Posture | 12.576180998769019 | 12.576180998769019 |
| 远区Posture | 29.332772756419 | -13.536493665143436 |

共有538个有效分配样本，其中远区475；G均值0.14713923。远区相对接近285、保持0、远离190；Red自身沿LOS正向运动119。进入2000m共4次，区间内63个样本。门控后远区净值可为负，是原负值保留、正值被削减的结果，没有增加额外惩罚。这不是新策略训练结果，也不能把Return下降当作性能退化或改善。

## 运行与后续比较

用户可自行执行的新2500回合实验：

```bash
conda activate uav
python -m algorithm.experiments train --experiment posture_approach_gated_3v3 --seed 1 --episodes 2500 --root outputs/approach_gated_2500
```

不传`--episodes`仍默认5000；`--smoke`沿用原入口。门控速度可通过AirCombatConfig配置对象调整；该实验注册默认100，避免偷偷增加第二个变因。

对照时保持相同预算、seed和独立评估种子，优先比较独立评估胜率、真实Blue击落和Red存活；训练胜率与独立评估分开。再看远区原始/实际奖励、G均值、远区自身接近和相对接近比例、进入2000m次数与区间内比例、近/中区奖励及暴露风险。两个奖励目标的Return不能直接用“更大”判优。旧2500训练没有新增区间诊断，不得补造；可另行只读评估旧模型，在新目录输出对照诊断。

本轮产物：`outputs/approach_gate_acceptance/full_tests.xml`、外层`approach_gate_full_tests.log`和`approach_gate_compile.log`；`audit/acceptance.json`及`gated_resume.pt`；`cli_smoke/posture_approach_gated_3v3/seed_1/`完整短程日志、配置、checkpoint和2局评估。

**没有执行正式2500/5000训练，没有修改历史模型、录像、结果或训练记录，没有自动恢复旧模型继续训练。** 新消融已具备运行条件，行为改善与否仍等待后续正式对照实验。

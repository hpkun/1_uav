# Xu (2026) APH-MAPPO 前三阶段实施报告

日期：2026-10-22。论文：工作区 Xu 等《Hybrid Advantage-Priority Hungarian Assignment MAPPO for Multi-UAV Cooperative Air Combat Decision-Making》，15 页作者接受稿，DOI 10.1109/TAES.2026.3734384。读取了全文，并对第 3、4、7、8、11、12、13 页的公式、动作表、参数表与初始状态表进行了渲染核对。

## A. 实现概述

本次实现了独立的三维双边空战任务 `env.aircombat.AirCombatEnv`。支持 3v3、5v5、3v5，固定双方初始数组长度与稳定 UAV ID，通过 alive mask 表示死亡。Red 使用共享 12D Beta Actor，Blue 使用论文 15 动作库和 aggregate-sum 80/20 规则，无双边学习、自博弈或难度调整。集中 Critic 输入为五个稳定友方 12D 槽的 60D 拼接，输出五个个体价值头。

当前具备 `reset/step → APH → 观测 → Red/Blue 动作 → 同步动力学 → 连续攻击事件 → 五项奖励 → 下一状态 → 并行 rollout → GAE/PPO → 完整 checkpoint/恢复 → 固定种子评估` 的运行闭环。训练、checkpoint 审计、评估明确要求 CUDA；没有 CPU 回退。

这是一套带显式假设的实现和最小验证，不是论文性能或作者原始代码的完全复现。未执行正式 5000 回合 × 5 种子实验、其他算法对比或论文曲线复现。

## B. 实际修改

### 新增文件

| 文件 | 作用 |
|---|---|
| `env/aircombat/__init__.py` | 独立空战环境和配置公开入口 |
| `env/aircombat/config.py` | 集中论文常数、规模配置、可序列化复现假设 |
| `env/aircombat/scenarios.py` | 双方固定槽状态、表 VI～X 数据、随机训练采样、显式双方状态 reset |
| `env/aircombat/geometry.py` | 统一三维 LOS、角度、攻击优势与三段姿态优势 |
| `env/aircombat/assignment.py` | 纯函数 APH；攻击预分配、冲突消解、剩余列复制和 Hungarian |
| `env/aircombat/combat.py` | 双方逐对连续攻击计数、同步死亡、事件去重和归属 |
| `env/aircombat/opponent.py` | 表 I 物理动作顺序转换、15 次虚拟积分、聚合求和和 80/20 选择 |
| `env/aircombat/observations.py` | 原始 SI 12D 当前目标观测及 60D 槽拼接 |
| `env/aircombat/rewards.py` | 五项独立奖励、共同暴露风险、击落及终局奖励 |
| `env/aircombat/environment.py` | 同步 reset/step、异常回滚、终止/超时、环境 snapshot/restore |
| `algorithm/common/aircombat_vector.py` | 空战 spawn worker 入口、持续 RNG 流、回合统计；复用通用进程传输 |
| `algorithm/mappo/aircombat_rollout.py` | 真实 12D/60D rollout、最终状态 bootstrap、双 mask、准确完成回合预算 |
| `algorithm/aircombat.py` | 独立训练和 checkpoint 评估 CLI，JSONL 日志、回合阈值存档 |
| `tools/validate_aircombat.py` | 三规模完整回合、重复轨迹、真实 CUDA 更新、新 worker 精确恢复、短训练验收 |
| `tests/test_aircombat_math.py` | 动力学/几何、APH 全部边界、Blue 动作和逐项得分手算 |
| `tests/test_aircombat_environment.py` | 双方事件、奖励手算、同步互毁、失效槽、数值异常、完整回合 |
| `tests/test_aircombat_training.py` | CUDA rollout/更新、真实终态 bootstrap、GAE、singleton、完整恢复、独立评估和预算 |
| `docs/xu2026_aircombat_implementation.md` | 本报告和运行契约 |

### 修改已有文件

| 文件 | 修改 |
|---|---|
| `algorithm/mappo/trainer.py` | `ProtocolMAPPOTrainer.collect` 接入空战；逐个体标准化保留单样本优势；历史 90D 路径保留原处理 |
| `algorithm/mappo/config.py` | Protocol 默认对称初始化，明确 5000 回合、5 独立种子、200 局评估、singleton 处理及其他假设 |
| `algorithm/mappo/networks.py` | 复用 Beta Actor，增加无 trim 先验的对称输出初始化；复用原 ProtocolCritic |
| `algorithm/mappo/paper_protocol.py` | 接口身份升为 `mappo-array-12x60-v1.1.0`，注明真实观测来源；继续拒绝 90D 截取 |
| `algorithm/common/vector_env.py` | 注入环境/回合统计构造函数；有 snapshot/restore 的环境直接使用，历史 TaskState 分支保留 |
| `algorithm/common/checkpoint.py` | 复用 12/60 模型/优化器/RNG 存取；空战额外保存环境身份、双方真实状态、worker/回合统计与阶段 |
| `tools/numerical_cases.py` | 独立 NumPy 手算参考同步修正单有效样本的优势保留 |
| `tests/test_numerical_diagnostics.py` | 将原先“singleton 被归零”的断言更新为保留策略梯度的数学期望 |
| `README.md` | 空战成为首要入口，明确历史巡检边界和启动命令 |
| `pyproject.toml` | 项目描述包含空战与历史巡检 |

`env/paper_dynamics.py`、`env/dynamics.py`、`env/integrator.py` 和 `env/assignment.py` 的数学实现未改写：直接复用 RHS、RK4、物理动作映射、速度向量和 Hungarian。没有复制另一套 PPO 优化算法。

## C. 论文公式对照与接口契约

| 论文位置 | 实现位置与核对要点 |
|---|---|
| 式 (1)、(2)，表 II | `env/paper_dynamics.py:rhs/rk4_step`；`env/dynamics.py:physical_actions`；`env/integrator.py:rk4`。内部六维 `[x,y,z,v,gamma,psi]`；物理动作 `[mu,zeta,zeta_hat]` |
| 式 (3)、几何定义 | `geometry.py:relative_geometry`。三维速度、共同 LOS、距离及双方速度夹角余弦；目标角使用 `p_j-p_i` |
| 式 (4)、(5) | `geometry.py:attack_advantage/pair_advantages`。距离、前向锥和严格目标方向三个条件；反向调用同一函数 |
| 式 (6) | `geometry.py:posture_advantage`。近距 `-1/d`、中距余弦相加、远距速度因子乘 beta；无距离进度奖励 |
| 算法 1、式 (7)～(10) | `assignment.py:aph_assign/assignment_matrices`。每友机最多一个目标；攻击预分配可复用目标；剩余目标列复制、匹配列取模 |
| 表 I、式 (21)～(23) | `opponent.py:TABLE_I/BLUE_CONTROLS/candidate_scores/choose_blue_controls`。先转换表原始列顺序，再用相同 RK4 虚拟推进；得分对全部存活 Red 求和 |
| 连续攻击两步 | `combat.py:resolve_attacks`。每对独立计数；先生成双方全部事件再应用死亡；同一目标一次事件 |
| 式 (15)～(17) | `rewards.py:allied_penalty`。只用存活双方的共同暴露；友机不足 2 时为 0；不影响 APH 代价 |
| 式 (18)～(20) | `rewards.py:compute_rewards`。攻击、姿态、友机风险、击落、终局五项；总和直接相加，无缩放/截断/额外项 |
| 式 (25)，表 III 12D/60D | `observations.py:observe`；复用 `paper_protocol.py:pack_centralized_slots`。原始 SI 字段，按稳定 Red ID 拼接五槽，不足补零 |
| 式 (26) | 复用 `trainer.py:_update`。逐 UAV 有效样本均值后相加，共享参数；价值 MSE 无 1/2；缺失 head 的 Adam momentum/step 也不更新 |
| 式 (27) | 复用 `trainer.py:gae`。当前有效、下一失效的当步奖励保留；死亡后不 bootstrap，不跨自动 reset 累积 |
| 表 III | `config.py:ProtocolTrainConfig`。Actor/Critic 128/128 Tanh；Beta；lr=3e-4、gamma=.954、lambda=.95、clip=.2、entropy=.001、10 epochs、128 minibatch、10 workers、每 worker horizon1500；250 步环境上限 |
| 回合与种子协议 | `ProtocolTrainConfig`、`AirCombatVectorEnv`、`algorithm.aircombat`。默认 5000 完成回合、5 个独立训练种子；worker 初始 seed=1000*s+r；默认 checkpoint 阈值间隔50回合、评估200局 |

环境 `reset(seed=None, options=None)` 返回 `(observations, info)`，`step(actions)` 返回 `(observations, rewards, terminated, truncated, info)`。关键形状：

```text
friendly_states [N,6]       adversarial_states [M,6]
friendly_alive  [N]         adversarial_alive  [M]
assignments     [N] (-1表示无目标)
friendly_dwell  [N,M]       adversarial_dwell  [M,N]
observations    [N,12]      centralized_state  [60]
Red actions     [N,3]，归一化到[0,1]，物理映射后送动力学
H_a/H_b/F       [N,M]，死亡行列屏蔽
rewards         [N]，info.reward_components保存五个[N]分量
```

Rollout 保存 `[T,E,N,12]` 观测、`[T,E,60]` 集中状态、`[T,E,N,3]` 实际 Beta 动作，以及 log-prob、当前/下一价值、奖励、当前/下一 mask 和两种 `[T,E]` 边界。当前目标用于当前观测及该 transition 的奖励；物理推进与事件结束后重新 APH，生成真实下一观测。并行器单独返回 `transition_observations/transition_states` 和 reset 后的 `observations/states`，bootstrap 只使用前者。

`info` 同时记录实际物理控制、Blue 动作编号、当前/下一存活 mask、transition 分配、下一分配、事件、异常原因和计数。回合 return 定义为各 Red 累计奖励的初始团队均值；五项奖励日志均值分母为当前有效 Red transition 数。胜率分母为完成回合数，分子为全 Blue 消灭回合数；异常回合不算胜利。击落数量独立统计真实 Red 事件，不把初始 inactive 槽算作击落。

## D. 论文一致性

### 严格对应明确发表的内容

六个运动方程、三控制量和范围；默认时间步/速度范围/250 步；三种对抗规模；三维攻击条件和 alpha_j 定义；三段姿态优势；APH 的优先预分配、允许复用、冲突消解、剩余 Hungarian、列复制和未分配边界；Blue 15 动作及求和80/20策略；五项奖励公式；12D 字段；逐体有效损失和双 mask；表 III 网络尺寸与训练默认值；初始 worker 种子公式；全消灭胜率。

表 VI～X 均完整抄录了双方状态。原表顺序为 `[x,y,z,v,psi,mu,gamma]`，在代码中严格选列 `[0,1,2,3,6,4]`，mu 只保存在 metadata。

### 论文未公开或未唯一规定的显式复现假设

下列项目均在 `AirCombatConfig.reproduction_assumptions()`、`ProtocolTrainConfig.reproduction_assumptions()` 及每次运行的 `config.json` 中记录：

1. 重力 `g=9.81`，经典 RK4。分母阈值 `1e-6`；拒绝非有限/奇点阶段和跨越航迹角坐标极点的 RK4 阶段。末态速度裁剪到 `[100,400]`，heading wrap 到 `[-pi,pi)`。无 gamma ±85°、高度、转率、平滑或额外控制器。
2. 几何使用双精度和 `1e-9 m` 退化保护。距离用等价的 hypot 欧氏范数；夹角用等价的三维 `atan2(||u×r||,u·r)`，避免 arccos 在精确45°附近放大舍入误差；没有扩大锥角或增加角度容差。姿态余弦只保护 `[-1,1]` 合法区间。零 LOS 余弦为0、角为pi/2、禁止攻击；近零姿态分母保护。
3. 随机训练状态各变量独立均匀：Red x∈[4500,5500]，Blue x∈[9500,10500]，双方 y∈[3800,9600]、z∈[5000,7800]、v∈[175,230]、gamma∈[-.3,.3]、psi∈[-pi,pi]。所有已采样 UAV 之间初始至少100m；每架位置最多10000次拒绝采样。未采用巡检航点采样。这不是作者原始联合分布。
4. `shootdown_r0=10`：**论文未报告，实验占位值**，配置/CLI可改。不同目标的多个事件可各获得一次 r0；同一目标只有一个事件。
5. 几何锥/距离边界闭区间，alpha_j<pi/2严格。reset 不预加攻击计数；只在完整推进后的判定连续两步触发。按用户任务要求逐对跟踪，攻击事件不依赖 APH 是否选择该对。论文阈值的 `>` 与离散两步规则取后者。
6. 双方同一后态产生事件后同步死亡。多个攻击者同时满足同一目标时，最低稳定攻击者 ID 取得该目标一次击落奖励；死亡者可以在同一后态反击。对真实六机环形互毁有测试。双方全灭标记 `mutual_elimination`，按“Blue全部消灭”定义统计 win；终局奖励按0名存活友机计算。
7. 当前有效但本步死亡者保留当前动作奖励。攻击/姿态使用推进后、应用死亡前的几何及推进前已分配目标；友机风险使用死亡后的存活集合。终局奖励只发给当前 transition 有效者（包括本步刚死亡者），更早死亡者不重新激活或获得 PPO 样本。
8. 存活且未分配：O=1，保留自己 v/gamma/psi/z，alpha_i、全部目标字段和距离为0；正常执行 Actor 动作，无 idle trim/自动巡航。死亡槽全零。
9. 60D Critic 为五个稳定 Red 12D 槽拼接，共享128/128 trunk和五个独立个体价值头；共享 Actor。论文未唯一规定式 (14) 到这两个固定尺寸的映射及输出头拓扑。
10. 原始 SI 观测不归一化、不裁剪；奖励也不归一化/缩放。Beta参数为softplus+1，对称小增益输出权重、零输出bias，无历史trim先验。对称指两个Beta形状参数初始化，不是额外机动控制。
11. 优势按个体总体标准差+1e-8标准化；某个个体整个rollout只有一个有效样本时保留原优势，不归零。有效均值相加的共享参数聚合、Adam epsilon=1e-5、梯度范数裁剪=.5、Beta内部动作边界1e-6均为假设；存储log-prob对应裁剪后实际执行的同一动作。
12. 同分时按稳定ID/表I原始动作先后决定，候选评分无随机性。Blue仅在最终选第一/第二名时消费自身worker Generator。候选积分不改变真实状态/计数/RNG；无效候选分数为负无穷，若少于两个可行动作，独立终止为数值异常。
13. 真正动力学异常回滚整个联合物理步，本步五项奖励全0、不增加击落事件、不改变alive；步数前进并以独立 `numerical_failure` 原因结束回合，GAE终止bootstrap。此前发生的真实战斗事件保留，异常本身不伪装成新击落。
14. worker仅首次使用1000*s+r播种，以后reset继续同一随机流。PyTorch初始化/动作流由train seed控制；Python、NumPy、Torch CPU和全部CUDA随机状态均存档。跨硬件/驱动版本不承诺逐位一致。
15. timeout使用自动reset前真实最终状态bootstrap，但停止GAE跨回合传播；自然终止和数值异常都停止bootstrap。最终预算通过缩小最后采样worker集合达到准确完成回合数，单worker采样不超过horizon。
16. checkpoint默认50个完成回合阈值，**在完整rollout/PPO更新边界检查**，因此实际保存时可能已超过阈值（一个rollout跨过多个阈值时只保存当时模型）。该序列化时机未由论文公开；不是恰好第50/100回合的中途rollout模型。phase=`between_rollouts_after_update`；该阶段没有待处理rollout，故无需保存半个PPO更新。环境/worker、累计回合统计和RNG足够重启后续完整rollout。恢复要求环境和训练配置一致。

### 尚未实现或尚未验证

正式5000回合×5种子、每模型200局的统计评价、论文图表和对比算法明确不属于本次范围。没有作者原始训练分布、r0、输出头拓扑、观测尺度及数值积分细节，不能声称完全复现。Windows、Torch/CUDA版本也与论文Ubuntu实验不同。

## E. 测试结果

实际运行在 **PowerShell / Conda uav**，解释器 `D:\conda_envs\envs_dirs\uav\python.exe`；PyTorch `2.11.0+cu128`；GPU `NVIDIA GeForce RTX 5060 Laptop GPU`。按用户最新明确要求使用Windows环境。首次Conda激活探测遇编码错误，设置PYTHONIOENCODING=utf-8并加载PowerShell hook后解决；所有仓库测试、训练、评估、checkpoint审计均在uav环境运行。

最终测试结果：**273 passed in 125.84s**，失败0；记录为 `outputs/xu_full_tests_final.xml`。`python -m compileall -q env algorithm tools tests` 退出码0。完整验收记录为 `outputs/xu_aircombat_final_acceptance/acceptance.json`。

手算测试覆盖控制顺序、三维速度、前后视角、严格alpha_j边界、最大距离/锥角、零距、三段姿态；APH无攻击/优先/复用/无剩余目标/复制/死亡/空敌方/稳定同分；双向两步击落、失锁归零、同一目标去重、同步不同目标击落和真实几何双方全灭；五项奖励手算与精确总和；单友机风险为0；活着未分配仍执行Actor；死亡槽冻结；死亡当步奖励保留；独立数值失败；完整250步timeout；12/60一致、实际Beta log-prob、终态bootstrap、有效损失、singleton保留、空更新跳过、有限梯度、独立评估RNG与恢复逐位相等。

首次新测试57通过、3失败：两个是测试比较CUDA Tensor/统计对象的方式问题；另一个是45°边界的arccos舍入。修正比较方式与等价的三维角度求值后，新数学测试35项通过。早期全仓库269通过、1边界失败；最终全部273项通过。

## F. 训练验证与实际输出

验收工具使用正常 `dt=.25/max_steps=250`，只对训练smoke显式减小worker、horizon、PPO epochs与回合预算。

### 完整环境回合

这些回合以固定表I匀速直飞作为验证输入，不是训练策略或自动控制器；环境内部仍由真实Blue规则机动和攻击。

| 场景 | seed | 步数 | 结束原因 | Red存活 | 数值异常 |
|---|---:|---:|---|---:|---:|
| 随机训练场景3v3 | 771 | 233 | Red全部被Blue击落 | 0 | 0 |
| 随机训练场景5v5 | 771 | 250 | timeout | 3 | 0 |
| 随机训练场景3v5 | 771 | 203 | Red全部被Blue击落 | 0 | 0 |
| 表VII确定性迎头3v3 | 772 | 145 | Red全部被Blue击落 | 0 | 0 |

固定seed重复3v3的逐步状态、mask、分配、计数、奖励轨迹SHA256完全一致：`e1d87dd2826ab373c3f727a00d08064cd926b6b8e5dfe952e2b64d567554c666`。

### 真实CUDA rollout与更新

每规模2个spawn worker、每worker4步、1个PPO epoch；均使用正常250步环境，而非人工数组。

| 规模 | 联合环境步 | 有效Red样本 | optimizer minibatch steps | 价值损失 | 新worker恢复后精确一致 |
|---|---:|---:|---:|---:|---|
| 3v3 | 8 | 24 | 1 | 1.1375114918 | 是 |
| 5v5 | 8 | 40 | 2 | 1.3952288926 | 是 |
| 3v5 | 8 | 24 | 1 | 1.2882250547 | 是 |

Actor参数实际改变，Actor/Critic/PPO指标及梯度有限。记录的梯度范数是裁剪前范数，分别约141.9233、208.0160、11.6126；训练按声明的0.5阈值裁剪，没有修改奖励。恢复比较包含下一rollout每个张量、事件/回合汇总、更新指标、模型参数、两个Adam状态和所有worker快照，采用零容差。

3v3短训练实际输出：

```json
{"completed_episodes":2,"sampled_steps":500,"updates":8}
```

两个训练回合均250步timeout，训练期间数值失败0、胜率0、平均回合return=-31.1042097538，平均Red存活2。完整分解及PPO日志为 `outputs/xu_aircombat_final_acceptance/short_train/train.jsonl`；模型为同目录 `latest.pt`。没有把两回合训练当作收敛证据。

从真实checkpoint恢复后，固定种子10000000/10000001各评估一次，接口运行完成、输出有限、胜率0；**两局均在第32步出现Red航迹角坐标极点跨越，记录为numerical_failure，不是正常空战终止**。其异常原因、ID与步数在acceptance.json中保存。随机动作压力用例也在第43步记录一次相同类型异常。完整无异常回合与这些压力/评估异常分开报告。

## G. 剩余问题与统计一致性的限制

最重要的未知项是r0、随机训练联合分布、式(14)到12D/60D的映射、价值头拓扑、观测尺度、初始化、数值积分与速度边界、奖励评价时序、同步归属和未分配观测。上述选择均会影响学习曲线及胜率。

原始SI观测可能使Tanh隐藏层饱和；本次没有擅自加入归一化。短训练模型的确定性Beta均值动作实际会将飞行带向坐标奇点，因此当前短模型并不具有可靠空战能力。按要求没有引入±85°限制、额外飞控或人为生存奖励来隐藏此问题；异常路径已明确标记和测试。checkpoint阈值在更新边界保存的实际回合数也可能与作者第50/100回合的模型不同，见D.16。

不能根据少量全失败/timeout回合或有限数值判断论文性能复现成败；本次只证明模块数学、环境双边因果链、真实学习数据流和可恢复优化闭环。正式统计实验尚未运行。

## H. 工程清理与使用

新路径不使用 `Cooperative3DEnv`、巡检 `TaskState`、waypoints、priorities、completed、300m到点、进度差分、每步-.01、idle trim、巡检Hungarian代价、90D Critic、completion_rate或导航基线。唯一复用的任务外内容是数学求解器、动力学、积分器、Beta Actor、ProtocolCritic、GAE/PPO、进程传输和checkpoint通用存取。

历史巡检环境、runner/evaluation、工具、配置、回归fixture及测试仍保留，便于追踪和避免破坏历史结果。旧90D和早期v1.0数组checkpoint均不迁移到新的空战身份；空战header同时核对环境身份与网络协议，并在torch.load之前拒绝错误家族。

PowerShell统一激活：

```powershell
$env:PYTHONIOENCODING='utf-8'
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate uav
python -c "import sys,torch; print(sys.executable); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

验证与小预算运行：

```powershell
python -m pytest -q --junitxml=outputs/xu_full_tests_final.xml
python -m compileall -q env algorithm tools tests
python -m tools.validate_aircombat --output outputs/my_fresh_acceptance
python -m algorithm.aircombat train --scale 3v3 --episodes 2 --num-envs 2 --horizon 32 --ppo-epochs 1 --checkpoint-interval 1 --output outputs/my_fresh_smoke
python -m algorithm.aircombat evaluate outputs/my_fresh_smoke/latest.pt --episodes 2 --seed 10000000 --output outputs/my_smoke_eval.json
```

训练恢复需重复同一训练/环境参数，并加 `--resume PATH`。若从训练中途已生成的checkpoint恢复，会恢复当时尚未完成的环境回合；不会从reset状态假装接续。训练日志自动移除checkpoint之后的陈旧update行，再追加新结果。

正式预算示例仅供后续用户自行启动，本次未执行：

```powershell
python -m algorithm.aircombat train --scale 3v3 --seed-index 1 --r0 10 --output outputs/formal_3v3_seed1
python -m algorithm.aircombat train --scale 5v5 --seed-index 1 --r0 10 --output outputs/formal_5v5_seed1
python -m algorithm.aircombat train --scale 3v5 --seed-index 1 --r0 10 --output outputs/formal_3v5_seed1
```

其中r0=10仍为显式占位假设，正式使用需在实验记录中保留此事实。

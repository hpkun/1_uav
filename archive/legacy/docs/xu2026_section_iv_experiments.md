# Xu (2026) 第IV节实验复现：实现与验收报告

日期：2026-10-22。此报告对应论文第IV节、表IV/V、图5/9～17。论文外部基线核查另见 `docs/xu2026_baseline_audit.md`。前三阶段的环境定义与数值假设保留在 `docs/xu2026_aircombat_implementation.md`；其中“checkpoint只能在更新边界保存、可能越过目标”现已由本轮精确目标功能替代。

## A. 本轮功能与修改边界

本轮扩展实验协议、指定的消融/敏感性开关、日志、统计和静态绘图。默认主方法的运动方程、积分器、攻击条件、连续两步击落、APH、Blue动作库与aggregate-sum/p=.8、五项奖励、12D/60D字段、Actor/Critic结构、Beta初始化、观测尺度、PPO超参数和双mask语义均保留。

没有针对性能做改进；没有归一化、额外奖励、飞控、±85°限制、安全控制、自博弈或新的优化算法。numerical_failure继续独立结束/统计，不作为被敌方击落。只有实验指定的那一项在相应变体中改变。

### 文件清单

| 文件 | 本轮作用 |
|---|---|
| `algorithm/experiments.py`（新增） | 13个具名配置、37个正式独立训练任务、单种子训练/最终评估、两类Blue冻结模型评估、正式模型协议检查 |
| `configs/xu_iv/plan.json`（新增） | 实际导出的完整正式任务清单，包含每种子的环境/表III配置、全部checkpoint目标及两种Blue评估协议；导出不启动训练 |
| `tools/report_aircombat.py`（新增） | 从本地日志生成图5/9/10/11/12、表IV/V及Blue敏感性CSV；均值/样本标准差、20更新移动平均、缺失结果标记 |
| `tools/aircombat_trajectories.py`（新增） | 从指定checkpoint运行表VI～X，记录双方真实状态/存活/控制/奖励、分配变化、击落事件及步数，输出3D、平面、分配事件图 |
| `tools/validate_xu_experiments.py`（新增） | CUDA小预算端到端验收；明确smoke标签；不启动正式任务 |
| `tests/test_xu_experiments.py`（新增） | 配置隔离、单因素、风险手算、mask消融、精确checkpoint及恢复、日志无反馈、统计、图表重建、真实轨迹回放测试 |
| `env/aircombat/config.py` | 增加assignment_mode、allied_penalty_enabled、opponent_aggregation三个显式实验选择；默认值保留主方法 |
| `env/aircombat/assignment.py` | 只为w/o APH增加独立最近存活目标选择；原APH函数未改写 |
| `env/aircombat/rewards.py` | 只为w/o allied增加移除该分量的开关；另加纯函数记录式(17)两项正风险；原allied_penalty计算保留 |
| `env/aircombat/opponent.py` | 在原逐对得分后增加式(24)max/min聚合；默认仍sum |
| `env/aircombat/environment.py` | info增加风险和终止元数据；不改变默认奖励/状态推进；数值失败步风险记零 |
| `algorithm/mappo/config.py` | 增加默认true的ppo_masking选择及checkpoint阶段说明；表III参数不变 |
| `algorithm/mappo/trainer.py` | w/o masking仅让真实团队N个槽参与PPO损失；物理mask、GAE和优势标准化有效集合保留 |
| `algorithm/mappo/aircombat_rollout.py` | 精确回合checkpoint回调；保留半个rollout、波次游标、预采样动作、真实终态；下一价值仍按原整波次求值 |
| `algorithm/common/checkpoint.py` | 保存/恢复collecting_rollout阶段及完整buffer/cache；模型单独评估可读取前三阶段省略新开关的默认配置 |
| `algorithm/common/aircombat_vector.py` | 逐回合回报、累计两项风险、三类终止统计 |
| `algorithm/common/vector_env.py` | 空战回合记录worker ID和worker episode index，便于归档；历史任务分支保留 |
| `algorithm/aircombat.py` | 逐回合日志、精确目标存档、半rollout恢复、分组评估流；拒绝训练时启用Blue评估变体 |
| `README.md`、`requirements.txt`、`pyproject.toml` | 实验入口说明与静态绘图matplotlib依赖 |
| 本报告及baseline audit（新增） | 一一对应、运行命令、真实结果和未公开信息 |

## B. 论文实验与实现对应

| 论文实验 | 实现/数据源 |
|---|---|
| IV.B主方法三规模及图5 | `main_3v3/main_5v5/main_3v5`；每种子独立目录；final_evaluation.json；跨训练种子样本std |
| IV.C.1表IV三项消融 | `without_aph/without_allied/without_masking`，仅3v3；同main_3v3作为对照 |
| IV.C.2表V、图9 | 7个唯一系数配置；每项一个训练种子、同5000完成回合预算；基准结果供三条单因素中心点复用 |
| IV.B.2概率敏感性 | `blue --kind probability`；一份固定完成5000回合的5v5主方法模型；p=1/.8/.6，每种5个评估种子×200局 |
| IV.B.3聚合敏感性 | `blue --kind aggregation`；5份独立训练种子的5v5 episode4950模型；sum/max/min，每模型每模式5×40=200局 |
| 图10 | 每完成回合的训练episodic reward；按episode对齐后跨种子均值和样本std |
| 图11、12 | train.jsonl中critic_loss/entropy；每种子先做完整20-update窗口均值，再对相同update取跨种子均值和样本std |
| 表VI～VIII、图13 | 三种3v3初始状态；每种分别输出trace.json、trajectory_3d/trajectory_xy/assignment_events的PNG和SVG |
| 表IX、图14～15 | 代表性5v5，三维和平面图及事件图 |
| 表X、图16～17 | 代表性3v5，三维和平面图及事件图 |
| 图7、8外部算法比较 | 本轮不输出算法成绩；材料不全，明确暂不实现，不用替代方法冒充 |

图表只读取记录，不调用环境训练/重置，不更新模型。没有把论文报告数字写成本地结果。算术测试中的人工数组仅位于pytest临时目录，不进入验收结果。样本不足20次更新时图11/12明确显示“Requires 20 recorded updates”，没有缩短窗口、补点或造曲线。

## C. 全部配置与正式协议

统一默认：每配置独立训练；5个训练种子索引s=1～5；每种子5000个**完成回合**；max_steps250；10个worker；每worker horizon1500；每50完成回合触发checkpoint回调，后续新训练默认每250回合实际写盘，并额外保留4950；最终每模型200个独立评估回合。worker初始seed=1000*s+r（r=0～9）。具体索引1～5是本地明确选择，论文只给出了数量和公式。

Actor/Critic均128/128 Tanh，Beta，学习率3e-4，gamma=.954，lambda=.95，clip=.2，entropy=.001，10个PPO epochs，minibatch128，保持表III。正式命令不使用smoke参数。

| 名称 | 规模 | 相对主方法唯一变化 | 训练种子 |
|---|---|---|---|
| main_3v3 | 3v3 | 无 | 1～5 |
| main_5v5 | 5v5 | 无 | 1～5 |
| main_3v5 | 3v5 | 无 | 1～5 |
| without_aph | 3v3 | assignment_mode=nearest | 1～5 |
| without_allied | 3v3 | allied_penalty_enabled=false | 1～5 |
| without_masking | 3v3 | ppo_masking=false | 1～5 |
| sensitivity_baseline | 3v3 | 无，eta/e1/e2=5/20/25 | 1 |
| eta_3、eta_7 | 3v3 | 仅attack_eta=3或7 | 1 |
| e1_10、e1_30 | 3v3 | 仅terminal_e1=10或30 | 1 |
| e2_15、e2_35 | 3v3 | 仅terminal_e2=15或35 | 1 |

合计13个名称、37个独立正式训练任务：15主实验+15消融+7单因素。九个表V单因素数据点中的三次中心设置是同一个实验，复用sensitivity_baseline，不重复训练三份等价配置。

### 消融约定

最近目标按本机到所有存活敌机的三维欧氏距离独立选择，允许不同友机复用同一目标；相同距离选择最低稳定敌机ID，无额外全局匹配、滞回或锁定规则。这是原文未给出tie细节时的最小约定。

w/o allied只将最终allied奖励分量置零，其他四项逐项不变。两项风险仍按真实物理状态计算、累计和评价，不因该消融而变成零指标。

w/o PPO masking只关闭损失样本筛选：初始真实Red数N个槽中的失效槽进入Actor/Critic loss；五槽协议的不存在槽和未执行worker padding不作为真实样本。状态O/alive、飞行、攻击、APH、奖励、GAE及优势标准化的有效集合均保留。失效槽用原本全零观测、合法Beta bookkeeping动作(.5,.5,.5)及精确对应log-prob，动作不会作用于物理状态。GAE仍使死亡后优势/return为0，故其Critic拟合0、Actor只可能通过熵项受影响。该数值约定是论文“仅PPO更新关闭mask”没有公开实现细节时的显式选择；没有重新定义死亡后奖励或延长死亡轨迹。

### 风险统计和日志单位

`risk_terms.proximity[i]`为式(17)中对其他存活Red的 `max(0,1-d/tau_min)`之和除以N_alive-1；shared_exposure同理使用原始c_ik。少于2名Red时两项为0。以本来reward使用的死亡应用后存活集合求值；数值失败回滚步为0。

每个episode保留逐Red的**时间步累计**风险数组，再对初始Red数取均值，作为该回合proximity_risk/shared_exposure_risk；不乘dt、不另设阈值、不裁剪。表IV对每模型200局先取均值，再对5个独立训练模型取均值和样本std。论文未完整公开最终团队聚合分母，本地选择在此明确披露，并保留原数组便于后续审计。

回合return同前三阶段：初始Red各自累计reward的团队均值。五项reward同时记录累计团队均值和当前有效Red transition分母的逐步均值。所有训练回合记录在train.jsonl的episodes中，并有worker/worker_episode_index；正常战斗终止、timeout、numerical_failure分别计数。日志中的训练win_rate属于episode_metrics；独立评估胜率只来自独立的final_evaluation.json，绘图不会混用。

所有统计std用ddof=1；只有一个训练种子时std=null，不伪造0。表IV/V胜率列使用百分比；summary.json及逐回合win_rate使用[0,1]比例。Blue概率实验std是在5个评估种子组之间计算；Blue聚合实验std是在5个独立训练checkpoint之间计算，二者不混淆。

## D. 单因素与记录无反馈验证

配置测试对dataclass完整字典逐字段做差：三个消融恰好改变assignment_mode、allied_penalty_enabled、ppo_masking中的一个；六个非基准系数配置恰好改变对应系数字段；Blue变体恰好改变p或聚合字段。训练入口拒绝p!=.8或非aggregate-sum，冻结Blue敏感性接口还检查5v5、主方法开关、正式5000/4950计数、表III配置及5个checkpoint的独立训练种子。

同一reset/seed/action下，w/o allied和主方法的状态、观测、分配、Blue动作、RNG完全相同；其他四项reward完全相同，总reward只差被移除分量，原始风险记录一致。

关闭PPO损失mask的手算用例中，真实有效样本仍为3，损失样本为6；GAE原数组及环境配置没有变化，损失和梯度均有限。继续使用双mask而不修改物理存活。

### 精确checkpoint的数据流问题及修复

上轮checkpoint只能在更新边界保存，无法保证存在恰好4950回合模型。本轮补齐精确目标，并保存中途rollout/cache；没有为checkpoint提前调用PPO。

新增验证发现：若记录checkpoint时拆分并行波次，同时按拆分后的批大小计算Actor采样/下一Critic，会改变随机采样或CUDA矩阵计算的舍入。这是记录路径影响学习数据的实现错误，不是论文设计问题。

具体复现：seed1000、2个worker、测试环境max_steps2、horizon4、checkpoint interval1，对照collect()与带实际save_checkpoint回调的collect()。保持整波Actor采样后，原拆批下一价值仍出现最多2.980232238769531e-7的差异。

最小修复：每个原始波次一次性采样Actor，缓存精确动作/log-prob/当前价值；物理执行可按目标拆分，先保留每个worker真实终态，再按原整波次一次求下一价值。snapshot保留buffer、cache、游标、RNG和worker状态。修复后，开/关实际checkpoint记录的rollout每个张量、回合记录、PPO指标和参数均零容差相等；从半波次及最终目标checkpoint恢复也精确一致。

这是本轮唯一针对明确数据流错误的修复；没有改变动力学、攻击、观测或奖励，也没有利用修复提高胜率。

## E. 实际测试和CUDA短程验收

运行环境为用户指定的PowerShell/uav，解释器 `D:\conda_envs\envs_dirs\uav\python.exe`，PyTorch2.11.0+cu128，RTX5060 Laptop GPU。训练、模型审计和评估均强制CUDA。

全仓库测试、最终实验测试及静态编译的真实输出分别保存于：

- `outputs/xu_iv_full_tests_final.xml`
- `outputs/xu_iv_experiment_tests_final.xml`
- `outputs/xu_iv_final_acceptance_console.log`

验收工具正式清单仅导出，未启动任何5000回合任务。实际只运行：一份3v3两回合短训练；其余配置每份2个真实联合步的一次PPO更新；为验证五种子归档及Blue接口，额外创建独立种子的极短模型。它们全部标记is_smoke=true，未完成预算的模型保留真实completed_training_episodes=0，不冒充训练完成。

两类Blue验收缩小为每评估种子1局：概率3×5=15局；聚合5模型×3模式×5=75局。正式默认仍分别为每种子200/40局，未更改。五个场景轨迹实际运行并绘图，保留真实numerical_failure和事件，不要求击落时刻匹配论文。

详细逐项输出和最终测试数量在本报告末尾的验收结果补充。早期测试曾有1项旧fixture缺少outcome字段导致统计KeyError，改为对缺失的历史字段不计入三类新统计；实际环境始终提供outcome。没有改变环境结束行为。

## F. 可直接使用的正式命令

以下正式命令是后续执行说明，本次没有运行。每次训练命令只启动选定实验的一个独立种子，完成后自动从latest.pt进行200局独立评估并保存final_evaluation.json。

```powershell
$env:PYTHONIOENCODING='utf-8'
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate uav
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"

# 只导出正式清单，不训练
python -m algorithm.experiments plan --output configs/xu_iv/plan.json

# 单个主实验/单个种子：默认5000回合，10环境，horizon1500
python -m algorithm.experiments train --experiment main_3v3 --seed 1 --root outputs/xu_iv_formal

# 三种主实验、各5个独立种子（后续用户启动）
foreach ($experimentName in @('main_3v3','main_5v5','main_3v5')) {
    foreach ($seedIndex in 1..5) {
        python -m algorithm.experiments train --experiment $experimentName --seed $seedIndex --root outputs/xu_iv_formal
        if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect the recorded exception.' }
    }
}

# 表IV消融：同主方法预算和种子
foreach ($experimentName in @('without_aph','without_allied','without_masking')) {
    foreach ($seedIndex in 1..5) {
        python -m algorithm.experiments train --experiment $experimentName --seed $seedIndex --root outputs/xu_iv_formal
        if ($LASTEXITCODE -ne 0) { throw 'Ablation training failed.' }
    }
}

# 表V：7个唯一训练任务，单一相同训练种子
foreach ($experimentName in @('sensitivity_baseline','eta_3','eta_7','e1_10','e1_30','e2_15','e2_35')) {
    python -m algorithm.experiments train --experiment $experimentName --seed 1 --root outputs/xu_iv_formal
    if ($LASTEXITCODE -ne 0) { throw 'Coefficient experiment failed.' }
}

# 中途checkpoint恢复，保留原训练配置和半个rollout
python -m algorithm.experiments train --experiment main_5v5 --seed 1 --root outputs/xu_iv_formal --resume outputs/xu_iv_formal/main_5v5/seed_1/episodes_004950.pt

# 单独重做最终200局评估
python -m algorithm.experiments evaluate outputs/xu_iv_formal/main_3v3/seed_1/latest.pt --output outputs/xu_iv_formal/main_3v3/seed_1/final_evaluation.json

# p敏感性：一个固定5v5最终模型，3×5×200=3000局
python -m algorithm.experiments blue --kind probability --checkpoints outputs/xu_iv_formal/main_5v5/seed_1/latest.pt --output outputs/xu_iv_formal/blue_probability

# 聚合敏感性：五个episode4950模型，每模型/模式200局
$aggregationModels = 1..5 | ForEach-Object { "outputs/xu_iv_formal/main_5v5/seed_$_/episodes_004950.pt" }
python -m algorithm.experiments blue --kind aggregation --checkpoints $aggregationModels --output outputs/xu_iv_formal/blue_aggregation

# 静态科研图表，只读日志，无训练
python -m tools.report_aircombat --root outputs/xu_iv_formal --output outputs/xu_iv_formal_figures

# 表VI～VIII/图13
foreach ($caseName in @('random_3v3','head_on_3v3','rear_end_3v3')) {
    python -m tools.aircombat_trajectories outputs/xu_iv_formal/main_3v3/seed_1/latest.pt --scenario $caseName --output outputs/xu_iv_formal_trajectories/$caseName
}
# 表IX/图14～15；表X/图16～17
python -m tools.aircombat_trajectories outputs/xu_iv_formal/main_5v5/seed_1/latest.pt --scenario representative_5v5 --output outputs/xu_iv_formal_trajectories/representative_5v5
python -m tools.aircombat_trajectories outputs/xu_iv_formal/main_3v5/seed_1/latest.pt --scenario representative_3v5 --output outputs/xu_iv_formal_trajectories/representative_3v5
```

评估主方法种子为10000000+10000*s起连续200个独立seed，各seed1局。Blue敏感性严格使用100～104分组，每组首次reset播种，后续episode继续同一个独立Generator流；该后续reset约定未由论文唯一规定，已在evaluation_protocol中记录。Red均采用Beta均值确定性评估，这是沿用前三阶段的显式假设。

smoke命令必须使用独立输出根目录，并带`--smoke`；它会显式减小回合数、worker、horizon、epochs及最终评估数，不作为正式实验结果：

```powershell
python -m tools.validate_xu_experiments --output outputs/my_fresh_iv_acceptance
python -m pytest -q
```

## G. 尚不具备完整条件的基线

GA-MCTS原文的区间不确定性、GA和MCTS参数/具体状态动作；CA MADDPG AFF的原始状态动作、历史上下文与特征融合网络连接/损失；LIORL原始离线交战数据及离线限制损失，均未完成原文/作者实现核验。

本次尝试原始DOI/IEEE来源，前两篇未获得全文，第三篇可读原始摘要但正文受访问验证阻断。GitHub精确算法/UAV组合检索没有找到可核验作者实现，不能据此断言代码绝不存在。完整来源与访问边界见baseline audit；本轮不以其他算法替代，不提供图7/8本地对比结果。

## H. 已知差异、未公开假设与运行状态

保留前三阶段的r0=10未报告占位值、随机场景联合分布、RK4/g/速度末态处理、奇点拒绝与回滚、原始SI观测、共享Actor/五价值头、Beta参数化和对称初始化、Adam epsilon/梯度裁剪/优势标准化、双边事件归属、未分配观测、互毁统计及奖励时序约定。本轮没有主动修正这些方法选择。

本轮额外明确：最近目标tie最低ID；loss-only mask消融的合法失效动作和零GAE目标；累计风险团队聚合；五个种子具体索引；评估Beta均值及分组reset流；checkpoint是当时已完成回合数的精确快照，可能位于rollout内，因此Actor仍是最近一次完整PPO更新后的模型；会同时保存其pending rollout，恢复时不丢数据或额外提前更新。这些序列化/评估细节不能声称来自作者源代码。

所有主实验、三项消融、单因素配置均具备启动条件；两种Blue敏感性接口与协议具备运行条件，但正式评估仍等待真正完成训练的5v5最终/4950模型。图表工具和典型场景工具已实际运行。完整5000回合×5种子的37个训练任务、200局正式最终评价、两组正式Blue评价仍未运行。任何smoke胜率、失败数或短曲线均不能当作论文实验统计。

## 最终验收输出补充

- 全仓库测试：**300 passed in 139.39s**，失败0、跳过0，见 `outputs/xu_iv_full_tests_final.xml`。
- 在该次全仓库测试后又增加了“不允许正式重试覆盖已有smoke元数据”的保护和对应测试；最终整个新增实验测试文件单独运行 **28项通过**，见 `outputs/xu_iv_experiment_tests_final.xml`。这些测试与上述300项有重叠，不将两次数量相加宣称328项。
- `python -m compileall -q env algorithm tools tests` 成功。早期全仓库299项通过记录也保留，但交付采用上述最终记录。
- 最终完整验收：`outputs/xu_iv_final_acceptance/acceptance.json`。所有训练/模型记录位于其 `runs/`；`formal_training_executed=false`。

真实短训练结果：3v3 seed1，两环境，horizon8，1个PPO epoch，2个完成回合、500个联合环境步、32次PPO更新。两个回合均250步timeout，数值失败0，平均训练return=-31.1528124406，平均Red存活2.5，Blue击落0。另有20份独立的2步rollout/单次更新验收记录（包括其余具名配置和用于种子/Blue协议验证的额外模型），不是20份训练完成模型。

精确checkpoint实际索引：

| 目标回合 | 实际完成回合 | 已更新PPO次数 | 联合环境步 | 阶段 |
|---:|---:|---:|---:|---|
| 1 | 1 | 31 | 499 | collecting_rollout |
| 2 | 2 | 31 | 500 | collecting_rollout |
| latest | 2 | 32 | 500 | between_rollouts_after_update |

其1/2回合缩小目标用于测试精确50/4950回合机制，不代表已经生成正式4950模型。

最终短模型的2局独立评价胜率0，2局均numerical_failure；Blue概率smoke的15局、聚合smoke的75局也均numerical_failure。它们证明协议路径、种子分组、冻结模型和归档可执行，**没有形成可用于比较对手强弱的正常空战统计**。统计和图表没有把这些失败标成战斗击落。

五种典型场景的真实输出如下；仅用极短模型，未修改初态、控制规则或事件时间：

| 表/场景 | 仿真步数 | 结束原因 | 真实击落事件 |
|---|---:|---|---:|
| VI random_3v3 | 37 | numerical_failure | 0 |
| VII head_on_3v3 | 33 | Red全部被击落 | 3 |
| VIII rear_end_3v3 | 28 | numerical_failure | 0 |
| IX representative_5v5 | 32 | numerical_failure | 0 |
| X representative_3v5 | 39 | numerical_failure | 0 |

迎头场景第33步的事件为Blue0击落Red0、Blue0击落Red1、Blue1击落Red2；直接来自trace.json中的环境事件。其他场景的异常保持真实记录，没有人为调整击落时刻或让未训练模型匹配论文图。

已生成的静态文件：`figures/fig5/fig9/fig10/fig11/fig12`各PNG/SVG，表IV/V和两类Blue CSV，及每种场景的三维、平面、分配事件PNG/SVG。图5和表IV的5种子样本std计算路径已用真实极短模型归档走通；所有图表标记LOCAL/partial_or_smoke。3v5仅实际验收一个种子，std留空；5v5/3v5没有足够20次更新的loss/entropy曲线，面板明确提示数据不足。真实记录重生成图表不涉及任何额外训练或仿真。

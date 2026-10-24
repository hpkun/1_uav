# 迎头初始化与 MADSAC-inspired 受控实验

2026-10-24。实现、确定性测试、CUDA 短程验收已完成，未运行正式2500或5000回合训练，未恢复历史模型继续训练。MADSAC-inspired 是将 Li 奖励移植到 Xu 环境的受控实验，不是完整复现 MADSAC 算法。

## 配置与修改范围

| 实验 | 初始化 | 奖励 | 是否进入 Xu 正式结果 |
| --- | --- | --- | --- |
| main_3v3 | random_training | paper | 保持原有协议 |
| posture_approach_gated_3v3 | random_training | approach_gated | 否 |
| headon_xu_3v3 | head_on | paper | 否 |
| headon_madsac_3v3 | head_on | madsac_inspired | 否 |

为兼容已有配置，奖励选择器仍使用历史字段名 `posture_reward_mode`；取值为 `madsac_inspired` 时切换**整个奖励函数**，不是只替换 Posture。旧配置缺少新增字段时，按现有配置正规化机制得到默认 paper/random_training；没有放宽跨模式恢复检查。

修改位置：

- `env/aircombat/scenarios.py`：增加随机位置/速度的 head_on；保存奖励用战场中心。
- `env/aircombat/config.py`：增加奖励模式与明确的奖励参数、边界参数及适配说明。
- `env/aircombat/madsac_reward.py`：独立几何与 R1～R4 结算。
- `env/aircombat/rewards.py`：增加显式分派；原 Xu 五项结算公式未改动。
- `env/aircombat/environment.py`：在 info 提供实际奖励分量名称和战场中心；MADSAC 不输出不适用的 Xu Posture 诊断。
- `algorithm/common/aircombat_vector.py`、`algorithm/mappo/aircombat_rollout.py`：仅将日志中的固定五项名称改为按配置选择的分量名称；采样顺序、Rollout 数组、PPO、GAE 均未改动。没有完成回合时，MADSAC 对应的四项回合统计为缺失值，不能填 Xu 五项或伪造零。
- `algorithm/aircombat.py`、`algorithm/common/aircombat_progress.py`：记录初始化及奖励身份，启动时显示初始化和奖励用半径。
- `algorithm/experiments.py`、`tools/report_aircombat.py`：注册、隔离两个实验，排除论文计划和结果汇总。
- `tests/test_headon_madsac.py`：新增90项测试。

APH、其 Xu 原始姿态优势、攻击几何、锁定、Blue 15动作策略、飞行动力学、Actor/Critic、观测维度及优化目标均未修改。武器仍为2000米、45°、连续两步有效攻击。4000米只是新奖励分段阈值。

## 迎头初始化

Red：γ=0，ψ=0；Blue：γ=0，ψ=π。位置和速度分布仍为 Red x∈[4500,5500]、Blue x∈[9500,10500]、y∈[3800,9600]、z∈[5000,7800]，v∈[175,230]，沿用原最小100米间隔拒绝采样。

迎头分支不采样角度。它消费旧采样器两次角度抽样所占的随机数槽位但不使用其数值，以保持同一种子下后续 UAV 的位置、速度、拒绝采样和重置随机流一致。原 random_training 分支的数值和随机数序列不变。3v3、5v5、3v5各测3个种子：新旧 xyz/v 和采样后的 RNG 状态逐值相等；两个新增实验在相同种子下的完整初始物理状态相等。

这只是共同朝向，与表 VII 的基本迎头方向一致；没有让每架 UAV 指向敌方实际坐标。已有固定表 VII 场景 `head_on_3v3` 也未改变。

## 论文几何与不可忽略的印刷歧义

已读取本地 Li 等（2023）论文第6页的图2、公式（6），以及第14～15页的公式（25）和说明。

公式（6）印刷的 ATA 写作 `angle(o3→o1, -V2_xy)`，AA 为 `angle(o1→o3, V2_xy)`；这两式按字面给出相同的角，无法同时表达图2中不同位置的 ATA 和 AA。**本实验依照图2采用本机水平速度定义 ATA**，不能声称逐字实现存在歧义的印刷式。

记本机为 a、目标为 b，L=b_position−a_position，Lh=(Lx,Ly)，双方真实水平速度为 uh、wh：

- ATA = angle(uh,Lh)。
- AA = angle(wh,Lh)，等价于目标指向本机的 LOS 与负目标速度的夹角；追尾同向为0°，迎头为180°。
- HA = atan2(|Lz|, ||Lh||)，即 LOS 与水平面的夹角，不是 LOS 与本机三维速度的夹角。

水平速度方向由 `v*cos(gamma)*(cos(psi),sin(psi))` 计算，保留 cos(gamma) 的符号；不是直接拿 Xu 三维攻击角替代。交换双方时必须反转 LOS 并交换两者速度。角度用 atan2 计算，再转为度数，与配置的度数阈值比较。仅容许1e-12度的浮点舍入误差。

**退化情形工程约定**：水平 LOS 为零或本机水平速度不超过现有 geometry_epsilon 时，ATA 无定义，不发放依赖 ATA 的过程奖励；有限哨兵角 ATA=180°并附 valid=False。目标水平速度为零只使 AA 无定义，不影响本机 ATA/HA 的 R3；AA=180°并附 AA_valid=False，不进入 AA 档位。零距离 HA=0、ATA/AA 哨兵，奖励有限。此约定没有更改武器或 Xu 几何。

## 实际奖励

总奖励 `R=R1+R2+R3+R4`，分量日志只含 R1、R2、R3、R4，不叠加 Xu Attack/Posture/Allied/Shootdown/Terminal。

1. **R1** = `10×本步归属该Red的真实Blue击落数 − 10×该Red本步阵亡指示量`。归属直接读现有真实事件；同时击落和阵亡独立相加。未参战者为0，多目标真实击落逐个累加。
2. **R2** = `−10×[ρ_before≤8000 且 ρ_after>8000]`，ρ为相对战场中心的水平距离。中心在每次 reset 时取双方全部初始位置的水平平均，保存于 CombatState.metadata。只结算仍存活 Red；持续在外不重复扣罚，返回再出去重新触发。没有高度边界、碰撞、击落或强制终止。
3. **R3** = `0.001×[d≥4000 且 ATA≤30° 且 HA≤30°]`，对象为动作前 APH 分配目标，使用动作后的真实几何。
4. **R4 = R41 + R42**。R41 对动作前 APH 目标，在 d≤4000 且 AA≤30°时，按 ATA/HA 均≤30°、15°、5°取最高档 +0.01、+0.02、+0.10。R42 将每个存活 Blue 作为本机反向计算几何，在 d≤4000、AA≤30°时取相应 −0.015、−0.025、−0.15；所有 Blue 中只取最严重惩罚，不求和。进攻和防御可同时成立。

R2～R4 仅对动作前后均存活的 Red 结算，R3/R41 的目标还须在武器结算后存活。已死亡槽不继续获取过程奖励；新阵亡者本步 R1 仍正常结算。未分配目标时没有进攻奖励，仍检查所有存活 Blue 的防御威胁。

**和 Li 式（25）相同**：±10、−10、+0.001，4000米阈值，30°/15°/5°档位和全部档位数值。嵌套档只取最高档。按题目明确的包含关系，恰好4000米时 R3 和 R4 可以同时成立。

**适配/工程假设**：沿用 Xu 真实武器事件而非 Li 原武器模型；进攻 APH 对象、防御所有 Blue 的最大威胁、攻防相加；post-motion/post-death 过程奖励时序；8000米圆形奖励区域和可重复出界触发；ATA 印刷歧义解释与退化处理。Li 没有完整公开这些多目标冲突代码。8000米不是两篇论文共同公布的参数。

全部奖励数值、角度/距离阈值和半径均在 AirCombatConfig 的 `madsac_*` 字段中保存，进入配置文件及 checkpoint；不是隐藏常数。配置校验拒绝非法范围、非有限值和逆序档位。没有增加权重、接近速度项或其他惩罚。

## 确定性实测

下表均为单架 Red 的实际结算；几何场景 R1/R2 为0。全部数据见本轮输出目录的 deterministic.json。

| 场景 | ATA/AA/HA | R3 | R4 | 总奖励 |
| --- | --- | --- | --- | --- |
| 同向追尾3000米 | 0°/0°/0° | 0 | +0.100 | +0.100 |
| 迎头3000米 | 0°/180°/0° | 0 | 0 | 0 |
| 同向追尾4000米 | 0°/0°/0° | +0.001 | +0.100 | +0.101 |
| 同向追尾5000米 | 0°/0°/0° | +0.001 | 0 | +0.001 |
| 本机侧向3000米 | 90°/0°/0° | 0 | 0 | 0 |
| 高度视线45°、3000米 | 0°/0°/45° | 0 | 0 | 0 |
| 进攻最高档、另两Blue威胁15°和5° | 各自反向计算 | 0 | +0.10−0.15 | −0.05 |

真实规则 Blue、种子100、固定配平动作、两步锁定：A1的 R0 击落 B0 得 R1=+10；A2的 R0 击落 B0 同步被 B1 击落，R1=0；A3的 R0 实际击落两架 Blue，R1=+20。单独被击落场景 R1=−10。以上事件来自真实环境；没有 mock 成功事件。

R2 测试半径100米（只用于单元公式测试），位置序列0→100→101→150→99→101，奖励为0、−10、0、0、−10。高度变化不影响此水平判定。正式实验默认8000米。

## 测试与 CUDA 验收

使用 WSL Ubuntu 的 uav Conda 环境，GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU。

- 新增测试：**90 passed**；完整测试：**570 passed in 147.40s**。`python -m compileall -q env algorithm tools tests` 和 `git diff --check` 均通过。
- 新测试实际10个 spawn worker，诊断配置 max_steps=2、horizon=3、PPO epoch=1，30联合步，90/90有效UAV样本，一次PPO更新；没有数值异常。回合结束自动reset、worker战场中心、半批次精确边界checkpoint、Actor/Critic/Adam、RNG和更新后结果全部精确恢复。此诊断配置不写回正式配置。
- 两个真实 CLI `--smoke` 均沿用既有2环境、horizon=8、2完成回合、1 PPO epoch、2局独立评估模式；max_steps仍为250。每组500联合步、32次PPO更新。正式配置仍是10环境、horizon=1500，未被修改。

| 真实 CLI 冒烟 | 训练回合 | 有效PPO样本总数 | 训练Return均值 | 训练Red存活均值 | 训练终止 | 独立评估 |
| --- | --- | --- | --- | --- | --- | --- |
| headon_xu_3v3 | 2 | 1066 | 17.408413 | 1.5 | 2 Timeout | 2局，0胜，Red均值0，Blue击落0 |
| headon_madsac_3v3 | 2 | 1214 | −11.980333 | 2.0 | 2 Timeout | 2局，0胜，Red均值0，Blue击落0 |

两组训练及独立评估 numerical_failure 均为0，PPO各指标有限，valid_samples 等于真实有效掩码样本数，死亡后的槽位按原规则排除；Reward总和均与分量一致。MADSAC每回合每Red槽平均累计分量：R1=−3.333333、R2=−8.333333、R3=+0.008000、R4=−0.321667。不同奖励的 Return 尺度不可直接比较；两回合不能推断性能提升。

另从**本轮新生成** MADSAC `episodes_000001.pt`，通过真实训练入口恢复到独立 resume 目录：继续原半Rollout到同一2回合预算。最终 checkpoint **全部内容逐值相等**，含Actor/Critic/优化器、RNG、worker环境与中心、计数和配置；独立评估除输出模型路径外逐值相等。未恢复历史2500回合模型训练。

历史原始 paper 的2500回合 final模型只读重放已有250步录像：Actor动作、状态、分配、Blue控制、五项Reward、击落事件与修改前归档逐值相等。历史模型 SHA256 前后同为 `8c2c368f11563a944f92167d823684607cb0f0289f943258a2b47fe48e92fe66`；同一只读审计还验证了原 approach_gated 的固定动作物理一致性。

验收输出位于 `outputs/headon_acceptance_20261024/`：acceptance.json、deterministic.json、historical_compatibility.json、三个控制台日志、cli及resume子目录内的完整训练/评估/模型记录。临时只读审计脚本位于tmp，不引入新的实验框架。

## 隔离与正式命令

两个新实验 family=headon_experiment、paper_experiment=false；论文计划仍仅37项，论文报告读取器排除该family、false标记和head_on模式。旧 main_3v3 / approach_gated 默认输出路径不变。

新实验默认根目录 outputs/headon_2500 或 outputs/headon_5000，子目录为 experiment/seed_N；两个实验及不同预算分开保存。已有训练日志或模型会拒绝覆盖。环境版本保持 v2.0.0，因为物理协议未变；完整 env_config 区分奖励、初始化及参数。恢复按现有严格配置等值检查，两种奖励之间、random/head_on之间以及半径或奖励参数不一致均在加载模型参数前拒绝。2500与5000亦不能跨预算直接恢复。

Ubuntu中，进入仓库后执行以下命令。没有在本轮执行它们：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/1_uav
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'

python -m algorithm.experiments train --experiment headon_xu_3v3 --seed 1 --episodes 2500 --root outputs/headon_2500
python -m algorithm.experiments train --experiment headon_madsac_3v3 --seed 1 --episodes 2500 --root outputs/headon_2500
```

每组训练结束自动对本组 final/latest 模型独立评估200局；保留现有训练超参数、worker种子、检查点策略和预算恢复限制。2500是半预算诊断，不是原论文完整5000回合结果。正式运行仍需选择未使用的输出目录。

## 比较指标

首先比较 main_3v3 与 headon_xu_3v3，观察初始化变化；再比较 headon_xu_3v3 与 headon_madsac_3v3，观察奖励变化。后两者固定训练种子、预算、相同位置/速度及全部非奖励配置。不要直接把 main_3v3→headon_madsac 的变化归因于单一奖励因素。

重点比较最终独立200局胜率、Red存活、Blue真实击落、Timeout/正常败局/数值异常，记录随机种子。训练曲线与独立评估分开。Reward诊断比较各自分量占比；MADSAC特别看R1真实战果、R2出界触发、R3引导和R4净姿态项；Xu看Posture是否仍占主导。现有轨迹/逐步记录可分析2000米真实攻击范围进入情况、锁定和击落、相对距离与双方轨迹；需要区分4000米奖励区与2000米武器区。没有为统计再增加奖励或改变策略输入。

本轮已具备运行两个正式新实验的配置和恢复条件；只实际运行了确定性测试、短程CUDA冒烟及历史模型只读验证。正式训练和正式200局统计仍待用户运行。

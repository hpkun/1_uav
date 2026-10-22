# MAPPO v2 协议与训练说明

## 网络与初始化

算法身份 `mappo-coop3d-v2.0.0`。环境仅通过 factory 创建，无算法内动力学/任务奖励副本。

共享 Actor：12→128 Tanh→128 Tanh→6，形成3维独立Beta，联合log probability/entropy按三维求和。参数保持softplus+1。正式默认 `initialization_mode=trim_centered`、`beta_concentration=16`、输出权重正交gain=.01：

```text
a_trim=[.5,1/3,1/8]
alpha=16*a_trim=[8,5.333333,2]
beta=16*(1-a_trim)=[8,10.666667,14]
bias=inverse_softplus(concat(alpha,beta)-1)
```

浓度须>8，保证最小alpha>1。小输出权重允许观测依赖而不破坏中心；不冻结任何层。legacy使用原PyTorch默认线性层初始化，仅用于受控诊断。随机采样仍有非零方差，确定性评估使用Beta均值；实际样本限制到[1e-6,1−1e-6]后重新计算对应log probability。

Critic：90→128 Tanh→128 Tanh→5，输出 `[batch,5]`。槽位与UAV ID固定。3 UAV仅使用前三项，后两项不参与损失；没有标量广播。每个有效UAV用自己的reward、V_i、return、advantage。PPO选择个体样本，同时将其ID用于gather对应Critic头。

## NaN-safe GAE 与损失

当前mask为m，后继mask为m'，真正终止D，超时Q：

```text
bootstrap = m & m' & ~D
continuation = bootstrap & ~Q
r=where(m,r,0); v=where(m,V_i,0); nv=where(bootstrap,V_i_next,0)
delta=r+gamma*nv-v
A=where(m,delta+gamma*lambda*where(continuation,A_next,0),0)
R=where(m,A+v,0)
```

不能用0×NaN屏蔽。输入、优势和return的有效部分须有限，否则报错。真正终止不自举，超时用真实末状态自举但切断递推；当前有效下一失效保留最终转移奖励。Actor先筛有效局部观测，Critic前60维的失效/不存在UAV槽位清零；不存在点槽位清零，有效点信息不可默默屏蔽。整局padding没有样本，不执行环境转移。

PPO ratio=`exp(new_logp-old_logp)`，clipped surrogate；有效优势标准化；Critic `.5*MSE`；熵系数.001。Adam两网络LR=3e-4、eps=1e-5，gamma=.954、lambda=.95、clip=.2，10epochs、minibatch128、梯度范数上限.5。无隐藏奖励、无gamma改为.99、无额外优化分支。gradient_norm记录裁剪前两个网络范数的合成值；approx_kl使用`mean((ratio−1)−log_ratio)`，clip_fraction记录超出PPO剪切区间的比例。

## 并行、严格预算与种子

默认10环境，horizon1500（Xu p.8表III）；单局上限仍为250步。完整rollout为15000个joint transitions，允许跨多局；旧checkpoint中保存的horizon原值继续用于恢复。`sampled_steps`是所有环境joint transitions之和，不是agent steps；25个joint transitions在3UAV场景最多对应75个体转移。

预算为绝对总量。最后rollout可缩短，最后vector step可只推进部分worker，其余worker不step、不reset、不计样本。buffer剩余槽位current mask=0。相同worker序列保持可恢复性。自动reset前后观测/90D状态分别保存在`transition_observations`/`transition_states`与`observations`/`states`，bootstrap只读前者。

worker i 第 e 局 seed=`training_seed+e*num_envs+i`，每个环境独立RNG。评估使用独立保留种子；训练碰到保留种子明确失败。CLI评估/录制检查种子是否已经用于训练。

## 配置、日志与保存

正式配置 `configs/mappo_coop3d.yaml`：评估间隔25000 sampled_steps、评估5局、保存间隔25000、控制台间隔2500。调度在rollout更新边界检查跨越的采样阈值，同一更新跨多个阈值只执行一次，最终预算必做评估和保存。

输出：`latest.pt`、`best.pt`、`run_config.json`、`training_metrics.jsonl`、`evaluation_history.csv`、`run_summary.json`。最佳按独立评估的成功率、完成率、平均回报字典序选择。summary分别评估best和latest并标注路径，录像由显式checkpoint指定，不互换标签。

日志三类：

- 累计：sampled_steps、PPO update、joint_episode_count。
- training：本rollout结束的真实训练回合的平均return、成功率、完成率、完成点数、长度、五项回合收益；没有完成回合时均值为null。
- rollout：本次有效UAV-step分母的五项奖励均值、clip rates、事件数、Beta动作均值/std、分配变化、当前目标距离均值、相同目标距离减少量（米）。

optimization记录loss、entropy、approx_kl、clip_fraction、gradient_norm。independent_evaluation单独存储，不混入训练回合统计。CSV的current表示该更新当时模型的独立评估，best选择由checkpoint和summary追溯。

完整checkpoint含模型、两个Adam、sampled_steps、update、最佳分数、环境/算法配置及版本、12/90/5维协议、所有Python/NumPy/Torch/ CUDA随机状态、worker状态/任务/分配/累计计数/局数、当前90D状态和episode统计累加器。恢复默认采用原配置，只允许提高命令行总预算；显式更改网络、环境、优化配置会被拒绝。旧60D身份在反序列化旧Python类前被拒绝，不保留旧包转发。只加载可信的本地checkpoint。

训练、评估和checkpoint审计强制CUDA，不CPU回退；环境和数学单元测试本身可用CPU。正式长训练命令仅写在README及报告，本轮未执行。

## 日志维护说明（不改变 v2 算法/Checkpoint 身份）

修复前的 `optimization.actor_loss` 错记为纯 PPO policy loss。新日志分别保存 `policy_loss` 与 `actor_total_loss=policy_loss−entropy_coef*entropy`，其中均值采用原有逐小批量等权统计；`actor_loss` 保留为真实 `actor_total_loss` 的兼容别名。原有历史 JSONL/CSV、模型和报告表格不重写，读取历史值时将旧 actor_loss 解释为 policy_loss，未记录的总损失显示 N/A。Beta 熵为微分熵，可以为负。

日志格式器只负责展示：普通训练进度为两行，训练回合数据缺失显示 N/A；独立评估使用 evaluate 返回值并明确显示样本数、完成点数/总数、BEST UPDATED/UNCHANGED；最终单独显示 best 与 latest 的指标和真实路径。steps/s 使用本次运行新增的 joint transitions / 本次运行墙钟时间，包含独立评估和保存耗时，不将恢复前的 sampled_steps 算入速度。

恢复预检先于目录写入、日志追加、模型覆盖与trainer启动。重复/倒序/晚于断点的日志、晚于断点的目标checkpoint或不匹配的best来源会明确拒绝；缺失/落后日志只提示缺口。预检不改文件、不消耗随机状态，不自动删除唯一历史记录。没有修改TrainConfig字段、checkpoint内容协议、GAE、PPO公式或Adam参数。

`python -m tools.report_training --run-dir <目录> --last-updates 10` 是纯标准库的只读日志工具，不加载模型、不调用CUDA；它与训练/模型审计的GPU要求不同。首10次与尾N次优化均值按update等权，Beta mean/std显示最后一个真实rollout的记录值，五项奖励使用日志内每有效UAV-step均值。样本不足时首尾窗口可重叠并明确提示。

诊断默认考察最近N个有效记录：至少3次独立评估成功率均为0、完成率未增长、目标距离逐次增长，或至少3个记录中≥80%超过 γ/速度裁剪率10%、KL 0.03、裁剪前梯度范数5.0。阈值是报告启发式，不是训练失败标准；缺失及非有限值不填0，也不输出为有效数字。该维护未提升算法版本，完整验证见training_log_maintenance.md。

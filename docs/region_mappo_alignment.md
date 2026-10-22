# Region MAPPO：论文明确参数、工程假设和自定义任务

2026-10-22。依据工作区两份原始PDF及已有审计文档，重新核对原文方法页和表格。以下X为Xu等2026，L为Li等2023，页码均按可见PDF实际页面。两文任务的交战部分未实现。本版本是**对齐公开通用动力学与列举网络/优化/采样参数的MAPPO基线**，不是APH-MAPPO完整方法或论文结果复现。

## A. 论文明确支持并采用的内容

|依据|原文定义|新实现及验证|
|---|---|---|
|X p.2–3式(1)–(2)|z向上3DOF位置/速度/路径角/航向ODE|`env/region_competition/environment.py`两队均直接调用现有`env.integrator.rk4`与`env.dynamics`，没有为Blue另写运动|
|X p.5式(11)、p.8表II|连续bank[-π,π]、切向[-1,2]、法向[0,8]|归一化3动作仿射映射与顺序保持，原始动作与物理控制均记录|
|X p.8§IV-A1|dt=.25、速度100–400、episode最多250步|新正式配置3v3/5v5均250；缩短episode仅用于明确标记Smoke|
|X p.8表III|Actor/Critic各两层128、Tanh、Beta、12/60/3尺寸|复用12D Actor，新60D Critic；输出头选择是B类假设，12维语义属于C类|
|X p.8表III|Actor/Critic LR=3e-4、Adam|新独立优化器分别设置，沿用已审计PPO更新|
|X p.8表III|γ=.954，λ=.95，clip=.2，entropy=.001|`configs/mappo_region.yaml`与RegionTrainConfig一致|
|X p.8表III|epochs10、mini128、并行10|正式配置一致；mini解释为有效个体样本，见B类|
|X p.8表III|每环境rollout1500、episode最大250|分别配置；1500步实采测试跨6个250步回合，不将episode改为1500|
|X p.7–8式(25)–(27)|有效性flag、有效样本损失、失效后不自举的GAE|原PPO/GAE数学复用；无效/不存在槽剔除；失效当步有效转移保留|
|X p.8表III|50episodes checkpoint|新配置50已完成联合episode阈值；在更新边界检查，跨阈值合并一次，因此实际存档episode可能超过50，见B类调度差异|

L p.4式(1)是NED、`dz=-v*sin(theta)`，与本版本选用X的z-up不同；不能把这当作本版符号错误。L p.13表1–2和式(23)采用速度/姿态期望增量及不同包线，不适用于本版三过载动作。L p.9–12式(16)–(22)/图6/算法1研究MADSAC双注意力Q Critic、replay和延迟更新；L p.15两层256、LR1e-4、γ=.99、batch1024不是本MAPPO参数出处。原文内容与现有源码分开核查，未拿README反推论文。

## B. 论文未明确公开的实现假设和协议差异

|项目|当前选择|论文证据边界及影响|
|---|---|---|
|Actor共享|全Red同构机共享一个12D网络|X表III未明确共享规则；不是获得作者原始代码的证据|
|Critic输出|共享干+固定5个个体价值头，按UAV ID gather|X p.8式(26)有V_i，但未给五头拓扑；不宣称原文就是此结构|
|Beta参数|三维独立，α/β=softplus(output)+1；联合logp/entropy只在动作维求和|X只给Beta家族；α/β>1限制单峰族；微分熵可负（X p.11§IV-D2明确解释）|
|初始化|trim中心(.5,1/3,1/8)，浓度16，输出正交gain=.01|X未给初始化；非线性动力学下均值配平不等于训练采样期望配平，不声称已解决探索问题|
|取样数值|Beta sample夹到[1e-6,1−1e-6]后重算对应logp；评价取均值|X未给这些细节；夹紧边界的密度是数值近似，未宣称完全一致|
|PPO与熵|复用原clipped surrogate与独立Adam，不重写更新数学|X p.8式(26)称熵正则PPO surrogate，未单独展开ratio/min/clip；当前已知答案/原回归测试约束具体实现|
|优势和价值|有效全局池优势总体std标准化；½MSE；mini128指128个有效个体；尾mini保留|X式(26)写无½的平方误差并逐个体归一化；mini单位/优势处理未明确。失效时共享池个体权重会不同|
|数值优化|Adam eps1e-5、grad norm .5、std epsilon1e-8|工程默认，非表III公开数值|
|积分保护|g9.81、gamma±85°、epsilon1e-6、角度wrap、区域交点1e-12分辨率|X未给RK4/这些保护细节；原模型没有受限gamma包线的物理验证，具体见环境规范|
|截止价值|episode截止truncated但有限时域V_next=0；rollout尾用真实后继|X p.8式(27)未说明超时δ细节；这是新任务明确的有限目标定义，旧巡检自举规则保留|
|种子|worker独立RNG，每局base+episode*E+i，独立评价保留seed|X p.8给worker seed1000s+r，未给此逐局重置历程；当前不是原作者随机数轨迹|
|存档和选模|50已结束episode阈值在update边界合并；评价每30000joint步及最终；best按平均相对得分/Red胜率/return字典序|update粒度可越过checkpoint/评价阈值；X未给当前选模规则。反复选best的seed是验证集，新最终评价须用未用过的新seed|
|预算/统计|正式超参数对齐；未执行论文5000episodes×5独立训练seed、最终200局|X p.8–9的研究预算与统计尚未复现；本轮只有单元测试和短CUDA验证，不能声称论文胜率|

默认E10、H1500，每完整update有15000联合转移。无失效3v3=45000有效个体样本，10epochs产生3520对Actor/Critic优化步；5v5=75000样本，10epochs产生5860对优化步。`sampled_steps`始终是joint transitions，不能将它与个体样本、episode或optimizer_steps混用。最后预算可部分推进worker，padding不计样本。

只在原Trainer增加默认仍90D的状态清理/维度与优势估计钩子，在原Vector增加默认仍巡检worker/accumulator的工厂钩子。新adapter覆写这些接口，PPO loss、ratio、熵、Adam步骤、梯度限幅和旧任务公式未改变。旧90D checkpoint仍由原协议读取，新60D checkpoint使用独立身份；双向误载测试明确拒绝。

## C. 完全属于本项目的新任务

动态球区、区域轨迹/半径/初始镜像布局、round-robin区域参考、两队exclusive覆盖计分/contested规则、三种Blue规则、12维新语义、剩余时间输入、60D压缩观测局限、五项新Reward、数值失效提前结束、回放与离线审计均是项目设计。

X p.5式(12)–(14)、p.7式(25)的目标飞行状态/三维视线夹角不等于当前区域相对xyz、sin/cos航向和剩余时间。X p.4–6式(4)–(10)/算法1的原任务分配不在这里实现；区域ID不是攻击对象。X p.6–7式(15)–(20)及L p.14–15式(25)的任务奖励不在这里实现，不能借相同“5项”或网络维数给出原任务复现的标签。

本版60维不追加hidden状态，因此尚非完整环境Markov状态；比分历史和区域速度等缺失会限制学习，文档明确披露，未为展示成功率改任务或扩输入。详见 [region_competition_spec.md](region_competition_spec.md)。运行、验证结果与完整Ubuntu命令见 [region_acceptance.md](region_acceptance.md)。

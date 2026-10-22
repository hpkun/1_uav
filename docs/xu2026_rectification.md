# Xu 2026 动力学与独立 MAPPO 协议整改（2026-10-22）

## 实际交付与六个层次的结论

本轮修改了实际运行路径，不只增加参考常量。`Cooperative3DEnv → integrator.rk4 → paper_dynamics.rk4_step/rhs` 已采用同一套无γ裁剪的式(1)/(2)计算；独立 `ProtocolMAPPOTrainer` 可以对外部提供的12/60数组执行CUDA PPO更新、保存和完整恢复优化状态。没有接入自主攻击对象分配、交战判定、毁伤或原论文任务Reward；没有新增替代任务，没有长训练。

|层次|完成的代码行为|不能据此声称的内容|
|---|---|---|
|数学公式|实际运行RHS复用Xu p.2式(1)、p.3式(2)，不裁剪γ、不替换分母；复用经典RK4/通用GAE；独立协议按p.8式(26)逐个体归一化、无½的MSE|作者使用了RK4、eps=1e−6或本项目失败处置；式(27)未公开的时间截止语义已确定|
|网络结构|Actor 12→128Tanh→128Tanh→三对Beta参数；独立Critic 60→128Tanh→128Tanh→五个个体价值头；旧90D Critic保留|五头、共享参数、softplus+1及trim初始化就是作者实现|
|观测语义|式(25)原始字段顺序、单位、flag校验及显式五槽打包可执行，未做归一化；90D输入直接拒绝|现有巡检数据就是合法的原任务观测，或已解决式(14)全部关系到60维的压缩细节|
|Reward|原a–e及式(20)继续完成文献核对；独立协议不计算/导入任务Reward|巡检五项等于原论文五项，或原论文Reward已经可以端到端计算|
|环境任务|旧巡检任务的点、分配、Reward与事件规则保留，实际动力学/γ特征版本改变|已实现论文对抗任务；没有区域竞争或新巡检任务|
|完整训练流程|巡检CLI可在新动力学下训练；独立12/60更新器有数组接口、mask、Adam及checkpoint|已有可运行的原论文训练环境、5000局×5seed或200局最终复现实验|

主要依据为仓库原始 Xu PDF，DOI 10.1109/TAES.2026.3734384；p.n指PDF页码。重新直接读取p.2/3/7/8，延用上一轮已核验的p.4–9公式和表格。完整12维逐索引及Reward文献表保存在 [严格审计](xu2026_strict_audit.md) 第3/4节；该报告的旧运行状态已由本文更新，不能把其历史兼容结论当作当前状态。

## 1. 85°已从实际路径消失

|位置|修改前|修改后/证据|
|---|---|---|
|`env/dynamics.py:derivatives`|独立RHS，γ中间裁剪，替换v/cosγ分母|仅委托`paper_dynamics.rhs`，六式不重复维护|
|`env/integrator.py:rk4`|k1–k4间接使用裁剪RHS；末态再投影γ|调用共享`rk4_step`，无任何γ投影；保留末态速度裁剪和ψ表示环绕|
|`env/config.py`|`gamma_limit=85°`|字段删除，不用89°等替代；旧`gamma_limit`参数明确拒绝|
|`env/scenario.py`|reset拒绝超出±85°|只检查速度范围、有限性和角坐标分母定义域；不设新的角度包线|
|`env/observation.py`|巡检γ/85°|巡检γ/π，无裁剪；这是明确的新工程特征缩放，**不是式(25)作者归一化**。独立原始字段协议不缩放|
|`env/cooperative_env.py`|非有限积分失效；闲置配平隐含cosγ≥0|所有积分定义域失败统一记录reason/detail和计数，保留上个有限状态、失效；cosγ<0时闲置配平用μ=π、ζ̂=abs(cosγ)，避免负法向载荷，仍是原式，不夹紧γ|

六式严格为：

```text
dx = v cosγ cosψ       dy = v cosγ sinψ       dz = v sinγ       [p.2 Eq.(1)]
dv = g(ζ − sinγ)
dγ = (g/v)(ζ̂ cosμ − cosγ)
dψ = g ζ̂ sinμ / (v cosγ)                                    [p.3 Eq.(2)]
```

控制顺序 `[μ,ζ,ζ̂]` 来自p.5式(11)，范围为p.8表II的 `[-π,π],[-1,2],[0,8]`；`physical_actions`的线性映射未改动。状态编码 `[x,y,z,v,γ,ψ]` 单位m/m/s/rad。p.3图1/文字是z向上；正γ使z增加，正负μ的航向响应按原分母符号计算，不做abs(cosγ)。测试中的100°只验证无额外角阈值和分母符号，不代表作者规定了这种飞行包线。

### 剩余约束与数值假设

- **论文明确**：p.8 §IV-A1速度100–400m/s、dt=.25s、单局250步；表II控制范围。重力g的数值9.81是配置假设，p.3只定义g含义。
- **定义域检查**：输入有限；v≤eps或abs(cosγ)≤eps拒绝；RK4阶段/末态跨过`π/2+kπ`的坐标极点拒绝。eps=1e−6是数值阈值，不是另一条角度飞行包线。负v不通过abs“修好”，cosγ不改符号。
- **工程约束实现**：共享RK4本身不做速度投影；实际环境在成功RK4末态把v投影到配置速度范围，ψ wrap至[-π,π)。这是作者未公开的速度范围实施/表示选择，不能称为原论文积分方法。γ没有wrap或clip；无高度、地面或世界坐标围栏。
- **失败处理假设**：返回原有限状态、accepted=False、明确reason；环境将对应机置为无效并记录`integration_failures`和`failures[].detail`，不新增失败Reward。`gamma_clips`仅为历史工具保留的恒零列。观测/集中状态编码若无法保持有限float32，直接报错，不把NaN/Inf交给网络。
- **保留的旧任务规则**：场景采样范围、300m巡检到达、100m安全参考、闲置配平、故障注入仍是旧任务规则，未变成论文的物理包线。

## 2. 版本与历史数据

环境身份变为 **`coop3d-v3.0.0`**；巡检算法身份 **`mappo-coop3d-v3.0.0`**。独立数组算法身份 **`mappo-array-12x60-v1.0.0`**。前者虽保留90D/五头/½MSE/池化损失，但动力学和γ特征语义已改变，故不能恢复旧v2模型继续训练或作新动力学评估。

`read_checkpoint`在反序列化前检查新环境/算法身份；旧v1/v2、Region及独立数组模型不兼容。独立模型通过ZIP内JSON协议身份先行检查，再校验60D网络/五个独立头、配置和完整状态；不能加载90D模型。两个保存/恢复API也检查trainer家族，防止把新损失协议装进旧格式。没有自动截断、权重迁移或静默升级。

巡检构造器拒绝独立协议配置，巡检评估器也拒绝独立数组trainer；不能因为Actor输入都为12维就把新Actor悄悄用于巡检评估。独立协议没有任务适配器，不能调用其collect来训练巡检或交战任务。

原始PDF、模型和历史输出均不改写。历史trajectory.json仍可用原有只读回放/轨迹工具查看，但这些轨迹不能标成v3生成。没有承诺在新代码中重新执行旧checkpoint的旧动力学；它们需在对应旧版本代码中运行。

## 3. 12D Actor与真正独立的60D Critic

|实现|输入/输出|来源与边界|
|---|---|---|
|`Actor`|12→128Tanh→128Tanh→6参数，3个独立Beta|隐藏层、12D、Beta家族/3动作对应p.8表III；独立性、softplus+1、最后六参数头、初始化均工程假设|
|`Critic`|90→128Tanh→128Tanh→5|仅巡检训练器使用，继续保留完整点表；不称为论文60D|
|`ProtocolCritic`|60→128Tanh→128Tanh→5个独立Linear(128,1)|60D和隐藏层对应表III；共享干、五个独立价值头是显式工程假设|
|`pack_centralized_slots`|`[...,N≤5,12]`和同形mask→`[...,60]`|按调用者给定UAV ID顺序打包，补零至5槽；拒绝90D、超5槽、mask/flag不符、非有限/非零无效槽。不接收90D再截取|
|`ProtocolMAPPOTrainer.update`|外部Rollout的[T,E,N,12]与[T,E,60]|有效Actor槽必须与显式集中状态对应槽完全一致；价值按固定UAV ID gather；训练前丢弃无效槽垃圾/NaN，不能成为损失样本|
|`sample_actions`|raw12D、mask→动作及对应joint logp|采样/均值→eps裁剪→重算同一个返回动作的logp，无效槽返回0；没有模拟器动作覆盖或Reward接口|

Beta初始化仍可配置为`trim_centered/legacy`，默认concentration16、trim均值(.5,1/3,1/8)、输出gain=.01，均不标为作者设置。α/β>1限制分布族；eps动作裁剪也是工程近似。本轮不更换初始化。Actor和Critic分别持有Adam，没有共享Parameter。

**逐槽零填充是本项目明确的接口约定。** 原文p.5式(14)的完整关系集合如何映射为p.8表III的60维仍有公开缺口；不能把“能输入60个数”等同于语义复现。

### 原始字段与数据来源

以下顺序完整保留，全部字段来自 **p.7式(25)**；几何定义来自 **p.3式(3)及其后文字**。无归一化接口，原始单位如下。

|索引|变量/单位|合法来源状态|
|---|---|---|
|0|Oᵢⱼ̂ / 1|需任务提供合法槽位/有效性；新协议验证0/1及与当前mask一致，不把has_task当原flag|
|1|vᵢ / m/s|本机通用飞行状态下标3可得|
|2|γᵢ / rad|本机下标4可得|
|3|ψᵢ / rad|本机下标5可得|
|4|αᵢ / rad|有合法指定对象时，可由本机速度与双方位置做纯几何计算；本轮未实现该对象选择/观测链|
|5|zᵢ / m|本机下标2可得|
|6|vⱼ̂ / m/s|需要合法指定的另一UAV飞行状态，不是静态点|
|7|γⱼ̂ / rad|同上，下标4|
|8|ψⱼ̂ / rad|同上，下标5|
|9|αⱼ̂ / rad|需要另一机速度与pⱼ̂−pᵢ；不可自行反向视线。对象语义/链路未实现|
|10|zⱼ̂ / m|需要合法指定另一机下标2|
|11|dᵢⱼ̂ / m|有双方位置即可数学计算三维距离；原任务对象身份未实现|

已有巡检环境不能提供这组完整合法来源，因此独立更新器**没有collect环境适配器**，调用会明确报错；只能对外部显式提供并满足协议的数组执行通用优化。不能用当前巡检12D输入冒充式(25)。原文未明确全部归一化；p.4式(6)的速度归一化属于奖励，不是观测归一化证据。

## 4. PPO/GAE整改与人工已知答案

通用PPO/GAE保留一套计算。巡检`update`仍使用90D、有效个体统一池化、½MSE；独立`update`选择60D、逐个体归一化、MSE系数1。有效路径的旧优化结果继续与冻结参考实现逐参数/Adam状态零容差对照，不改旧损失语义。

独立协议对每个小批内的个体i分别计算 `Lπ,i=mean_valid_i(−min(ρA,clip(ρ)A)−cH)` 和 `LV,i=mean_valid_i((V_i−R_i)²)`，对应 **p.8式(26)**。因为本项目选用共享Actor和Critic干，执行的总损失为各个有样本个体Lᵢ的**求和**；这种跨个体共享参数聚合不是论文公开细节。没有样本的头用`grad=None`跳过Adam，包括已有动量与step计数；不能用零梯度代替跳过。

优势归一化在独立协议中按每个体的有效rollout样本计算，受`normalize_advantages`配置控制；巡检仍保持全池标准化。两者都是明确假设，不编造作者标准化方式。PPO ratio和熵在三动作维求和，正/负优势使用同一个min clipped surrogate。

人工例子（`tests/test_xu_protocol_update.py`直接调用真实更新函数、零LR隔离测量）：

- UAV0两样本：A=(1,3)，ρ=(1.4,.7)，裁剪后surrogate=(1.2,2.1)，均值1.65；UAV1一样本：A=−2、ρ=1.1，surrogate=−2.2。新policy loss=−(1.65−2.2)=**.55**。
- 三熵(.1,.3,.9)对应个体均值之和 **1.1**，测试系数.1，actor总损失 **.44**。测试系数只用于手算用例，不改变正式默认.001。
- 价值平方误差UAV0=(1,9)、UAV1=(4)，新loss=mean(1,9)+4=**9**；旧池化½MSE=**7/3**，差异确实存在而非换名。新价值头梯度为 **(−4,4,0,0,0)**，策略logp参数梯度为 **(0,−1.05,2.2)**。
- 补测某头已存在Adam动量、下个batch没有该机样本时，其参数、momentum、step完全不变；共享干仍由有样本个体更新。这只保证当前显式共享拓扑的处理，不冒称作者也采用此拓扑。

GAE仍使用 `bootstrap=current_mask & next_mask & ~terminated`，carry再排除truncated。真正终止不自举，人工时间截断用真实末状态自举但不把reset回合接入carry，rollout末尾可自举；无效值用where排除而非0×NaN。边界形状必须[T,E]且bool，样本必须[T,E,N]，避免错误广播。p.8式(27)的δ行没有显式b、未完整交代时间截止语义；这些完整终止/截断选择继续标为工程假设。

1500/250回归运行一个真实worker、配平采样、不做PPO更新：1500个joint transitions，4500有效UAV样本，**六个250步截断回合**。在边界249/499/749/999/1249/1499逐一检查truncated。另对整个1500数组用手算δ=1+.954×2−1=1.908检查每个边界无跨局carry，并检查每段首项几何级数答案。既有2步实际后继/auto-reset分离、真终止、故障mask和partial worker测试继续保留。

旧巡检的idle仍会把候选动作覆盖成配平控制，这是既有任务规则；存储的随机Beta动作并非该步实际物理控制，不能宣称该链路与论文任务完全一致。独立数组接口只保证返回动作及其logp对应同一数值；没有实际环境执行适配器可供声称端到端一致。

## 5. 其他公开协议逐项核对

|论文依据|当前实现及修改结果|是否真正对齐/未确认|
|---|---|---|
|p.8表III Actor/Critic128,128 Tanh；12/60/3|独立网络如第3节；旧90D单独保留|独立维数/隐藏层对齐，完整观测语义未对齐|
|p.8表II三动作范围|实际运行保持μ/ζ/ζ̂线性映射，未换序|范围对齐；Beta→物理线性映射是工程实现|
|p.8表III LR3e−4/3e−4、γ=.954、λ=.95、clip=.2、entropy=.001|两种配置均保留|明确数值对齐；Adam eps1e−5、grad clip.5未公开|
|p.8表III epochs10、mini128、E10、H1500、ep250|默认同值；完整采样15000 joint transitions；有效个体45000(3机)或75000(5机)|数值对齐；mini以个体计数，作者计数单位未明确。无失效时每update为3520/5860次actor及对应critic优化（10×ceil(样本数/128)）；预算截断另计|
|p.8式(26)|独立协议无½、逐个体小批有效数归一化；旧巡检½/池化未动|独立单个体损失定义对齐；共享聚合/优势标准化是额外假设|
|p.8式(27)|复用已验证GAE、真实后继、终止/截断/rollout尾分开|mask原则对齐；终止δ和有限时域timeout的具体语义仍有公开缺口|
|p.8正文seed1000s+r|`ProtocolTrainConfig.worker_seeds()`给出该规则，10worker默认1000..1009；巡检实际worker仍每局base+episode×E+r重置|独立规则函数对齐，**未接入独立采样器**；巡检轨迹种子历程仍不同。torch网络seed是独立工程配置|
|p.8表III每50episodes存档|独立配置明确50，独立保存/恢复API可执行；巡检runner仍按25000joint步的更新边界|**未完成按50局自动调度**：独立协议无task runner，不把配置字段当已运行；巡检不一致|
|p.9 §IV-B1每final checkpoint 200局；p.8各规模5000局/5seed|独立配置evaluation_episodes=200；巡检默认5个固定验证seed、best字典序规则；本轮只跑回归|**未完成原任务训练/最终评估**；独立配置不代表已有评估器。固定验证不充作独立最终测试|
|论文未公开Beta变换/初始化、头拓扑、归一化|保留并标明配置假设；五头独立参数用来实现缺样本跳过Adam|未编造作者实现；这些选择仍需独立实验或作者代码确认|
|论文未公开checkpoint格式|12/90巡检与12/60数组完全不同身份/API，完整RNG/两Adam恢复|工程正确性已回归，不叫论文原始存档协议|

独立更新器只统计本次有效个体样本和optimizer_steps，**不从一个可能含padding的数组猜测已执行环境步数**；没有collector时sampled_steps/episode计数不会自动增加。巡检collector继续统计全部worker真正执行的joint transitions，与有效个体样本数分开。

## 6. Reward审计与尚未实现的部分

原文顺序与公式没有改动：

|原顺序|文献数学内容/依据|参数与缺口|实现状态|
|---|---|---|---|
|a Attack Advantage|p.6 a引用p.4式(5)，ηᵢⱼ̂−ηⱼ̂ᵢ；条件在式(4)|η=5(p.8)，几何/有效对象/时序所需状态；不是Δ距离|仅文献审计，未接入|
|b Posture Advantage|p.6 b引用p.4式(6)：近距−1/d，中距cosαᵢ+cosαⱼ̂，远距归一化本机速度×[cosαᵢ+max(0,cosθ)]|τmin100、τmax2000、v100–400(p.8)；d=0处理未公开，未发明clip|仅文献审计，未接入|
|c Allied Proximity/Shared Exposure|p.6式(15)威胁indicator、(16)共同暴露，(17)近距项与共同暴露之和/(N−1)|N=1、N的失效细则未完全说明；active状态每步评价(p.6)|巡检仅保留旧近距项，不能叫完整原项；独立协议未实现|
|d Shootdown|p.6式(18)，连续条件时长>阈值时r₀|r₀未公开；p.8两步/.5s与严格>的离散边界需澄清|未实现，不用巡检+10/首次访问事件代替|
|e Episodic|p.7式(19)，M̂=0时N̂e₁−floor(T/e₂)|e₁20/e₂25(p.8)；有效性/同步细节未完整公开|未实现；巡检20−floor(T/25)的条件与N̂系数都不同|
|合成|p.7式(20)按上述五项相加|没有区域占用、Δ距离、−.01常数或新增总Reward clip|原任务Reward整体**尚未实现**|

巡检`env/reward.py`继续保留progress/completion/separation/efficiency/terminal，既有规则/系数未改，且只服务巡检训练入口。独立`ProtocolMAPPOTrainer`不计算奖励，不隐式调用此文件；测试输入是人工数值，不是仿造的任务奖励。涉及自主攻击/资格/毁伤的机制依本轮范围不接入。需要合法任务事件及未公开参数才能完成式(20)，不能用替代Reward填空。

## 7. 验证及可复制命令

本机WSL再次报`Wsl/EnumerateDistros/Service/E_ACCESSDENIED`，按AGENTS使用Windows `uav`。Python为`D:\conda_envs\envs_dirs\uav\python.exe`，CUDA=True，RTX5060 Laptop GPU；无CPU训练/模型审计回退。

执行记录：

- 定向测试（动力学、873转移回归、旧优化零容差参考、旧PPO手算）：35 passed in 8.35s。
- 首轮全量：187 passed in 30.82s；补充已有Adam动量的跳过测试、有限观测保护及GAE边界形状保护后189 passed in 31.34s；补测积分失败传到collector指标且采样张量全部有限后，**全量190 passed in 32.70s**。`python -m compileall -q env algorithm tools tests`和`git diff --check`通过。
- 最后增加巡检构造器/评估器与独立数组协议互斥检查后，定向重跑`test_xu_protocol_update.py`、`test_mappo.py`、`test_entrypoints.py`：**23 passed in 9.37s**；再次编译与差异检查通过。
- 原fixture数据不改。873转移均没有旧γ限幅；测试只把已知γ观测分母改为π、补充新的恒零诊断字段，并移除退役配置键以比较新版本。原状态/事件/Reward基准和容差保持不变，未用新运行重造历史轨迹。
- 没有执行正式长训练或用户200000步模型评估；唯一较长的采样用例是必要的1500/250边界测试，单worker、不更新参数。其余CUDA更新为极短人工/回归用例。
- 128份既有论文/模型/输出SHA-256均一致；19份真实历史checkpoint在两个新加载入口均于反序列化前拒绝（将torch.load替换为禁止调用进行验证）。运行源码无gamma_limit引用，无Region导入，`tests/fixtures`无修改。复核记录与JUnit XML位于`tmp/xu_rectification/`。

Ubuntu在仓库根目录复制：

```bash
conda activate uav
python -c 'import sys,torch; print(sys.executable); assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name())'
python -m pytest -q
python -m compileall -q env algorithm tools tests
```

定向验证本轮新增接口及边界：

```bash
conda activate uav
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
python -m pytest -q tests/test_runtime_unclipped.py tests/test_xu_protocol_update.py tests/test_xu_paper_reference.py tests/test_paper_math_audit.py
```

报告、代码和测试结果应按以上六层分别解读，不能笼统称为“论文对齐完成”。

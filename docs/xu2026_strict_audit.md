# Xu 2026 严格核对与 Region 删除记录（2026-10-22）

## 结论与实施边界

RegionCompetition 的环境、算法、三份配置、两个工具、两份测试和三份专属文档已删除，其公共适配钩子已撤销。没有新增替代任务，没有运行正式训练。所有既有 outputs、模型和两份原始 PDF 保留；核验方式及结果见文末。

论文依据仅为仓库内 Xu 等《Hybrid Advantage-Priority Hungarian Assignment MAPPO for Multi-UAV Cooperative Air Combat Decision-Making》，DOI **10.1109/TAES.2026.3734384**，作者接受稿，共 15 页。以下 p.n 指 PDF 第 n 页，与页脚页码一致。已直接读取原文相关页，并检查动力学、观测、Reward、参数表的页面图像；不把 README 或历史代码当作论文证据。

**当前仍不是原论文完整对抗环境复现。** 保留的可运行环境是历史 Coop3D 巡检环境，Actor 观测语义、90D Critic 和 Reward 与论文不同。本轮新增的 `env/paper_dynamics.py` 仅作独立数学核验；`algorithm/mappo/paper_protocol.py` 仅保存原始观测字段、校验器及 Table III 常量，均未接入 environment factory、训练、对象选择、交战事件或奖励链路。

需明确一项未完全满足的要求：**历史 Coop3D 中的 ±85° 规则仍可执行，未全局删除。** 它属于 C 类额外规则，不是必要的数值防护。为保留用户要求的旧实现及历史 checkpoint 仿真语义，本轮没有用新方程层替换其运行路径；它在独立论文数学路径中完全停用。因而不能宣称“整个保留环境的 C 类规则已全部清除”。源码、环境规范和本报告均标明该边界。

## 1. 删除与保留清单

|范围|实际删除内容|
|---|---|
|`env/region_competition/` 全目录|`__init__.py`, `config.py`, `environment.py`, `observation.py`, `policies.py`, `regions.py`, `reward.py`, `scenario.py` 及缓存|
|`algorithm/region_mappo/` 全目录|`__init__.py`, `__main__.py`, `checkpoint.py`, `config.py`, `evaluation.py`, `metrics.py`, `runner.py`, `trainer.py`, `vector_env.py` 及缓存|
|配置|`configs/region_3v3.yaml`, `configs/region_5v5.yaml`, `configs/mappo_region.yaml`|
|工具|`tools/audit_region_trajectory.py`, `tools/region_replay.py` 及对应缓存|
|测试|`tests/test_region_environment.py`, `tests/test_region_mappo.py` 及对应缓存|
|专属说明|`docs/region_acceptance.md`, `docs/region_competition_spec.md`, `docs/region_mappo_alignment.md`|
|临时入口和配置|`tmp/region_dev/ten_worker_smoke.py`，该目录的 `3v3_env.yaml`, `3v3_train.yaml`, `5v5_env.yaml`, `5v5_train.yaml`，`ubuntu-0.sh` 至 `ubuntu-4.sh`|
|公共代码中仅服务于 Region 的钩子|`algorithm/common/vector_env.py` 的可替换 worker/tracker；`algorithm/mappo/trainer.py` 的可替换状态维数、状态清理和 GAE 方法。恢复原有 Coop3D 直接调用，不改变 PPO/GAE 数学|

共删除 **27 个已跟踪业务文件**，另清理上述 10 个临时入口/配置及相关缓存。`tmp/xu_strict_audit/deleted_manifest.json` 保存删除前路径和散列。保留的 `outputs/region_*` 是历史实验数据，不是当前运行入口；不能再用已删除的 Region 模块恢复这些模型，不做迁移或改写。

实际保留模块：

- 环境：`env/{__init__,config,models,scenario,assignment,dynamics,integrator,observation,reward,cooperative_env,factory}.py`；旧三种规模 `3U3P/5U5P/3U5P`，配置 `configs/coop3d_*.yaml`。
- 算法：`algorithm/mappo/{config,networks,trainer,runner,evaluation}.py`；`algorithm/common/{checkpoint,metrics,protocol,runtime,serialization,vector_env,progress,resume_logs}.py`；入口 `algorithm/train_mappo.py`, `algorithm/evaluate_mappo.py`；配置 `configs/mappo_coop3d.yaml`。
- 旧导航、记录、回放、轨迹审计及相关测试继续保留。新增两份纯数学/协议模块及 `tests/test_xu_paper_reference.py`，不新增任务或训练入口。
- 原始 Xu/Li PDF、历史模型、JSON/CSV/XML/HTML 实验输出全部保留。

## 2. 三自由度动力学：逐公式与参数

### 2.1 状态、坐标及控制

状态存储协议为 `[x,y,z,v,γ,ψ]`，单位 `[m,m,m,m/s,rad,rad]`。原文 p.2 式(1)、p.3 式(2)/图1定义这六个量，但没有逐字给出这一六维数组的存储顺序；顺序是显式工程编码。世界坐标 z 向上，γ 为速度与水平面的夹角，ψ 为水平速度投影相对 +x 的航向角（p.3 图1及式(2)前文字）。正 γ 对应上升；在 v>0、cosγ>0 时正 sinμ 对应 ψ 增大，这是方程符号推论，不是机体滚转/俯仰角定义。

|论文依据|物理量/公式|当前历史代码|独立论文数学层|结论|
|---|---|---|---|---|
|p.2 式(1)|ẋ=v cosγ cosψ|`env/dynamics.py:derivatives` 同项，但先 clip γ|`paper_dynamics.rhs` 原式|内部无夹紧区域对齐；大 γ 历史实现不同|
|p.2 式(1)|ẏ=v cosγ sinψ|同上|原式|同上|
|p.2 式(1)|ż=v sinγ|同上|原式|同上|
|p.3 式(2)|v̇=g(ζ−sinγ)|同项，使用被 clip 的 γ|原式|同上|
|p.3 式(2)|γ̇=(g/v)(ζ̂ cosμ−cosγ)|`max(abs(v),eps)` 且 clip γ|原式；非法分母显式拒绝|历史分母替代不是论文方程|
|p.3 式(2)|ψ̇=g ζ̂ sinμ/(v cosγ)|替代 v、cosγ 分母且 clip γ|原式；奇点拒绝|历史极限附近不对齐|
|p.5 式(11)，p.8 表II|μ：倾侧角，[-π,π] rad；ζ：切向载荷，[-1,2]；ζ̂：法向载荷，[0,8]；后两项无量纲|`physical_actions(a)=[2πa₀−π,3a₁−1,8a₂]`|仅接收这个次序的物理控制，越界拒绝|范围/次序已对齐；线性 Beta→物理映射为工程选择|

注意 p.7 表I的列顺序是 **μ,ζ̂,ζ**，不等于式(11)的动作顺序；不能据表I交换策略动作轴。g 在 p.3 被定义为重力加速度，未明确给出 9.81 数值。独立数学层要求调用者显式传入 gravity，不宣称测试使用的 9.81 为作者参数。

### 2.2 A/B/C 分类和数值行为

|类别|条目|论文依据|本轮处理及影响|
|---|---|---|---|
|A 论文明确|式(1)/(2)，控制定义与表II范围|p.2–3、p.5 式(11)、p.8 表II|独立 rhs 原式保留；不将 μ/ζ/ζ̂ 改成姿态增量|
|A 论文明确|dt=.25s，v∈[100,400]m/s，单局≤250决策步，无界三维空间|p.8 §IV-A1|历史配置相同。独立 RK4 返回 `within_published_speed_range`，不声称其无投影轨迹自动满足速度约束|
|B 数值必需的有限性处理|输入形状/NaN/Inf 检查、奇点拒绝、失败保留上一个有限输入|论文没有规定实现；计算 1/v、1/cosγ 的客观定义域需要处理|独立数学层拒绝 v≤eps 或 abs(cosγ)≤eps，不替换分母；跨越角度坐标极点的 RK4 阶段也拒绝。返回 accepted=False 和 reason，无 UAV 失效/惩罚事件|
|B 的具体工程假设|eps=1e-6、经典 RK4、失败步不推进|原文未公开|明确不是作者求解器、步长自适应策略或物理约束。eps 是拒绝阈值，不是飞行包线|
|工程表示/约束实施选择|末状态速度投影到[100,400]、ψ wrap 到[-π,π)|论文给速度范围，未给投影算法或 wrap 区间|历史积分器保留；独立数学层两者均不做，以免把投影当作原方程。其 endpoint 可超速，必须查看速度范围标志，不能直接作为完整环境积分器|
|C 缺乏依据|γ 的±85°中间阶段裁剪及末状态投影|论文无此包线|独立论文数学路径删除/停用；86°测试不夹紧。**历史 `dynamics.py/integrator.py` 仍保留，仅用于旧 Coop3D 兼容，未完成全局停用**|
|C 缺乏依据|`max(abs(v),eps)`、`max(cosγ,eps)` 继续生成有限导数|论文无此替代|独立数学路径停用。负速度不能通过 abs 转成正分母；cosγ 不改符号|
|C 旧任务规则|300m 点到达、固定采样范围、无任务配平覆盖、故障注入、巡检分配|不是论文动力学约束；p.3/8为另一研究任务|留在历史巡检环境，不进入独立论文数学层；没有新增地面、天花板或区域边界|

RK4 明确采用 k₁=f(y), k₂=f(y+h k₁/2), k₃=f(y+h k₂/2), k₄=f(y+h k₃)，y′=y+h(k₁+2k₂+2k₃+k₄)/6。历史积分器的阶段组合正确，但阶段 γ 已被投影，故在边界附近不是原光滑 ODE 的纯 RK4。独立版本全部阶段使用未裁剪原式，并检查输出；不跨过奇点假装有限。

## 3. 12 维观测逐索引核对

原文 p.5 式(12)为 11 个量，p.7 式(25)在首位加入 Oᵢⱼ̂，成为 12 维。下表所有字段的直接来源均为 **p.7 式(25)**；几何定义见 **p.3 式(3)及其后文字**。ĵ 指被分配的目标 UAV，不是区域或航路点。

掩码协议：索引0是二元槽位 flag；p.7–8另用当前友机有效性 Ôᵗᵢ,ₚ 控制传播、分配、梯度与零槽。不能把这些 flag 改释为“有无区域/剩余驻留时间”。论文没有把双旗各种边界情况完整展开；不擅自给未分配对象生成规则。下表“零槽”仅指 p.8 明确要求的无效友机槽整体置零。

|索引|原变量|物理含义|单位|有效性掩码|原文归一化|当前 `env/observation.py:observe` 的同索引|判断|
|---|---|---|---|---|---|---|---|
|0|Oᵢⱼ̂|二元观测槽 flag|1|flag自身；无效零槽|无缩放，原值0/1|active|有效性原则相近，双旗完整映射未公开|
|1|vᵢ|本机速度|m/s|无效零槽|未说明|`(v−100)/300`|物理量同，归一化非论文已知方法|
|2|γᵢ|本机飞行路径角|rad|无效零槽|未说明|`γ/gamma_limit`|85°分母无论文依据|
|3|ψᵢ|本机航向|rad|无效零槽|未说明|`ψ/π`|归一化未公开|
|4|αᵢ|vᵢ 与 pⱼ̂−pᵢ 的三维夹角|rad|无效零槽|未说明|`z_i/5000`|不一致|
|5|zᵢ|本机世界高度|m|无效零槽|未说明|相对点 dx/10000|不一致|
|6|vⱼ̂|目标速度|m/s|无效零槽|未说明|相对点 dy/10000|不一致|
|7|γⱼ̂|目标飞行路径角|rad|无效零槽|未说明|相对点 dz/10000|不一致|
|8|ψⱼ̂|目标航向|rad|无效零槽|未说明|点距离/10000|不一致|
|9|αⱼ̂|vⱼ̂ 与 **pⱼ̂−pᵢ** 的三维夹角|rad|无效零槽|未说明|水平 bearing−ψᵢ 的 wrap/π|不一致；不能把水平误差当三维夹角|
|10|zⱼ̂|目标世界高度|m|无效零槽|未说明|has_task|不一致|
|11|dᵢⱼ̂|本机到目标三维欧氏距离|m|无效零槽|未说明|最近友机距离/10000|不一致|

尤其 αⱼ̂ 的参考向量仍是 pⱼ̂−pᵢ，不能自行反向为 pᵢ−pⱼ̂。原文 p.3 以向量夹角定义 α，p.4 式(4)又写 −π/2<αⱼ̂<π/2；有符号角的额外编码、角度 wrap 方法没有完整公开，不能据此编造 signed-angle 观测协议。p.4 式(6)或 p.7 式(21)出现的速度归一化是奖励/对手评分公式，不是观测归一化证据。

新增 `paper_protocol.OBSERVATION_FIELDS` 固定上述顺序/单位；`validate_observation_slots` 只验证调用者已提供的 raw 12D 数据、二元 flag、无效全零及有限性，返回副本，不缩放、不计算几何、不选对象、不进入训练。

集中状态：p.5 式(13)/(14)还定义友机关系和关系集合，p.8 表III明确 Critic **60D**，p.8文字给出最多五个友机槽的 padding。**5×12 是合理接口解释，但从式(14)所有关系如何压缩/排序为60维未完整交代。** 当前 `centralized` 是5×12局部槽加5×6航路点表=90D，后30维是后来新增；五价值头也是工程实现，不能冒称作者公开拓扑。本轮不把90D数组截断成60D来伪装对齐。

## 4. Reward 文献审计：严格按原文 a–e 顺序

这里只记录文献和数学关系；**没有把以下交战条件、目标选择、事件或 Reward 接入仿真/策略训练**。各项的合成是 p.7 式(20)，不是巡检五项、区域奖励或另加的辅助项。

|原顺序/名称|原公式及页码|所需状态|明确参数|未公开/需澄清|与当前 `env/reward.py` 的差异及状态|
|---|---|---|---|---|---|
|a Attack Advantage Reward|p.6 a 引用 p.4 式(5)：rᵢⱼ̂=ηᵢ,ⱼ̂−ηⱼ̂,ᵢ；η 由式(4)条件取 η 或0|双方位置/速度、视线角、Ω几何、分配对象|p.8 η=5；τmax=2000m、αmax=π/4；p.3说明Ω是沿速度向量的前向球冠锥体|一步内计算时刻/并列事件细节未完全公开|巡检没有这项；`progress` 是2Δd/1000，**不一致**|
|b Posture Advantage Reward|p.6 b 引用 p.4 式(6)：d<τmin时−1/d；τmin≤d≤τmax时cosαᵢ+cosαⱼ̂；d>τmax时[(vᵢ−vmin)/(vmax−vmin)]βᵢⱼ̂；β=cosαᵢ+max(0,cosθᵢⱼ̂)|距离、本机速度、双方速度/视线角、速度夹角θ|p.8 τmin=100m、τmax=2000m、vmin=100、vmax=400m/s|d=0未定义，未给epsilon/clip；边界跳变未给平滑法；近距项单位1/m与其他项无量纲|巡检没有姿态分段项；不能用Δd替代。**不一致**|
|c Allied Proximity and Shared-Exposure Penalty|p.6 式(15) qⱼ̂→ᵢ=I(ηⱼ̂,ᵢ>0)，式(16) cᵢₖ=maxⱼ̂(qⱼ̂→ᵢqⱼ̂→ₖ)，式(17) r̃ᵢ=−Σₖ≠ᵢ[max(0,1−dᵢₖ/τmin)+cᵢₖ]/(N−1)|有效友机间距离、双方有效性、共同威胁关系|τmin=100m；p.6说明每步使用有效UAV相对状态，且不改变Hungarian分配|N=1除零处理；N取初始/当前数及所有失效边界的细则；前/后态时序未完整公开|巡检只有近距项并除以当前其他有效友机数，无 shared-exposure。近距结构**部分已对齐**，整体**不一致**|
|d Shootdown Reward|p.6 式(18)：连续条件时长 tᵢⱼ̂ > t̄ 时给 r₀，否则0|连续条件累计时间、有效对象/事件及归属|p.8 §IV-A1称两决策步、0.5s；dt=.25s|**r₀数值未给出**；式(18)严格 > 与“两步”表述如何离散对齐；一次性事件、同时多机归属/重置细节未明确|巡检 `completion=10` 是航路点首次进入300m球的低ID归属，不能叫论文r₀，也没有论文参数“10”。**不一致**|
|e Episodic Reward|p.7 式(19)：M̂=0时 r*ᵢ=N̂ e₁−floor(T/e₂)，否则0；正文说所有友机获得|剩余目标数M̂、存活友机数N̂、完成时决策步T|p.8 (e₁,e₂)=(20,25)|终止步与有效mask的具体广播、同一步多事件归属未完全公开|巡检成功时每个存活友机20−floor(T/25)，缺N̂乘数，成功条件也是所有点完成。**不一致**|

p.7 式(20)：rᵢ = rᵢⱼ̂ + r̂ᵢⱼ̂ + r̃ᵢ + r*ᵢⱼ̂ + r*ᵢ，对应式(5),(6),(17),(18),(19)。原文没有独立的−.01每步常数、Δ距离进度、区域占用、驻留计分、Reward总值裁剪。p.6 d 中提到 individual/global supplement，不能额外发明一项“全局+10”叠加在式(20)之外。

### 4.1 当前 Reward 调用链和数值反馈判断

`Cooperative3DEnv.step` 保存 previous/current_mask/prior_assignment → 积分 → 事件/失效 → `compute_rewards` → 新分配/新观测。progress 对同一个前分配对象计算前后距离，efficiency按前有效mask，separation/terminal按后有效mask；completion只处理首次事件。它们与 `docs/environment_spec.md` 的历史巡检规范一致，不等于论文任务一致。

没有在本轮发现该调用链的确证重复计数/错对象Bug；既有事件、mask、轨迹奖励重算及回归测试继续通过。**未复测200000步模型，也没有新的收敛因果结论。** 巡检progress每米.002、常数每步−.01是现有自定义尺度；论文的η=5、远距姿态约量级1、终奖与N̂/T相关是另一反馈机制。是否足够引导某策略，需要固定任务下独立消融，不能凭公式或PPO loss断言有效。本轮不更改任何奖励系数。

## 5. MAPPO 网络和公开参数对齐

本表“论文”除另注外均来自 **p.8 表III**。本轮只修正通用配置的 rollout 默认值250→1500；不改论文超参数、不改历史 checkpoint 内配置。

|项目|论文|当前实际实现/本轮结果|状态/说明|
|---|---|---|---|
|Actor隐藏层|128,128 Tanh|`networks.Actor` 同|已对齐|
|Critic隐藏层|128,128 Tanh|`networks.Critic` 同|已对齐|
|Actor分布/动作|Beta / 3D|独立3个Beta、6个参数头|家族/维数已对齐；独立性和头参数化未明确公开|
|Actor输入|12|12；字段语义见第3节|维数已对齐，语义不一致|
|Critic输入|60|旧 Coop3D 90；参考常量明确60|不一致；没有接入新的60D训练器|
|Critic输出头|表III未说明；式(26)记Vᵢ|五个固定UAV头，按个体ID gather|论文没有明确说明；五头是工程假设|
|Actor/Critic LR|3e−4|均3e−4|已对齐|
|优化器|Adam|分别两个Adam，eps=1e−5|Adam已对齐；eps未公开|
|折扣γ / GAE λ|.954 / .95|同|已对齐；γ与飞行路径角γ不是同一个物理量|
|PPO clip / 熵系数|.2 / .001|同|已对齐|
|epochs / mini|10 / 128|10遍，有效个体样本mini128，保留尾批|数值已对齐；论文未明确mini按联合还是个体计数|
|并行环境 / 每环境rollout|10 / 1500|`TrainConfig`与YAML默认现均10/1500|已对齐；旧checkpoint若保存250仍按250恢复|
|Episode上限|250|`EnvConfig.max_steps=250`，可配置更短诊断|默认已对齐；与rollout1500独立|
|checkpoint间隔|50 episodes|每25000 joint transitions，在更新边界触发|不一致；不假设25000步=50局|
|总训练协议|p.8正文5000局/规模、5独立训练seed；worker1000s+r|CLI绝对joint-step预算；worker每局变seed，默认base1000|不一致；未启动原协议训练|
|评估/选择|p.9 §IV-B1：最终checkpoint 200局；表III存档不是best选择规则|独立固定验证seed默认5局；Beta均值动作；best按success/completion/return字典序；latest分别保存|不一致；均值评估和best规则非作者公开细节|

### 5.1 实际 PPO、GAE、mask 与样本量

- `Actor.evaluate` 在最后3个动作维求和 log_probability 和微分熵，不跨个体求和。α=softplus(h₀:₃)+1、β=softplus(h₃:₆)+1，限制α/β>1；trim中心(.5,1/3,1/8)、浓度16及权重gain=.01未在论文给出。采样后按eps=1e−6裁剪并重新计算存储动作的logp，训练用同一个动作，未发现 old_logp/action 不一致。裁剪产生端点质量，严格数学上是工程近似，不能称为原始连续Beta的完全相同分布。
- `trainer.update` 使用 ρ=exp(logπnew−logπold)，Lpolicy=−mean(min(ρA,clip(ρ,1−.2,1+.2)A))，Lactor=Lpolicy−.001mean(H)。这是代码的通用 PPO 展开式；Xu p.8式(26)只称 entropy-regularized surrogate，**没有给这些展开项一个单独公式编号**。优势在全体有效样本上用总体标准差标准化；作者未公开标准化方法。
- 价值网络按slot gather Vᵢ，目标是对应个体return，没有共享标量广播Bug。代码 **½MSE**，论文 p.8 式(26)为**无½的MSE**；这是损失尺度差异，不是索引错误。本轮保留经过测试的通用实现，没有偷偷调整梯度尺度。
- 式(26)逐个体按有效数归一化；代码共享Actor/五头Critic，合并全部有效个体样本。失效后不同个体样本数不等会改变加权；不能宣称严格同式(26)的逐个体优化。无有效样本跳过、无效观测/状态/梯度排除符合p.8原则。
- p.8式(27)印出的δ=Oₜ[r+γ Oₜ₊₁Vₜ₊₁−Vₜ]，A=δ+γλ bₜ Oₜ₊₁Aₜ₊₁，b为nonterminal。**δ行未显式乘终止b**，终止状态如何令V为0未展开。当前实现额外用terminated门控bootstrap：真正终止不自举；truncated使用真实末状态自举，但切断跨回合carry；rollout尾可自举。不能把当前完整逻辑逐字称为印刷式(27)。250步若为有限时域终局，是否应该自举仍需明确任务价值定义，不能直接改gamma。
- 当前/后继mask分别控制当步奖励保留与后继bootstrap，使用where而非0×NaN。vector env分开保存真实transition状态和reset状态，bootstrap不读取reset值。partial worker从不执行额外转移，padding mask=0。
- 默认完整rollout为10×1500=**15000 joint transitions**，单局250步则一个worker的rollout可以跨多局；不是把1500改成episode上限。全有效3机时45000个体样本，10×ceil(45000/128)=3520次actor及对应critic优化；5机时75000个体样本，5870次。故障/提前budget停止会减少实际有效样本，不把joint步数乘UAV后仍称环境步。
- checkpoint保存12/90/5协议、完整配置、两网络/两Adam、RNG、worker及episode状态，维度/身份不兼容时拒绝；没有把旧60D或Region模型当作当前90D模型加载。新默认horizon不覆写恢复配置。独立验证没有混进训练回合统计；反复用于选best的种子属于验证集，不能充作最终独立测试集。

现有 `test_paper_math_audit.py` 对真实 `update` 做手算对照：ρ=(1.4,.7,1.1)，A=(−2,1,3)，policy=−.4；测试熵项、优势标准化、个体价值索引及无效头梯度。GAE终止/截断/NaN mask、partial budget、checkpoint连续恢复等原有回归保留。**本轮没有发现新增的确证 PPO/GAE 代数Bug**；已确认的是上述论文协议差异，不能用测试通过抹去。

## 6. 未确认细节与未实现功能

|问题|依据/状态|下一步|
|---|---|---|
|原作者数值积分器、g数值、垂直奇点处理、速度约束实施|p.2–3方程/p.8范围有，具体代码细节未公开|获取作者补充材料；不能拿RK4/85°反推原作者实现|
|观测缩放、双旗完整边界、式(14)→60D具体排序|p.5式(12)–(14)、p.7式(25)、p.8表III不完整|只保持raw字段协议；在有来源前不接入策略|
|网络输出头、共享方式、Beta参数化/初始化、优势归一化、grad clip、Adam eps|表III没有这些细节|保留显式工程假设；不能宣称论文已公开|
|reward r₀、d=0/N=1数值定义、连续时长严格比较与两步规则、同步事件时序|p.4式(6)、p.6式(17)/(18)、p.8两步说明|报告缺口；不发明参数/数值clip冒充原式|
|超时价值语义、δ终止门控、mini计数单位|p.8式(26)/(27)/表III未充分展开|以独立已知答案检查通用实现；若设计复现实验须先写明假设|
|APH任务分配|p.4–5式(4)–(10)、p.6算法1：资格优先→每行保留至L=1→残余−Hᵇ Hungarian，残余行多于列时复制真实目标列；允许目标复用|仅学术差异标注。旧巡检独占矩形匹配不等价；不接入自主攻击对象选择|
|动态对手/交战/毁伤/原五项Reward及其观测链|p.3任务定义、p.6–7奖励、p.7式(21)–(24)/表I|依本轮范围未实现，不新增替代竞争任务，也不接入可训练自主交战流程|
|完整训练/统计复现|p.8–9训练与评价协议|未开展5000局×5seed训练或200局最终评价；不声称完成原论文复现|

建议下一步仅三项：①向作者核实上述公开缺口；②决定是否允许用新的数值语义取代历史Coop3D（会影响已存模型轨迹），再单独处理其C类兼容规则；③在不连接交战的范围内继续验证通用PPO/GAE和孤立动力学已知解。当前证据不足以支持重新设计任务或启动长训练。

## 7. 验证结果与复制命令

- WSL发行版枚举失败：`Wsl/EnumerateDistros/Service/E_ACCESSDENIED`，故按AGENTS使用Windows `uav`环境。Python `D:\conda_envs\envs_dirs\uav\python.exe`，CUDA=True，RTX 5060 Laptop GPU，无CPU训练/模型审计回退。
- `python -m pytest -q --junitxml=tmp/xu_strict_audit/pytest.xml`：**176 passed in 8.87s**。未修改旧测试基准或历史轨迹fixture；新增独立方程/协议已知答案测试。
- 本轮没有正式训练、延长200000步实验、重评用户模型或新造轨迹结论。完整测试中的极短CUDA更新/恢复是既有回归测试。
- `python -m compileall -q env algorithm tools tests` 与 `git diff --check` 通过（Git仅提示工作区LF/CRLF转换，没有空白错误）。AST检查全部保留Python源码，无Region包导入；删除目标均不存在。
- **128份受保护文件SHA-256全部一致**：包括所有既有outputs、模型及两份原始PDF。删除前清单、pytest XML及复核JSON在 `tmp/xu_strict_audit/`；摘要为 `verification.json`。未删除历史Region输出，也未把它们重写为别的实验。

Ubuntu仓库目录下可复制执行（无需长训练）：

```bash
conda activate uav
python -c 'import sys,torch; print(sys.executable); assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name())'
python -m pytest -q
python -m compileall -q env algorithm tools tests
```

只检查本轮独立数学与既有PPO手算用例：

```bash
conda activate uav
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
python -m pytest -q tests/test_xu_paper_reference.py tests/test_paper_math_audit.py
```

这些命令运行测试，不是原论文对抗训练入口。

# 三维动态双队伍区域竞争环境 v1

本环境为本项目独立定义的非武器化任务，身份 `region-competition-v1.0.0`。现有 `env.cooperative_env.Cooperative3DEnv`、巡检配置、90D模型协议和历史实验保留。创建入口：

```python
from env.region_competition import RegionCompetitionEnv, RegionConfig
env = RegionCompetitionEnv(RegionConfig(scale='3v3', blue_policy='region_safe'))
obs, info = env.reset(seed=42)
obs, rewards, terminated, truncated, info = env.step(red_actions)  # [N,3], [0,1]
```

支持3v3和5v5。Red由学习策略控制，Blue由确定性规则控制；`policy_actions(team, mode)`提供独立规则接口，`step_teams(red_actions, blue_actions)`用于显式双策略/动力学诊断。Blue每步先根据同一前态生成动作；双方动作同时确定后再积分，Blue不会看到本步执行后的Red状态。所有有效UAV逐步积分，不因任务分配或区域空缺而停止。

没有武器、攻击对象、许可、发射、毁伤、压制接口，没有将它们改名为区域优势或分配函数。安全距离不会使另一方失效。有效性下降仅用于非有限积分的数值故障处理；没有碰撞损伤模型。

## 1. 动力学与数值边界

直接导入已有 `env.dynamics.physical_actions` 与 `env.integrator.rk4`，两队无不同的积分器副本。

状态 `[x,y,z,v,gamma,psi]`，右手系z向上，单位m、m/s、rad。控制为 `[mu,zeta,normal]`，归一化映射 `[2πa0−π,3a1−1,8a2]`；不是速度或姿态增量。运动方程采用Xu p.2–3式(1)–(2)，详见参数对齐文档。

dt=.25s，速度100–400m/s，bank±π，切向[-1,2]，法向[0,8]。每步RK4固定控制；速度、gamma、psi的推进均由相同ODE计算。所有输入动作在**双方任何状态改变前**验证。无效槽输入可以忽略，记录前置0保证JSON有限；有效槽非有限/越界立即失败。

以下为工程数值假设，不是论文飞行包线：g=9.81；gamma末态及RK4导数阶段保护±85°；分母epsilon=1e-6，速度分母取max(abs(v),epsilon)，psi末态wrap到[-π,π)。非有限结果保留上一有限状态并将本机置无效。速度/gamma裁剪按每架完整积分结果各计一次，另记nonfinite_states。两队各自累计和逐架逐步计数，绝不拿累计最大值作本步事件数。

区域交点数值保护 `residence_fraction_epsilon=1e-12`：小于等于该步长分数的子区间不记控制分或事件，避免对称交点相差几ulp产生虚假独占事件；默认相当于2.5e-13s。驻留长度仍由原始交点求得，离线重算容差1e-8秒/奖励单位。配置禁止将该保护放大超过1e-9步长分数。回合分数差小于1e-10视为平局。

## 2. 场景、动态区域和公平性

每环境独立 `numpy.random.Generator`，没有使用全局随机状态决定布局或区域运动。默认两队数量相等、x分别为±2600m；逐ID匹配相同y/z/v/gamma，Blue航向=`wrap(π−psi_Red)`。z在3000±150m、速度210–240m/s、gamma±.04rad、Red航向±.1rad由该局RNG产生。这给出严格的初始x反射公平布局。

默认3个半径900m球形区域，中心锚点x=0、y间隔2200m、z=3000m。它们共享一条平移：

```text
theta(t) = 2*pi*t/40 + phase
center_r(t) = anchor_r + [0, 250*sin(theta), 150*sin(theta/2)]
phase ~ Uniform[-pi,pi]（只在reset采样一次）
```

区域全过程位于x=0，保持对两边的反射对称；共同平移保持区域互不重叠。人工reset options支持双队状态/有效mask、锚点和phase；维度、有限性、包线及锚点间距会验证。自定义不对称options不再代表默认公平分布。

每架指定区域ID=`uav_id % num_regions`，两队采用同一规则。这仅决定局部导航反馈和控制器的区域参考；计分统计所有实际进入的区域。没有静态waypoints/completed机制，没有一次完成后删除区域。区域数量至少1，即使某队全无有效机，区域状态仍可计算。全队数值失效是显式提前终止条件，不是“没有目标”。

规则策略均不调用随机数，也不选择对方UAV：

|模式|定义|
|---|---|
|cruise|物理控制`[0,sin(gamma),cos(gamma)]`，保持v/gamma/psi，照常推进位置|
|region|根据指定区域相对位置调节期望航向/路径角/速度；接近区域后采用切向盘旋；Red/Blue使用镜像盘旋方向，避免世界坐标的转向偏好破坏基准对称性|
|region_safe|上述控制加友机三维距离小于200m时的水平避让反馈；同位置按ID确定方向，不产生NaN|

控制器目标速度225m/s，期望gamma±.3rad、路径角速率±.08rad/s、航向速率±.12rad/s；由ODE反算bank/法向/切向并限到物理动作范围。它们是可解释的规则基线参数，不来自论文，不是经过调参的最优策略。安全反馈是软避让，**不是硬性避碰或安全可达性保证**。Blue规则可读完整本队位置和指定区域，Actor只读下述12D；这种规则信息优势是基准假设，需要在结果中注明。

## 3. 连续时间驻留和唯一记账

在每个dt内，用已记录的前后UAV位置及前后区域中心构造相对直线段，解球面进入/退出根，精确积分这个**分段线性近似**下的驻留时间。它不是只看步末是否在球内，也不是声称精确积分了真实RK4曲线与解析移动区域的连续交点。曲率引起的物理近似误差尚未做步长收敛研究。

对一个区域，将两队所有有效个体的交点划成互不重叠的子时间段：

- 仅Red在内：Red得该子段的秒数，仅Blue在内则Blue得秒数。
- 双方都在内：contested，两队控制分均为0，但覆盖时间和每机驻留仍累计。
- 同队多人在内：队伍只记一次区域秒数，均分给该段的在场个体作为control_credit，堆叠不会倍增分数。
- 计分只包含本步前后均有效的UAV；无效槽无驻留/控制信用。

每段事件有唯一的 `(step,region_id,start_fraction,end_fraction)`，并记录duration及双方在场ID。`scores`累加各区域exclusive秒数，单队单步每区不超过dt。计分是持续时间的积分，不是反复发放同一个“完成事件”奖励。

## 4. 12D Actor与60D集中输入

**这是本项目新任务语义，不是Xu原始12D观测的复现。** 有效UAV逐索引定义如下；无效UAV整行0，未裁剪归一化距离或高度。

|索引|物理量/单位|实际归一化|
|---|---|---|
|0|有效性|0或1|
|1|速度m/s|`(v−100)/300`|
|2|飞行路径角rad|`gamma/radians(85)`|
|3|航向|`sin(psi)`|
|4|航向|`cos(psi)`，消除±π标量表示跳变|
|5|高度m|`z/5000`|
|6–8|指定移动区域中心减本机位置，xyz m|分别除5000|
|9|最近有效本队友机三维距离m|除5000；无友机为−1，与零距离区分|
|10|最近有效另一队UAV三维距离m|除5000；对方全无效为−1，仅用于空间关系|
|11|回合剩余步数|`max(0,max_steps−step)/max_steps`，reset=1，截止=0|

Critic严格将最多5架Red的12D观测按固定ID拼接为60，缺机槽全0；没有附加区域、Blue状态、分数或时间侧通道。共享干+5个个体V头；3v3只用前3头，5v5使用全部。

60维**并不是环境的完整Markov状态**：它缺累计比分差、另一队速度/方向、区域运动相位/速度、友机相对方向和未见区域状态。尤其相同局部状态/剩余时间、不同累计比分会使终局结果不同；增加时间只解决原审计中的时钟缺失，不解决所有状态别名。此处是按要求保持60D接口的压缩集中Critic/部分可观测基准，不能宣称与完整状态价值等价。需要更完整状态时应独立升级协议并先做实验论证，不悄悄扩维。

## 5. 五项新任务Reward

Reward不复现Xu或Li的任务奖励。默认系数均可在新环境配置显式查看，未据Smoke成功率调节。

|项|精确定义|单位与时刻|
|---|---|---|
|control|`1 * 本机均分exclusive区域秒数`|每独占区域秒1奖励单位；由真实驻留子段记账|
|tracking|`1 * (norm(x_before−c_before)−norm(x_after−c_before))/5000`|每向指定区域移动5000m得1；固定前态区域位置，移动区域自身靠近不能冒充本机progress|
|separation|`−1 * mean(max(0,1−d_friend/100))`|步末有效友机三维距离；没有友机为0，仅本队、不含对方距离|
|numerical|`−.1 * (speed_clip+gamma_clip+nonfinite)`|本步前有效机各自数值事件计数|
|result|实际终局时`5 * sign(S_own−S_other)`，平局0|仅一次；正常时间截止或数值提前结束均按当时真实累计比分|

control/tracking/numerical/result属于动作前有效机；separation属于步末有效机。失效当步保留其有效动作转移与数值惩罚，后续不参与损失。没有效率-.01、巡检+10完成奖励或`20-floor(T/25)`。没有隐蔽交战/接近对方奖励。两队奖励和分数分别记录；它们只有终局结果项为零和，全部训练Reward并不是零和。

环境分项和总和用float64记录及返回，learner统一转换float32训练。原始JSON可精确核算环境Reward；转换到训练张量的float32舍入属于网络数值类型，不是额外Reward。

## 6. 边界、采样与恢复

|边界|API|后继价值|GAE递推|
|---|---|---|---|
|数值提前真实结束（一队无有效机）|terminated=True，truncated=False|0|切断|
|有限回合时间截止（正式250步）|terminated=False，truncated=True|0|切断，不能进入reset|
|rollout采样切断，任务尚未结束|两者False|真实后继60D的V_i|buffer末尾后不递推，但δ保留自举|

buffer保留真实terminated/truncated两个标记。RegionTrainer仅在计算价值目标时使用二者OR作为value-terminal，调用原已验证的GAE数学函数；没有修改旧巡检版的超时自举约定。区域环境另记录`bootstrap_terminal`用于审计。

1500是**每worker的rollout长度**，250才是episode上限，一个完整rollout可以跨越6个完整回合。每次自动reset前的真实后继状态与reset状态分别保存；尾部不足预算时只step部分worker，未执行槽current mask=0，不产生伪样本。

Region checkpoint身份为 `mappo-region-v1.0.0`，协议12/60/3/5头、deadline_bootstrap=False；旧巡检身份及12/90协议互相拒绝。保存Actor/Critic、两个Adam、Python/NumPy/Torch/CUDA RNG、joint步/episode/update、每个worker全部双队状态/积分计数/区域相位/比分/时钟/RNG、父进程观测和回合统计。恢复在配置相同前提下进行，后续采样/优化可逐位复现。

## 7. 评价、记录、回放与离线审计

训练与独立评价种子隔离；worker i第e局seed=`base+e*num_envs+i`，训练禁止碰到保留评价种子；checkpoint评价检查其seed是否已用于训练。每个不同seed的完整结果保存在`per_episode`，不只给平均Reward。

指标包括两队各自得分/相对得分、平均和逐机驻留秒数、有效飞行步、速度、gamma五分位分布、三维航程、本队安全违例率、跨队接近率、速度/gamma/nonfinite率、episode长度与结果。安全率按“该机步末是否有距离<100m的友机/对方”分别统计，分母为该队动作前有效UAV步；不是连续时间碰撞率。优化日志保留Actor/Critic loss、KL、clip fraction、entropy、valid_samples、optimizer_steps。

JSON逐帧记录双队状态、动作、物理控制、mask、分项、计数和动态区域事件；学习策略还记录Beta参数、三动作求和logp/entropy及mean选择方式。HTML使用嵌入Plotly，无CDN依赖；Red红色实线，Blue蓝色虚线，区域球随时间移动，slider/play显示真实累计得分和步号。

`tools.audit_region_trajectory`不导入env/torch、不加载模型、不重跑积分；使用另一种几何求根表达独立重算驻留、exclusive信用、分数和五项Reward，核查事件在场ID、无重复/重叠段、动作物理映射。输出拒绝覆盖源文件或已有报告。它核查的是已记录分段线性计分模型，不证明飞行包线物理真实性或学习收敛。

numerical奖励从记录的逐机逐步数值事件重算系数与归属，不声称离线重新积分验证了事件计数本身；计数逻辑由双方动力学/裁剪单测另行核查。

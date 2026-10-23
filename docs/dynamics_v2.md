# Xu 空战动力学 v2：数学关系、延续假设和故障协议

本次修复改变积分表示和仿真故障语义，不改变论文的控制范围、重力、dt=0.25、五项奖励、APH、攻击判定、网络、Beta、GAE、PPO 或并行采样顺序。原始 `env/paper_dynamics.py` 和 `env/integrator.py` 未修改。协议升级为 `xu-aircombat-v2.0.0`。

## 1. 为什么无法声称全局严格等价

论文角度方程的 `psi_dot = g*zeta_hat*sin(mu)/(v*cos(gamma))` 在垂直方向失去定义。速度方向是良好的物理量，水平航向和以其定义的银行控制参考轴却不唯一。换坐标只能消除表示奇点，不能补出论文没有公开的参考轴延续。

v2 因而明确增加**工程假设**：保存连续局部参考系，在极区平滑减弱原球面参考系的绕轴旋转，离开极区后以有界速度向等价球面参考轴恢复。这并非作者公开的算法，尤其不能宣称穿越奇点之后的控制转移与作者实现严格等价。极区经历后的参考系记忆也可能在非极区继续影响控制，直到其恢复；不是只有极区内部才存在差别。

## 2. 正常区域的微分关系

以列向量保存 `F=[u,a,b]=[e_v,e_gamma,e_psi]`。按照论文的基向量定义，`b=a×u`，故 F 是正交、行列式为 **-1** 的标架，不应错误地把它检查成右手旋转矩阵。

```
u = (cos(gamma)cos(psi), cos(gamma)sin(psi), sin(gamma))
a = (-sin(gamma)cos(psi), -sin(gamma)sin(psi), cos(gamma))
b = (-sin(psi), cos(psi), 0)
q = z_axis - u_z*u
p_dot = v*u
v_dot = g*(zeta-u_z)
u_dot = g/v * (zeta_hat*(cos(mu)*a+sin(mu)*b)-q)
alpha = u_dot·a; beta = u_dot·b
a_dot = -alpha*u + omega*b
b_dot = -beta*u - omega*a
```

当标架与球面参考轴一致时，`q=cos(gamma)*a`，所以 `alpha=gamma_dot`、`beta=cos(gamma)*psi_dot`。进一步 `v_vec_dot = v_dot*u+v*u_dot`，恰为用户给出的矢量加速度方程。球面标架的旋转率为 `omega_s=-sin(gamma)*psi_dot`。正常区、球面一致的任意合法状态与完整控制范围满足这些等式；测试同时检查位置、速率、速度矢量导数和正交约束导数。

## 3. 有界延续规则（工程假设）

设 `h=hypot(u_x,u_y)`，默认极区半径 `h_c=0.1`，即距离最近垂直方向约 5.739 度内。**这是参考系连接的过渡区，不是航迹角限制**；可自由通过垂直方向。设置 `t=min(h/h_c,1)`、`w=t²(3-2t)`；当 h>0，定义

```
b0 = (-u_y/h,u_x/h,0)
omega_s = -u_z*(u_dot·b0)/h
delta = atan2(q·b,q·a)
omega = w * (omega_s + k/2*sin(2*delta)), k=2 /s
```

当 h=0，直接定义 omega=0；不是给分母加 epsilon。由于 `w/h <= 1.125/h_c`，单位标架满足

```
|u_dot| <= g*(zeta_hat+1)/v
|omega| <= 1.125*g*(zeta_hat+1)/(v*h_c) + k/2
```

因此极点没有无界参考轴旋转。sin(2delta) 允许两个等价的展开球面角度图，避免选择正反参考轴时发生离散旋转。正常区球面一致状态的恢复项为零。离开极区后，小参考轴误差按速率 k 恢复；一般情况下 `delta_dot=-k/2*sin(2delta)`。严格正交于目标参考轴的 delta=±pi/2 是该平滑规则的非吸引平衡，不承诺任意参考轴误差都能在有限时间精确归零。实际动力学始终使用保存的连续标架，绝不偷偷重建或瞬时重置。

实现还直接计算极区内的 `w/h=t*(3-2t)/h_c`，而不是先算可能溢出的 omega_s 再乘趋零的 w。附加测试把 h 设为次正规数1e-320及恰好0，在禁止除零、溢出和无效浮点运算的条件下验证有限极限。这是代数重排，不改变延续方程。

参数选择理由也属于工程假设：h_c=0.1把速度100m/s、最大控制下的上述单位标架旋转率界约束到10.93rad/s，配合8子步使每子步界约0.34rad（实际RK4阶段速率变化另有影响），让正常初态区域完全不进入连接削弱区；k=2/s对应小偏差约0.5秒的连续恢复时间，避免瞬时重建。这些值没有按胜率选择，也没有宣称是唯一或最优值。实际误差、子步加密和0.05/0.1/0.2半径敏感性见验收报告；长期轨迹对半径明显敏感，不能把数值上有限误认为复现结论对假设不敏感。

完整重力投影 `-g*q` 在延续过程中保持不变；不会错误地把“重力的法向分量”固定在已偏离重力投影的参考轴上。法向控制模长仍由 zeta_hat 给定，不裁剪或改写网络控制。

实现为标架/位置/速率联合 RK4。正常步 1 个子步；当一步内可能接近极区（`h < h_c + dt*g*(zeta_hat+1)/v`）使用 8 个子步。**穿越能力来自标架状态和有界连接，不来自子步数**。每子步执行 Gram–Schmidt 到正交流形：先检查未投影正交误差不超过 0.05，再单位化 u、正交化 a 并置 b=a×u；范数塌缩和非有限状态直接报错。该阈值是退化检查，不是宣称全局积分误差小于 0.05。入口状态的正交误差阈值为 1e-8，出口恢复到机器精度。原有步末速度范围投影保留；不新增高度/航迹角限制。

对外仍为 `[x,y,z,v,gamma,psi]`。从物理 u 选择距离前一角度最近的等价展开图，不用其 atan2 来重建内部参考轴。水平范数≤1e-14 时航向物理上不可观测，保留前值；此阈值只影响出口角度表示，最大方向误差为该量级，不进入力或控制计算。原有 ±pi 航向环绕接口保留；普通 2pi 环绕不代表速度方向或标架跳变。穿越垂直后 gamma 可越过 ±pi/2，绝不裁剪至 85 度。

## 4. 状态生命周期和 Blue

双方每个稳定槽均保存 3×3 标架。reset 从公开初态角度初始化；真实积分只更新存活槽；阵亡槽的状态和标架冻结。snapshot 深拷贝标架、计数、数值诊断和 worker RNG；restore 校验版本、完整配置及存活标架。候选推演使用同一 advance、同一初始标架，返回独立数组；不修改真实状态、计数、dwell 或 RNG。15 个候选、原评分公式及原 80/20 选择规则不变。

`integrator=angle_rk4` 保留原积分作为明确标记的历史数学对照。其虚拟候选仍把不可积候选记为 -inf；真实积分失败和少于两个可行候选会 fail-fast，不能再生成原有的伪空战终局。它是 v2 故障协议下的历史积分对照，不冒充完整的 v1 终局语义。

## 5. 仿真故障与训练

双方阵亡、双方同时阵亡及 250 步上限沿用原规则。成功的极区延续没有 kill、win、numerical_failure 或额外奖惩。真正故障抛 `SimulationFault`，包含原因、步数、控制、双方状态和标架；关节物理推进失败时保持原位置、掩码、事件、步数和 worker RNG，环境进入必须 reset 的故障状态。战斗输出的非有限值同样拒绝返回，并禁止保存故障环境。

spawn worker 把异常和 traceback 返回主进程并退出。vector 收齐成功结果后才处理 episode/自动 reset，因此失败波次不会产生回合或 rollout 行。collect 失败后 train 仅追加 `simulation_faults.jsonl`，不执行 PPO 更新，不替换 latest 或已有目标 checkpoint。已有有效 pending buffer 也不自动提交。手工恢复必须使用最近的有效 checkpoint，失败波次剩余 worker 不继续训练。

原 `numerical_failure`/旧日志字段仍可读；v2 真故障属于独立失败记录，不能从“已完成回合中故障为0”推断运行没有发生过中断，应同时查看 fault 文件和训练是否完成预算。未完成评估因异常停止，不输出伪完整的评估结果。

## 6. 诊断和兼容性

每回合记录：采样到的最小水平范数（含 RK4 stage，非解析轨迹全局最小值）、进入极区次数、成功极区子步数、投影前误差、投影后正交误差、最大参考轴旋转率和参考轴偏差。训练每 rollout 保存最后各 worker 诊断，以及稀疏极区进入事件（真实控制、步数、进出标架）。这些统计只读积分结果；不参与奖励/采样/PPO。角度历史模式未测得的标架旋转率字段直接缺省，不填假零。

checkpoint 已有的环境版本、env_config 和 reproduction_assumptions 保存模式及全部延续参数，worker snapshot 保存附加标架。v1 checkpoint 与 v2、不同积分模式/阈值以及不同训练预算之间，常规训练恢复均拒绝。诊断工具仅在显式旧版本身份验证后，读取旧 Actor 权重，绝不恢复旧优化器、RNG、环境或训练次数到 v2，也不修改历史 checkpoint。

新默认根目录为 `outputs/xu_iv_v2` / `outputs/xu_iv_v2_2500`。原数据不迁移、不覆盖。图表工具拒绝混合版本或不同延续参数。旧无版本元数据保留为未标版本，不能悄悄与 v2 合并。

## 7. 复查命令

在 Ubuntu 的 uav 环境执行，训练/模型审计强制 CUDA。每次审计使用新输出目录。

```bash
conda activate uav
python -m tools.audit_dynamics_v2 --phase deterministic --output outputs/dynamics_audit_new
python -m tools.audit_dynamics_v2 --phase policies --historical-checkpoint outputs/xu_iv_2500/main_3v3/seed_1/latest.pt --output outputs/dynamics_audit_new
python -m tools.audit_dynamics_v2 --phase cuda --output outputs/dynamics_audit_new
python -m pytest -q
python -m compileall -q algorithm env tools tests
```

需要正式重训时另行执行下列命令；本次不会执行：

```bash
python -m algorithm.experiments train --experiment main_3v3 --seed 1 --episodes 5000 --root outputs/xu_iv_v2_5000
```

这会从头训练，默认仍是 10 spawn、1500 horizon、250 步、50 回合 checkpoint、最终独立 200 局。v2 结果必须注明延续假设，不能与 v1 或作者未公开的极区实现混称全局严格复现。

# Coop3D 重构与验收报告

实施日期：2026-10-21（Asia/Shanghai）。唯一正式环境：`coop3d-v1.0.0`，类 `coop3d.Cooperative3DEnv`。

## 1. 实施结果

已将原项目重构为非武器化的三维多 UAV 空间巡检强化学习基准。完成了真实代码替换、旧实现删除、自动测试、三规模十进程 CUDA 采样、完整 PPO 梯度更新、断点保存与继续训练、独立评估和离线交互式轨迹文件生成。

工程验收通过；短程策略尚未学会巡检任务。验收不等价于训练收敛，也没有复现论文的任务成功率。

## 2. 文献核对与方法取舍

先阅读本地 Xu 等 2026 原始 PDF，提取全文，并渲染检查第 3、8 页：

- 第 3 页公式 (1)、(2)：z 轴向上的六维状态与三自由度动力学。
- 第 8 页公式 (26)、(27)：当前有效性用于损失，下一有效性用于 GAE 自举。
- 第 8 页表 II：三个连续控制变量的物理范围。
- 第 8 页表 III：128×128 Tanh、Beta、12D Actor、60D Critic、优化与并行参数。

任务定义、独占航路点匹配、观测字段和巡检奖励采用用户任务文本。论文的原始交战任务不进入新实现。论文 rollout 为 1500，而本次自行选择 rollout=250，并在配置中显式暴露，详见 `coop3d_spec.md`。

用户在本次会话明确要求 PowerShell `uav`，因此所有 Python 实施、测试、训练、评估与审计均通过该 Windows Conda 环境执行。CUDA 检查成功：PyTorch `2.11.0+cu128`，GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU。没有训练/审计 CPU 回退。

## 3. 旧实现删除与数据保护

先扫描 821 个原文件并记录实际 Python 导入依赖，再对原始文献和数据建立 SHA-256 清单。检查发现原动力学使用 NED 符号，旧 MAPPO 使用 Gaussian/Attention 网络，旧 vector/checkpoint/记录模块与历史环境身份和统计紧密耦合，不能直接保留。

经用户明确批准，且新实现验收通过后，删除以下 803 个旧实现文件：

| 原目录 | 删除数量 | 内容 |
|---|---:|---|
| `env/` |145|旧正式环境、历史继承链、版本分发、控制增量、旧状态机、旧奖励、旧策略与专属公共封装|
| `algorithm/` |140|全部旧算法分支与入口、旧 MAPPO 网络/训练分支、旧公共训练协议|
| `tools/` |136|旧环境审计、历史迁移脚本、诊断、评估及旧回放工具|
| `tests/` |92|旧环境和旧算法专属 Python 测试|
| `configs/` |215|全部旧环境与算法 YAML 配置|
| `docs/` |75|不再适用的旧版本与算法技术文档|

空的旧环境/算法目录以及旧字节码已清除，README、依赖和项目配置已替换。最终正式代码没有导入 `env.*` 或 `algorithm.*`，不存在旧环境混用或历史兼容入口。

完整删除和导入清单保存在 `docs/migration_inventory.json`；恢复副本为 `tmp/pre_coop3d_source_backup.zip`，包含 803 个旧实现文件和旧 README/依赖/项目配置，共 806 项。恢复副本不属于正式源码，也不被导入或测试。

20 个原始 PDF、模型和 JSON 数据文件的 SHA-256 全部保持不变。旧 JSON 回归数据及模型 fixture 原位保留，属于受保护数据，不提供旧环境执行能力。没有清理原训练产物或文献。

最初批量清理被自动审批拒绝，未执行。随后获得用户明确批准，完成新实现验证和可恢复备份，再依据逐文件清单删除；本项已经解决，无待批准的清理操作。

## 4. 公共能力保留情况与最终结构

保留并整理了通用能力：角度环绕、经典 RK4 方法、spawn 长驻并行采样、JSON 训练记录、checkpoint 和三维回放。由于旧模块接口直接依赖历史环境，最终没有继续导入旧公共 Python 模块；这些能力在新包中以干净接口实现。没有同时保留两套运动模型、网络或配置。

```text
coop3d/
  __init__.py __main__.py
  config.py                 配置及参数校验
  models.py                 状态、任务数据、间距
  dynamics.py               三自由度方程、动作映射、速度向量
  integrator.py             RK4、限幅、非有限诊断
  scenario.py               固定/随机初始化
  assignment.py             矩形 Hungarian / nearest 分配
  observation.py            12D 观测与 60D 集中状态
  reward.py                 五项可重算奖励
  environment.py            唯一环境转移/完成/失效/终止
  vector_env.py             spawn、多进程快照/恢复、真实末观测
  recording.py replay.py    真实轨迹与离线交互 HTML
  mappo/
    config.py networks.py trainer.py checkpoint.py evaluation.py runner.py
configs/
  coop3d_3U3P.yaml coop3d_5U5P.yaml coop3d_3U5P.yaml coop3d_mappo.yaml
tests/
  test_coop3d_dynamics.py test_coop3d_environment.py test_coop3d_mappo.py
tools/validate_coop3d.py
docs/coop3d_spec.md docs/migration_inventory.json docs/refactor_report.md
outputs/coop3d_acceptance_v1/
  3U3P/ 5U5P/ 3U5P/ cli_holdout/
  acceptance.json final_audit.json protected_files_verified.json pytest.xml
```

## 5. 关键参数及补充约定

| 项目 | 实现 |
|---|---|
| 场景 |3U3P、5U5P、3U5P，一个环境类|
| 状态 |`[x,y,z,v,γ,ψ]`，右手系 z 向上，有效性另存|
| 动力学 |严格三自由度微分方程，经典 RK4|
| g / dt / 步数 |9.81 / 0.25 s / 250|
| 速度 / γ |100–400 m/s；γ ±85° 数值稳定性假设|
| 动作 |Beta `[0,1]^3` → μ `[-π,π]`、ζ `[-1,2]`、ζ̂ `[0,8]`|
| 巡检 / 安全半径 |300 m / 100 m|
| 分配 |每步独占矩形 Hungarian；最近独占贪心基线|
| Actor / Critic |12→128→128→6 Beta 参数；60→128→128→1，Tanh|
| 优化 |Adam，两网络 LR=3e-4，gamma=.954，lambda=.95，clip=.2|
| 熵 / PPO / batch |.001 / 10 epochs / 128|
| 并行 / rollout |10 / 250|

全部归一化公式、五项奖励公式和额外优化参数在 `coop3d_spec.md` 逐项列出。主要补充约定：

1. Hungarian 自行实现，不依赖未安装的 SciPy，使用穷举验证矩形最优性；同代价按固定遍历顺序确定。
2. 默认优先级为 1；最近分配按全局最短距离对贪心去重。
3. 完成按步末位置判定，不在 reset 时完成、不做线段穿球检测；实际到达可完成未分配给自己的点；最低有效 UAV ID 唯一记账，重叠的不同点可分别完成。
4. 动作前分配用于 progress，动作后重新分配用于下一观测，防止虚假进度。
5. 确定性失效发生在积分后、完成判定前；当步保留转移及进度/时间奖励，失效 UAV 不参与步末安全项或完成事件。后续状态冻结、观测全零。
6. 无任务有效 UAV 用 `[0,sinγ,cosγ]` 配平控制保持当前速度与飞行方向，仍通过相同微分方程积分；安全项继续记录。
7. 非有限积分保留前个有限状态、计数并失效；非有限输入明确报错。限幅为数值保护，不新增飞行包线奖励。
8. Critic 单一集中标量作为各个体 GAE 的共享基线，MSE 对有效个体 return 平均；不额外拼接个体观测。
9. 真终止不自举；超时使用真实末观测自举，同时切断跨局 GAE；自动 reset 观测与真实末观测分开保存。
10. Adam eps=1e-5、梯度限幅=.5、有效优势标准化、Beta 样本保护 epsilon=1e-6、Critic `.5*MSE`；没有额外隐藏奖励或研究算法分支。
11. 逐更新独立评估，按成功率/完成率/回报字典序选最佳模型；完整保存随机与 worker 状态。恢复自动读取未显式覆盖的断点配置，显式配置冲突则失败。

## 6. 自动化测试与静态检查

清理后的完整 `python -m pytest -q`：**66 项全部通过**。覆盖：

- 动力学：水平匀速、加减速、左右转弯、爬升/俯冲、单个及批量三维速度、速度限幅、解析圆周轨迹下 RK4 精度、航向环绕、奇异点保护、γ 限幅和非有限保护。
- 任务：三种规模完整单局、同种子复现、初始间距、三维到达、同时到达最低 ID、去重、矩形匹配穷举最优性、任务/飞行器数量不对称、失效与闲置槽位、完成后重新分配、成功终止及超时。
- 奖励：进度正负、事件奖励只一次、安全平均公式、仅一架安全惩罚为零、时间与终端项、五项求和、用记录重算。
- 失效：当前/下一掩码语义、后续状态冻结、无效动作忽略、观测清零、全部失效终止、闲置配平。
- MAPPO：网络维度、Beta 解析 log probability、PPO ratio、GAE 真终止/超时/失效、单进程与三规模多进程采样、worker 快照恢复、真实 CUDA 更新、完整 checkpoint、精确恢复后采样一致、独立评估、失效样本污染不影响损失、无有效样本不更新。

同时通过 `compileall`、`git diff --check`；回放应用脚本通过 Node `--check` 语法检查。仅有 Git 对 Windows 行尾转换的提示，没有差异空白错误。

最终依赖审计检查：只有 `Cooperative3DEnv` 正式仿真类和 `ParallelVectorEnv` 批量包装类；只有四份新 YAML；原 803 个实现路径已删除；正式源码没有旧包导入或旧机制关键词；20 个受保护文件哈希未变化。7 份真实轨迹的 **1,750 个 step 帧**全部逐步重算五项奖励，数值一致且状态有限。机器可读证据：`outputs/coop3d_acceptance_v1/final_audit.json`。

## 7. 实际 MAPPO 验收结果

命令：

```powershell
conda activate uav
python -u tools/validate_coop3d.py --output outputs/coop3d_acceptance_v1
```

每种规模：10 个不同 PID 的 spawn worker ×250 步，执行 10 个 PPO epoch，真实 Actor/Critic 反向传播，保存并重新加载模型，在独立种子评估并生成 251 帧实际轨迹（初始帧+250 转移）。首轮结果：

| 规模 | 环境转移 | 有效个体样本 | PPO 小批量次数 | Actor loss | Critic loss | 独立评估均回报 | 额外 holdout 均回报 |
|---|---:|---:|---:|---:|---:|---:|---:|
|3U3P|2500|7500|590|-0.002466|0.200958|-8.381088|-8.327370|
|5U5P|2500|12500|980|-0.001616|0.187504|-6.505633|-5.523004|
|3U5P|2500|7500|590|-0.003646|0.281964|-8.204773|-8.503886|

此处每个小批量均分别更新 Actor 和 Critic。首轮合计 2,160 次小批量成对梯度更新，不是网络前向或初始化检查。三种规模都通过精确模型参数重新加载检查。训练中非有限状态计数均为 0；速度限幅有触发，记录已保存。

又实际执行命令行完整恢复：

```powershell
python -m coop3d train --output outputs/coop3d_acceptance_v1/3U3P --updates 1 --resume outputs/coop3d_acceptance_v1/3U3P/latest.pt
python -m coop3d evaluate --checkpoint outputs/coop3d_acceptance_v1/3U3P/latest.pt --output outputs/coop3d_acceptance_v1/cli_holdout --seeds 30000000 30000001
```

3U3P 从 update=1 成功继续到 update=2，新增 7,500 个体样本和 590 次小批量更新，Critic loss=0.134980。CLI holdout 均回报=-6.242404。最终本次主验收及恢复合计采样 10,000 个环境转移，执行 2,750 次小批量成对更新。

所有短程独立评估及额外 holdout 的完成率/成功率均为 **0**，通常超时 250 步。上述均回报只是工程验证数值，不能解释成学习效果已经达标。初始 Beta 策略的法向过载偏高，确定性均值评估频繁触发 γ/速度限幅，例如 CLI seed=30000000 的 γ 限幅 610 次、速度限幅 271 次，但非有限状态为 0。下一步训练研究应关注这一行为，不能通过隐藏奖励修改掩盖。

`acceptance.json` 保存三规模首轮结果；3U3P 的 `latest.pt`、`evaluation.json` 和 `replay.html` 已更新为恢复后的第 2 次更新结果。`metrics.jsonl` 保留两个更新记录；首轮验收数字没有被重新解释为第二轮数据。

## 8. 回放产物与待解决事项

每种规模目录均有 `best.pt`、`latest.pt`、完整 `config.json`、`metrics.jsonl`、`evaluation.json`、`trajectory.json`、`replay.html`，以及 `holdout_trajectory.json` / `holdout_replay.html`。CLI holdout 另有独立轨迹、报告和页面。HTML 内嵌 Plotly 库及原始记录，不依赖 CDN；UI 只读取记录，不运行环境或改动仿真结果。

主要回放入口：

- `outputs/coop3d_acceptance_v1/3U3P/replay.html`
- `outputs/coop3d_acceptance_v1/5U5P/replay.html`
- `outputs/coop3d_acceptance_v1/3U5P/replay.html`

仍然存在的限制：

1. **尚未完成充分训练与多种子性能验证**；仅完成要求的真实短程训练验证，策略成功率目前为 0，不能宣称已收敛。当前任务从 5–12 km 初始距离起飞，包含稀疏完成奖励，长期学习难度需要单独实测。
2. **浏览器内视觉及交互人工式验收尚未进行**。自动打开本地 HTTP 回放页时，浏览器安全策略报告本地源访问权限未获批准；未改用其他浏览器或绕行。页面生成、离线内嵌结构、脚本语法、真实数据和奖励重算已验证，但实际播放、滑块和三维视角效果仍需用户本地打开检查。
3. ±85° γ 和速度限幅属于数值约定，短程确定性策略频繁触发限幅。没有将此误称为论文飞行包线，也没有加入未授权的惩罚。
4. 本基准没有地形、能源、传感器噪声或碰撞损伤模型；100 m 安全项是奖励参考，不能作为严格无碰撞保证。
5. 原训练产物、原文献和历史数据保留，但不能由新代码加载旧环境 checkpoint 继续训练；新断点严格验证身份和配置。

没有已知未处理的导入/维度错误、NaN 或失效样本参与后续更新的问题。代码未自动提交 Git，也未推送远端。

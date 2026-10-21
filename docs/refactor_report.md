# Coop3D v2.0 重构、修正与训练前诊断报告

日期：2026-10-21（Asia/Shanghai）。本轮完成实际代码重构与有限预算诊断，没有启动正式长步数训练、百万级采样、大规模参数搜索或多种子长训练。

## A. 最终目录与职责

正式代码树如下；历史数据、文献、outputs 和测试fixture保留，不列出每个历史文件。完整正式源码路径清单见 `docs/v2_file_changes.json`，数据保护清单见 `docs/v2_migration_manifest.json`。

```text
env/
  __init__.py
  assignment.py
  config.py
  cooperative_env.py
  dynamics.py
  factory.py
  integrator.py
  models.py
  observation.py
  reward.py
  scenario.py
algorithm/
  __init__.py
  common/__init__.py
  common/checkpoint.py
  common/metrics.py
  common/protocol.py
  common/runtime.py
  common/serialization.py
  common/vector_env.py
  evaluate_mappo.py
  mappo/__init__.py
  mappo/config.py
  mappo/evaluation.py
  mappo/networks.py
  mappo/runner.py
  mappo/trainer.py
  train_mappo.py
configs/
  coop3d_3U3P.yaml
  coop3d_3U5P.yaml
  coop3d_5U5P.yaml
  mappo_coop3d.yaml
tools/
  __init__.py
  audit_navigation.py
  navigation_baseline.py
  record_episode.py
  recording.py
  render_replay.py
  validate_coop3d.py
tests/
  test_dynamics.py
  test_entrypoints.py
  test_environment.py
  test_environment_regression.py
  test_mappo.py
  test_mappo_v2.py
  test_navigation.py
  fixtures/coop3d_v1_semantics.json.gz + 受保护历史数据
docs/
  environment_spec.md
  mappo_spec.md
  navigation_audit.md
  refactor_report.md
  migration_inventory.json v2_migration_manifest.json v2_file_changes.json
README.md / pyproject.toml / requirements.txt / AGENTS.md
outputs/coop3d_v2/ + 原有 outputs（完整保留）
原始 PDF（完整保留）
```

env仅负责配置、动力学、任务与观测/奖励，AST检查确认不导入algorithm、tools或torch。algorithm通过env.factory.make_environment()构造环境。common负责vector/checkpoint/指标/协议，mappo负责网络/GAE/PPO/调度。tools导航只用于诊断，不进入训练或专家数据。

## B. 文件迁移、删除与新建清单

以下“迁移或重写”不表示都属于行为不变的纯搬运。结构和数值变更单独列于C节。

|原路径|新路径|
|---|---|
|`coop3d/config.py`|`env/config.py`|
|`coop3d/models.py`|`env/models.py`|
|`coop3d/dynamics.py`|`env/dynamics.py`|
|`coop3d/integrator.py`|`env/integrator.py`|
|`coop3d/scenario.py`|`env/scenario.py`|
|`coop3d/assignment.py`|`env/assignment.py`|
|`coop3d/observation.py`|`env/observation.py`|
|`coop3d/reward.py`|`env/reward.py`|
|`coop3d/environment.py`|`env/cooperative_env.py`|
|`coop3d/vector_env.py`|`algorithm/common/vector_env.py`|
|`coop3d/mappo/checkpoint.py`|`algorithm/common/checkpoint.py`|
|`coop3d/mappo/config.py`|`algorithm/mappo/config.py`|
|`coop3d/recording.py`|`tools/recording.py`|
|`coop3d/replay.py`|`tools/render_replay.py`|
|`coop3d/mappo/networks.py`|`algorithm/mappo/networks.py`|
|`coop3d/mappo/trainer.py`|`algorithm/mappo/trainer.py`|
|`coop3d/mappo/runner.py`|`algorithm/mappo/runner.py`|
|`coop3d/mappo/evaluation.py`|`algorithm/mappo/evaluation.py`|
|`tests/test_coop3d_dynamics.py`|`tests/test_dynamics.py`|
|`tests/test_coop3d_environment.py`|`tests/test_environment.py`|
|`tests/test_coop3d_mappo.py`|`tests/test_mappo.py`|
|`configs/coop3d_mappo.yaml`|`configs/mappo_coop3d.yaml`|
|`docs/coop3d_spec.md`|`docs/environment_spec.md`|

彻底移除而不保留转发包装：
- `coop3d/__init__.py`
- `coop3d/__main__.py`
- `coop3d/mappo/__init__.py`

全部旧coop3d/源码和字节码目录已删除，旧模块路径不再能作为入口导入。旧Python实现共21个；三份原测试文件更名迁移，没有删去有效测试函数。失效旧YAML与旧规范由新路径替代。

新增正式Python文件（不包括上表迁移文件）：
- `algorithm/__init__.py`
- `algorithm/common/__init__.py`
- `algorithm/common/metrics.py`
- `algorithm/common/protocol.py`
- `algorithm/common/runtime.py`
- `algorithm/common/serialization.py`
- `algorithm/evaluate_mappo.py`
- `algorithm/mappo/__init__.py`
- `algorithm/train_mappo.py`
- `env/__init__.py`
- `env/factory.py`
- `tests/test_entrypoints.py`
- `tests/test_environment_regression.py`
- `tests/test_mappo_v2.py`
- `tests/test_navigation.py`
- `tools/__init__.py`
- `tools/audit_navigation.py`
- `tools/navigation_baseline.py`
- `tools/record_episode.py`

新增文档与清单：
- `docs/mappo_spec.md`
- `docs/navigation_audit.md`
- `docs/v2_migration_manifest.json`
- `docs/v2_file_changes.json`

原位重写：
- `tools/validate_coop3d.py`
- `README.md`
- `pyproject.toml`
- `docs/refactor_report.md`

三份场景YAML维持原参数；统一算法配置改为configs/mappo_coop3d.yaml，项目打包范围改为env、algorithm、tools。v1源码恢复副本为tmp/coop3d_v1_before_v2.zip，不属于正式可执行实现。

## C. 版本和变更分类

- **A 纯迁移**：模型数据、三自由度方程、RK4、随机/固定场景、Hungarian/nearest和五项奖励的合法输入行为。迁移前保存12个场景、873个转移；状态、12D观测、reward、分配、事件、累计数值计数及终止/截断全部精确匹配。版本字段和Critic状态不在不变断言中，因为它们是本次明确修改的接口。
- **B 获准的算法/接口调整**：环境身份coop3d-v2.0.0；全局状态60→90；Critic标量→5个体价值；Actor默认trim_centered；vector/rollout/checkpoint/评估同步；严格joint预算、按采样步调度与新日志；新增非学习导航基线。
- **C 数值与健壮性修复**：GAE/损失选择真正排除无效NaN；有效NaN/Inf显式失败；非法配置类型/非有限参数/故障步号检查；逐步numerical_delta和逐UAV裁剪事件，替代累计max统计；精确保留worker和episode累加器状态。没有改变合法输入的动力学方程、奖励权重或任务几何。
- **D 尚未解决/未验证**：没有收敛模型；3U5P标准场景的保守导航反馈在重分配后常超时；正式长训练性能与多种子泛化未验证；浏览器视觉交互验收未完成。

保持原值：g=9.81、dt=.25s、RK4、v=100–400m/s、γ±85°数值限制、250步、300m完成球、100m安全参考、独占匹配和五项巡检reward。没有武器机制，也没有为了成功率修改任务。

## D. Actor / Critic 接口与数值验证

Actor固定[N,12]，共享128×128 Tanh，输出3维Beta动作。全局状态前60维为最多五架按ID补零的局部观测；后30维为五个点的[x/distance_scale,y/distance_scale,z/altitude_scale,priority,completed,exists]，不存在点槽位全零。未分配点也进入Critic；完成点仍存在。Actor不接收90D全局信息。

Critic为90→128 Tanh→128 Tanh→5，接收[batch,90]，输出[batch,5]，每个UAV只用自己的ID槽位。3UAV场景后两头不参与损失，测试验证其输出层梯度为0。各自reward、advantage、return独立，禁止标量广播。

同一个全局状态、五个不同目标回报的监督拟合测试：五头Critic MSE=5.516228541e-08，旧式标量广播参考MSE=7.440000057。对照仅在测试内构造，没有保留旧算法分支。该监督单元测试不执行环境采样或PPO训练。

GAE覆盖真终止、超时真实末状态自举、跨局切断、当前有效下一失效、无效NaN、有效NaN/Inf报错、多环境部分失效同时超时。向无效观测/动作/logp/reward/value/global slot注入NaN后，更新结果仍与干净输入一致。

## E. 导航控制和任务可达性

控制公式：psi_des=atan2(dy,dx)，gamma_des限制±20°；psi_rate=clip(.8*wrap(error),±.06)，gamma_rate=clip(.6*error,±.03)。A=cosγ+v*gamma_rate/g；B=v*cosγ*psi_rate/g；normal=hypot(A,B)，roll=atan2(B,A)；tangential=sinγ+(225−v)/(g*6)。物理动作按[-π,π]、[-1,2]、[0,8]裁剪，再映射[0,1]。全部增益/限值/目标速度通过NavigationConfig配置，记录饱和及期望/实际角速度；只通过env.step执行，不直接改变飞行状态。

## F. 配平初始化与legacy对照

默认Beta总浓度16，alpha=[8,16/3,2]、beta=[8,32/3,14]；输出层bias=softplus_inverse(parameter−1)，输出权重gain=.01。softplus+1约束未改变，所有层可训练。

采用相同环境seed101/102、相同Torch初始化seed17和各局相同采样seed，共24局未训练策略对照。以下按有效UAV-step加权合并：

|执行方式|初始化|归一化动作均值|归一化动作std|实际μ/ζ/ζ̂均值|γ裁剪率|速度裁剪率|
|---|---|---|---|---|---:|---:|
|确定性均值|legacy|0.5084, 0.5056, 0.5083|0.0026, 0.0035, 0.0034|0.0530, 0.5169, 4.0662|83.873%|44.800%|
|确定性均值|trim_centered|0.5002, 0.3334, 0.1251|0.0000, 0.0000, 0.0000|0.0013, -0.0006, 1.0004|0.000%|0.000%|
|随机采样|legacy|0.5125, 0.5004, 0.5053|0.2408, 0.2326, 0.2372|0.0739, 0.4846, 3.9509|0.000%|28.600%|
|随机采样|trim_centered|0.4997, 0.3314, 0.1249|0.1211, 0.1131, 0.0782|-0.0017, -0.0058, 0.9992|0.000%|1.055%|

确定性legacy明显集中于法向过载约4；trim_centered接近1，并保留可量化的随机探索。物理动作包含环境闲置配平覆盖。累计任务进度与逐步分布、物理动作std、每规模/每种子的详细数据见navigation_audit.md和initialization_comparison.json；不以这些未训练策略结果声称收敛。

## G. 三规模小样本审计实际结果

主审计54局：每规模3种策略各5个相同随机种子，固定场景各策略只运行1次。以下标准任务参数没有调整：

|规模|Navigation固定场景|Navigation随机5种子|随机平均完成率|Trim随机成功率|Random随机成功率|
|---|---|---|---:|---:|---:|
|3U3P|成功率100%，完成率100%|5/5成功|100%|0%|0%|
|5U5P|成功率100%，完成率100%|5/5成功|100%|0%|0%|
|3U5P|成功率0%，完成率60%|1/5成功|76%|0%|0%|

额外手工可达3U5P任务79步完成5/5，各UAV完成数和后续分配事件均记录。它是单独标注的控制/任务接口测试，不替代标准随机场景。

3U5P失败的5局（固定1局、随机4局）均复核：250步超时，无γ/速度裁剪、无法向过载饱和，剩余点最近距离约2.78–5.46km。225m/s和.06rad/s转向上限对应3750m转弯半径、一圈104.72s，长于整局62.5s。重分配的近距离侧后方点会与剩余时限冲突；证据更符合保守导航控制的转向限制，不支持宣称积分错误、动作包线饱和或所有随机场景物理不可达。随机seed104确实完成标准3U5P。详见docs/navigation_audit.md。

每局原始数据包含所要求的成功率/完成率/完成数/长度、初末最近点距离、累计progress、飞行距离、有效UAV-step裁剪率、平均|γ|/速度、初始及逐步最小友机距离、分配切换，以及逐步控制/数值事件。3U5P包含个体完成数、第二/后续目标分配时刻及重分配完成事件。

## H. 自动化测试与静态检查

完整测试：**109 passed，0 failures，0 errors**，最近执行约7.63s。原66个参数化用例所对应的39个测试函数全部保留；只对获准变化更新维度/配置接口断言，新增873步语义回归、90D状态、个体价值拟合、配平、NaN、部分vector预算、导航、seed、checkpoint和CLI等测试。没有删除失败有效用例来制造全通过。

通过compileall、AST依赖检查、git diff --check；Git仅提示Windows行尾转换。环境无算法反向依赖；正式入口没有coop3d包导入；唯一正式仿真定义为env/cooperative_env.py。

受保护文件：本次开始捕获的40个原PDF/outputs文件均未变化；结合上一轮历史数据保护清单，共58个唯一文件SHA-256一致。没有删除历史模型、原outputs、JSON回归数据或PDF。

11份v2真实轨迹的1398个转移逐步重算五项奖励全部一致，状态有限，逐UAV数值事件和step delta求和一致。回放HTML内嵌库/数据并通过脚本语法检查。浏览器内视觉/交互未验证，语法检查不能替代浏览器验收。

机器证据：outputs/coop3d_v2/pytest.xml、final_audit.json；迁移来源/保护清单：docs/v2_migration_manifest.json。

## I. CUDA Smoke Test

环境：PowerShell conda activate uav，Python3.10.21、PyTorch2.11.0+cu128、RTX5060 Laptop GPU。CUDA已检查，没有CPU训练/审计回退。

仅两次短程Smoke调用：2环境、horizon8、1 PPO epoch、minibatch32、测试环境最大8步并在第4步使UAV1失效。短回合仅用于Smoke的故障/超时测试配置，正式三种场景仍250步。第一次严格总预算17，得到2个短rollout更新；第二次恢复到绝对总预算25，新增8个joint transitions和1次更新。

|sampled_steps|PPO update|本次有效个体样本|小批量更新数|Actor loss|Critic loss|entropy|approx KL|
|---:|---:|---:|---:|---:|---:|---:|---:|
|16|1|40|2|0.01281602|0.05580967|-2.692407|1.9304454e-05|
|17|2|3|1|0.00000024|0.00384076|-2.695317|0|
|25|3|23|1|0.00000002|0.01857367|-2.695529|0|

真实梯度更新改变Actor参数；累计25 joint transitions、3个PPO更新、4个小批量成对更新。预算17和25均精确达到，最后不整除worker数时只推进实际请求的worker。没有正式长训练或百万采样。独立Smoke评估未完成任务（短回合成功率0），不据此评判长期学习能力。

## J. Checkpoint、恢复、日志与录像

保存并重新加载后参数完全一致；原测试增强为：恢复后下一段动作、观测、reward、mask逐元素一致；再执行同一个后续更新，Actor/Critic权重和两个Adam状态逐元素完全一致。

当前Smoke latest为sampled_steps=25、update=3；best为sampled_steps=16、update=1。它们已分别评估，不能将latest录像当作best。旧60D真实历史模型在旧包已移除后仍能给出明确拒绝：incompatible checkpoint: environment/algorithm identity (60D models cannot resume as 90D)。

输出latest.pt、best.pt、run_config.json、training_metrics.jsonl、evaluation_history.csv、run_summary.json全部存在。配置包含版本、90D/5头维度、状态/掩码/预算协议；断点保存sampled_steps、两个Adam、全部RNG、所有worker状态、当前global state及未结束episode统计。默认恢复原配置，只有命令行总预算可以提高。

日志区分累计sampled_steps/joint_episode_count、本rollout训练回合、按有效UAV-step计算的rollout统计与独立评估。记录五项reward、目标距离/减少量、loss/entropy/KL/clip_fraction/gradient_norm、Beta动作mean/std、实际限幅率和分配变化。数值事件按显式delta计数，未使用累计最大值。

真实录像入口已实际执行：指定latest的独立evaluate、指定best的record_episode、规则navigation的record_episode、从JSON重新生成HTML。元数据包含实际checkpoint路径/角色/update/sample count或navigation标签。显示轨迹、速度方向、分配/点完成、控制、失效、数值事件、五项reward和任务进度。render_replay不导入环境或重新执行仿真。

可查看产物：
- outputs/coop3d_v2/navigation_audit/manual_reachable.html：手工可达3U5P，79步完成。
- outputs/coop3d_v2/navigation_recording/replay.html：标准3U5P，独立seed104的导航基线。
- outputs/coop3d_v2/smoke/latest_replay.html：真实Smoke训练latest。
- outputs/coop3d_v2/best_recording/replay.html：明确指定的best模型。
- outputs/coop3d_v2/checkpoint_evaluation/：latest的独立CLI评估。

## K. 仍未验证与限制

1. 未运行正式Ubuntu长训练、跨设备随机复现、多训练种子收敛/泛化实验；本轮不要求收敛模型，未用加预算代替修复。
2. 标准3U5P的导航反馈仍可能在重分配后超时。它有成功样本和独立可达场景，不能据此证明全部随机任务在当前限时都可完成或都不可完成。
3. 未完成浏览器内视觉交互验收：此前本地源自动访问未获权限批准，本轮未绕过；用户需本地打开HTML检查实际播放、滑块、旋转和状态面板效果。
4. 同一场景只有小样本诊断；初始化改善控制中心和裁剪不等于解决长期探索/优化。导航不是安全保证，也没有地形、能源或传感器模型。
5. v1模型有意不兼容v2；原文件均保留，不能作为90D/5头模型直接继续训练。

## L. Ubuntu正式训练及评估命令（仅交付，未执行）

```bash
conda activate uav
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name())'
python -m algorithm.train_mappo --env-config configs/coop3d_3U3P.yaml --algorithm-config configs/mappo_coop3d.yaml --device cuda --seed 1000 --num-envs 10 --total-sampled-steps 1000000 --output-dir outputs/mappo_3U3P_seed1000
python -m algorithm.train_mappo --env-config configs/coop3d_5U5P.yaml --algorithm-config configs/mappo_coop3d.yaml --device cuda --seed 1000 --num-envs 10 --total-sampled-steps 1000000 --output-dir outputs/mappo_5U5P_seed1000
python -m algorithm.train_mappo --env-config configs/coop3d_3U5P.yaml --algorithm-config configs/mappo_coop3d.yaml --device cuda --seed 1000 --num-envs 10 --total-sampled-steps 1000000 --output-dir outputs/mappo_3U5P_seed1000
python -m algorithm.evaluate_mappo --checkpoint outputs/mappo_3U3P_seed1000/best.pt --device cuda --seed-base 20000000 --episodes 5 --output outputs/eval_3U3P_best
python -m algorithm.evaluate_mappo --checkpoint outputs/mappo_3U3P_seed1000/latest.pt --device cuda --seed-base 20000000 --episodes 5 --output outputs/eval_3U3P_latest
python -m algorithm.train_mappo --device cuda --total-sampled-steps 2000000 --output-dir outputs/mappo_3U3P_seed1000 --resume outputs/mappo_3U3P_seed1000/latest.pt
```

以上预算只是可运行示例，未验证足够收敛。代码未自动提交Git或推送远端。

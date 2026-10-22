# 动态区域竞争环境与MAPPO验收记录

日期：2026-10-22。代码已实现，可独立创建环境、采样、优化、保存/恢复、独立评价、录制和交互回放。没有启动5000episode、百万步训练或大规模调参；不声称短验证获得有效竞争策略或论文性能。

## 1. 交付入口

|内容|路径/入口|
|---|---|
|新环境|`env.region_competition.RegionCompetitionEnv` / `RegionConfig`|
|两种正式场景|`configs/region_3v3.yaml`、`configs/region_5v5.yaml`|
|正式网络/训练配置|`configs/mappo_region.yaml`：12/60/3、128/128Tanh、10env、horizon1500、episode250|
|训练|`python -m algorithm.region_mappo train`|
|独立评价|`python -m algorithm.region_mappo evaluate`|
|单局录制和HTML回放|`python -m algorithm.region_mappo record`；`tools.region_replay.render_replay`|
|独立离线核算|`python -m tools.audit_region_trajectory trajectory.json`|
|环境规范|[region_competition_spec.md](region_competition_spec.md)|
|论文参数/假设/自定义分界|[region_mappo_alignment.md](region_mappo_alignment.md)|
|测试|`tests/test_region_environment.py`、`tests/test_region_mappo.py`，以及全部原有测试|

两个既有公共文件只增加维度/状态/优势接口及worker工厂钩子，默认仍为原巡检行为。没有修改旧巡检环境、Reward、网络、配置、checkpoint身份或历史实验。

## 2. 真实验证环境和结果

按AGENTS先探测WSL，`wsl --list --quiet`返回`Wsl/EnumerateDistros/Service/E_ACCESSDENIED`；因此使用Windows/PowerShell的uav环境。Python `D:\conda_envs\envs_dirs\uav\python.exe`，CUDA可用，GPU为NVIDIA GeForce RTX 5060 Laptop GPU。模型训练/评价/存档审计没有CPU回退。

全套测试最终结果：**184 passed，34.37s**，其中原有160项及新增24项通过。JUnit存于`tmp/region_dev/all_tests.xml`；新模块静态编译及`git diff --check`通过。旧fixture与旧测试预期没有放宽。

|要求|验证方法及实测结果|
|---|---|
|两队同一真实3DOF|两规模直飞解析值±50m/步，双方爬升/转弯符号、独立速度/gamma clip计数与惩罚均验证；没有给Blue直接改位置|
|场景/规则复现|相同seed/动作逐帧完全相同；cruise/region/region_safe均验证；区域位置连续、非静止；镜像布局与规则盘旋方向核对|
|12/60/3与mask|3v3和5v5尺寸、缺失槽0、无效槽冻结/观测0、非法双动作先验证再执行；未推进worker没有伪样本|
|独立计分/奖励|球穿越已知答案：-2→+2穿半径1球，dt=.25，驻留.125s；同队3机只得.125总分，各.125/3；双方同时占用0控制分但保留驻留；零本机位移tracking=0|
|1500/250共存|真实采1500 joint transitions、4500有效3机样本、6完整episode；truncated仅249/499/749/999/1249/1499位置；真实截止V_next=0，reset后的V=10未用于上一局|
|GAE人工答案|γ=.9、λ=.8案例优势[2.52,1,108]、return[3.52,3,118]；失效当步return5，之后无效槽0；真实终止与截止两种边界均核对|
|PPO实际更新|3v3尾部部分worker预算、5v5所有五价值头均更新；Beta合法、联合logp与训练动作对应；参数真实变化|
|10并行环境|另做10worker×1步CUDA Smoke：10独立worker PID，10联合步、30有效个体样本、1次真实PPO更新，Actor参数变化|
|完整恢复|保存后再采样/更新，与新trainer+新workers恢复后的rollout所有张量、Actor/Critic、两个Adam全部状态、worker状态逐位一致；Python/NumPy/Torch/CUDA RNG恢复；新旧协议双向拒绝误载|
|离线核查/回放|独立几何算法重算score/dwell/Reward/事件账本/动作映射；损坏记录能被识别；独立HTML含Red/Blue、动态球、逐帧slider/play和实际比分|

新开发过程中发现并修复的实际问题：无效动作NaN未清理会使JSON无效；同一世界坐标盘旋方向破坏镜像规则基线；同一几何交点浮点差约3.44e-15步产生虚假极短exclusive事件。分别以无效槽清零记录、镜像转向和专用子步分辨率解决，未改控制符号、任务半径、Reward系数或旧测试容差。

## 3. CUDA Smoke与完整回合证据

Smoke全部保存在新目录 `outputs/region_acceptance/`，与旧实验分离。

|运行|实际总joint步|更新/结束episode|明确覆盖的范围|
|---|---|---|---|
|`3v3_smoke`|17|2更新、4结束episode|E2、H8、episode4、epochs1、mini16；首rollout48个体样本，尾部3个；best实际step16/update1|
|`3v3_resume`|19（新增2）|3更新、4结束episode|从17步latest恢复，保留best16；CLI恢复链路可用|
|`5v5_smoke`|5|2更新、1结束episode|E1、H4、episode4、epochs1、mini16；首批20个体、尾批5个；best step5/update2|
|`ten_worker_smoke.json`|10|1更新|E10、H1、episode2、epochs1；验证十进程采样/真实优化，不冒称完整1500×10训练|

这些是调试配置，正式YAML参数没有因此缩短。Smoke时长4步，尚未到达区域，评价控制分为0是预期，不拿它证明任务学习成功。Checkpoint恢复的逐位测试采用相同后续预算/mini边界；改变绝对总预算导致不同部分rollout的优化次数时，不声称与另一次从0训练自然等价。

另以标准250步环境做以下完整回合验证，每项独立seed=30000000、30000001，轨迹只录第一局，统计保留两局。它们**没有进行训练更新**，与Smoke模型明确区分：

|目录/Red对Blue|两局平均Red/Blue区域秒数|平均相对分|说明|
|---|---|---|---|
|`3v3_untrained_250`：未训练Beta均值 vs region_safe|.629193 / 21.389971|-20.760777|update0，不能当已训练策略|
|`5v5_rules_250`：镜像region_safe vs region_safe|0 / 0|0|双方实际同时覆盖而contested，非静止或没有计分；平均每机驻留各11.893653s|
|`3v3_rules_cruise_250`：region_safe vs cruise|21.206870 / .413441|20.793429|验证不同独立规则可产生不同控制分；不是MAPPO性能|

第一局的原始轨迹显示：

- 3v3未训练Red每机实际航程14.67/13.94/14.65km，Blue14.12/14.01/14.07km；各自有效飞行步均250。Blue的gamma范围约−6.009°至7.332°，非静止目标，也非仅在水平面直接更新位置。
- 5v5两队镜像规则每机约14.01–14.11km，gamma约−3.580°至8.900°，高度约2860.8–3150.4m，速度约211.9–237.8m/s；两队均无速度/gamma裁剪。友机安全间隔违例率各4.48%，跨队接近率各1.68%，说明当前软避让不保证硬安全；该缺陷未通过改半径/阈值掩盖。
- 三份独立离线报告的最大score/reward/dwell/事件残差均小于6e-14，动作物理映射残差0；计算精度一致不等于对真实连续物理驻留误差的验证。

每个目录有`trajectory.json`、`evaluation.json`、`offline_audit.json`、`replay.html`。摘要是`outputs/region_acceptance/full_episode_checks.json`。可打开以下本地HTML：

- [3v3动态竞争回放](../outputs/region_acceptance/3v3_untrained_250/replay.html)
- [5v5双规则回放](../outputs/region_acceptance/5v5_rules_250/replay.html)
- [区域规则对巡航规则](../outputs/region_acceptance/3v3_rules_cruise_250/replay.html)

正式配置的一次完整rollout将是15000联合步；本轮未启动该规模优化实验，只验证单worker1500跨局及十worker极短采样这两个核心条件，不将此表写成“已完成论文训练”。

## 4. Ubuntu完整命令

先将本轮代码同步到实际Ubuntu仓库。以下第一段以正式E10/H1500配置示范**一个完整rollout、15000联合步**的训练和独立评价；它是供用户主动执行的命令，本轮没有自动执行。每次使用新的目录，CUDA不可用会明确退出。仓库路径通过输入获取，避免猜测远程Ubuntu路径。

```bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uav
read -r -p 'Ubuntu中1_uav仓库绝对路径: ' REPO
cd "$REPO"
python - <<'PY'
import sys, torch
from algorithm.common.runtime import require_cuda
require_cuda('cuda')
print(sys.executable, torch.__version__, torch.cuda.get_device_name())
PY
RUN="outputs/region_3v3_$(date +%Y%m%d_%H%M%S)"
python -m algorithm.region_mappo train \
  --env-config configs/region_3v3.yaml --train-config configs/mappo_region.yaml \
  --device cuda --total-sampled-steps 15000 --output "$RUN"
python -m algorithm.region_mappo evaluate \
  --checkpoint "$RUN/latest.pt" --device cuda --seed-base 20000000 --episodes 5 \
  --output "${RUN}_latest_eval"
python -m algorithm.region_mappo evaluate \
  --checkpoint "$RUN/best.pt" --device cuda --seed-base 20000000 --episodes 5 \
  --output "${RUN}_best_eval"
python -m algorithm.region_mappo record \
  --checkpoint "$RUN/best.pt" --device cuda --seed-base 30000000 \
  --output "${RUN}_best_record"
python -m tools.audit_region_trajectory "${RUN}_best_record/trajectory.json" \
  --output "${RUN}_best_record/offline_audit.json"
printf '运行目录: %s\n回放: %s\n' "$RUN" "${RUN}_best_record/replay.html"
```

评价会报告固定Blue配置及各seed的两队分数、驻留、飞行/安全/数值指标。评价只记录第一局，`record`明确只允许1局；Beta动作按均值执行并记录参数，不混称为训练随机轨迹。

5v5相同入口，换专用场景配置并使用新目录：

```bash
RUN5="outputs/region_5v5_$(date +%Y%m%d_%H%M%S)"
python -m algorithm.region_mappo train \
  --env-config configs/region_5v5.yaml --train-config configs/mappo_region.yaml \
  --device cuda --total-sampled-steps 15000 --output "$RUN5"
python -m algorithm.region_mappo evaluate \
  --checkpoint "$RUN5/latest.pt" --device cuda --seed-base 20000000 --episodes 5 \
  --output "${RUN5}_eval"
```

同环境/种子配对规则Red，无需改变checkpoint环境或Blue：

```bash
python -m algorithm.region_mappo evaluate \
  --checkpoint "$RUN/latest.pt" --red-rule region_safe --device cuda \
  --seed-base 20000000 --episodes 5 --output "${RUN}_rule_eval"
```

仅在明确选择继续时恢复，预算是绝对总joint步；这不是本轮自动执行的延长实验：

```bash
python -m algorithm.region_mappo train \
  --env-config configs/region_3v3.yaml --train-config configs/mappo_region.yaml \
  --device cuda --resume "$RUN/latest.pt" --total-sampled-steps 30000 --output "$RUN"
```

复核测试（同一uav环境），环境本身不要求GPU计算，模型测试和checkpoint审计需要CUDA：

```bash
python -m pytest -q
python -m compileall -q env/region_competition algorithm/region_mappo \
  tools/region_replay.py tools/audit_region_trajectory.py
```

## 5. 验收回答和仍存在的局限

双方真实三维机动、独立动作/积分/状态、12/60/3协议、MAPPO实际采样与更新、1500/250跨局、Reward独立重算、完整Checkpoint恢复均已完成验证。原有巡检全部测试通过，既有89个outputs/PDF文件结束时以SHA256复核保持不变；没有删除、覆盖原模型或原轨迹。

论文明确对齐的是X的3DOF、指定物理数值及表III列举的网络/优化/采样参数。Actor共享、五价值头、Beta初始化、优势标准化、mini单位、截止自举和更新边界存档等是公开细节不足下的工程选择；动态区域、12维新语义及Reward完全属于本项目。60D有剩余时间但仍缺比分历史和完整空间/对方状态，是压缩的部分可观测Critic输入。当前规则安全控制存在违例，未训练Actor竞争表现差；本轮完成的是可验证基线和工具链，不是收敛或硬安全结论。

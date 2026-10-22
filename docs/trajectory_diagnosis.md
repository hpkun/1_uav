# 三维飞行轨迹诊断：实证范围与待补材料

> 本文数值来自整改前历史轨迹，文件未被改写。当前v3动力学/γ特征已改变并拒绝旧checkpoint直接恢复；不能将这些轨迹标成v3结果。当前代码变更见 [整改报告](xu2026_rectification.md)。

审计日期：2026-10-21。**本地没有200000步实验对应的v2模型或轨迹，因此不能实证解释该训练为何失败。** 用户报告的 Best 0/50、完成率0.67%，Latest 0/50、完成率2.00%，规则导航50/50，记录为待核验的外部实验结果。它们说明报告中学习策略没有完成整局任务，不能单凭这些比例判断是偏航、爬升、饱和、探索、奖励还是更新错误。

本轮直接分析已有JSON，不重新训练、不重新生成旧实验、不改写原轨迹。论文差异与数学审计见 [paper_alignment_audit.md](paper_alignment_audit.md)。

## 1. 实际文件身份

扫描工作区全部 `.pt` 和 trajectory JSON，CUDA门禁后只读核验兼容v2 checkpoint元数据：

|本地文件|实测身份|允许使用的证据范围|
|---|---|---|
|`outputs/coop3d_v2/smoke/best.pt`|sampled_steps=16，updates=1，已结束训练episode=2|短Smoke模型，不能当200000步Best|
|`outputs/coop3d_v2/smoke/latest.pt`|sampled_steps=25，updates=3，episode=2|短Smoke模型，不能当200000步Latest|
|`outputs/log_maintenance_checks/smoke_checked/{best,latest}.pt`|sampled_steps=2，updates=1|日志验证模型，不能评价学习效果|
|`outputs/coop3d_acceptance*`模型|v1/不兼容当前v2协议|只作为历史证据，不加载到v2训练器|
|`outputs/coop3d_v2/best_recording/trajectory.json`|v2，3U3P，seed30000001，9帧/8步，provenance=best/16步/update1|2秒控制响应；第4步UAV1注入失效，max_steps=8|
|`outputs/coop3d_v2/checkpoint_evaluation/trajectory.json`|v2，3U3P，seed30000000，8步，latest/25步/update3|同样有第4步失效；与上一行seed不同，不能作为受控Best/Latest比较|
|`outputs/coop3d_v2/smoke/latest_trajectory.json`|v2，seed20000000，8步，latest/25步|独立短录制；同样不代表20万步|
|`outputs/coop3d_v2/navigation_recording/trajectory.json`|v2，3U5P，seed104，179步，规则navigation|一局真实成功轨迹；不能代替用户50局统计|
|`outputs/coop3d_acceptance_v1/3U3P/trajectory.json`|v1，3U3P，seed10000000，250步，无checkpoint provenance|可证明历史记录的控制/爬升现象，不能把原因迁移给v2训练模型|

本轮开始时outputs内共有11份既有轨迹JSON（7份历史v1、4份v2），没有以200000标识的provenance；本地兼容v2 checkpoint中最大sampled_steps仅25。旧v1 JSON缺模型身份，不猜测它来自哪一个训练更新。本轮另外在tmp内生成了一份2步新录制用于CLI验证，它不纳入历史轨迹归因。

## 2. 逐架实测轨迹

以下距离固定参考**初始分配点**，因此可比较起末；重新分配后的逐步距离另按每步前分配计算。规则导航在完成旧点后继续飞往新点，其末尾离旧点变远完全正常，不能误判退步。完整逐点距离/分段位置/动作/奖励见四份生成的文本证据：

- [规则导航逐架证据](trajectory_evidence_navigation.md)
- [短Smoke Best逐架证据](trajectory_evidence_smoke_best.md)
- [短Smoke Latest逐架证据](trajectory_evidence_smoke_latest.md)
- [历史v1逐架证据](trajectory_evidence_legacy_v1.md)

### 2.1 v2 短Smoke模型：接近配平直飞，时间与任务不足以检验导航学习

|轨迹/UAV|有效步/路程m|初始点距离 起→末m|速度min/mean/max m/s|γ范围°|ψ起→末°；累计有符号转向°|初始点航向误差 起→末°|
|---|---|---|---|---|---|---|
|Best/U0|8 / 456.871|10239.982→9808.655|228.303/228.436/228.570|-0.929→-0.908|-26.522→-26.551；-.029|17.893→18.746|
|Best/U1|4 / 222.045|9702.810→9506.556|221.775/222.165/222.316|-3.307→-3.300|25.363→25.348；-.015|-26.922→-27.513|
|Best/U2|8 / 447.361|7840.406→7397.239|223.302/223.680/224.060|2.066→2.083|16.048→16.019；-.028|-6.554→-6.920|
|Latest/U0|8 / 462.885|10382.326→9926.101|230.889/231.443/231.999|-3.478→-3.466|20.744→20.721；-.023|-7.223→-7.535|
|Latest/U1|4 / 242.092|9193.271→8958.604|241.760/242.239/242.423|-4.103→-4.099|-26.090→-26.100；-.010|13.845→14.228|
|Latest/U2|8 / 450.262|7032.626→6585.823|224.492/225.131/225.771|3.523→3.534|-.208→-.229；-.021|-1.498→-1.578|

位置端点、各点最近距离、分段转移在证据文件，不由PPO loss推测。Best三个UAV平均执行物理控制分别约 `[-.005921,-.002448,.995578]`、`[-.005965,-.002528,.995697]`、`[-.005605,-.002388,.995916]`；反映射的执行归一化控制约 `[.49906,.33252,.12445]`。Latest相应平均控制约 `[-.004744,-.003987,.995661]`、`[-.004470,-.003854,.995539]`、`[-.004208,-.003660,.995917]`，执行归一化约 `[.49925,.33200,.12446]`。

这些控制接近trim，极短2秒内转向很弱，部分机的目标航向误差变大；仍有正progress，因为初始航向本来朝向任务区域。Best/U0、U1、U2的progress累计=.849398/.392507/.886335，efficiency=-.08/-.04/-.08；Latest=.856715/.469333/.893605，efficiency相同。其余分项均0。全部有效步无γ/速度限幅，无动作范围1%端点饱和。

第4步UAV1失效后U0由P0重新分配P1，故“每步已分配点误差”与“固定初始点误差”不是同一个指标。8步飞行不足500m，但任务初始距离约7–10km，0完成本身不能证明策略训练失败。这些现象不适用于未提供的20万步模型。

### 2.2 v2 规则导航：真实三维到达与重新分配

3U5P/seed104在179步（44.75s）完成5/5。事件：step100 U1→P3，118 U2→P2，134 U0→P4，158 U1→P1，179 U0→P0；独立事件无重复。U0/U1/U2分别完成2/2/1点。

|UAV|路程m；位置起→末m|初始点距离起/末/最近m|速度min/mean/max|γ范围°|ψ起→末°；净/总转向°|分配期间平均绝对航向误差°|
|---|---|---|---|---|---|---|
|U0|10058.830；[-16,-134,2765]→[9819,1058,2170]|P4 7719.742/2284.509/21.346|223.101/224.775/225.051|-7.115至3.695|34.235→-8.303；-42.538/42.538|2.998|
|U1|10134.152；[298,60,3062]→[10246,1050,2878]|P3 5943.506/4155.938/24.882|225.150/226.484/235.701|-18.587至5.896|4.780→6.619；1.839/1.839|.0582|
|U2|10164.556；[-234,198,2560]→[9790,156,3640]|P2 7008.978/3144.787/13.939|225.104/227.175/241.233|-1.216至8.869|-15.105→9.948；25.053/25.053|2.976|

平均执行物理控制：U0 `[-.327078,-.054800,1.12531]`、U1 `[.039492,-.042731,.802026]`、U2 `[.175169,.069824,1.15603]`。μ分别跨越[-1.34893,0]、[0,.981561]、[0,1.35398]rad；正常转向和法向负载变化确实引起了水平航向、垂向轨迹变化。U1/U2分别有21/45步idle，执行配平，所以不能把全局平均法向负载低于1直接判成错误控制。

三机全部无γ/速度限幅、无动作端点1%饱和；实际progress累计19.993811/17.748477/14.949850，completion20/20/10，efficiency各-1.79，terminal各13，separation均0。终奖13严格等于 `20-floor(179/25)`。这证明该局环境和动作映射允许成功三维导航；不同seed/规模/故障配置的Smoke模型不能与之做因果比较，更不能替代用户所说的同种子50局对照。

### 2.3 历史v1：确实存在高法向、近垂直爬升和持续偏航

此项特意标为**历史v1证据**。它展示应如何用物理量判断失败，而不是证明当前v2有相同问题。

|UAV|初始点距离起→末m|末高度m；路程m|速度min/mean/max|γ均值/最大°；首次85°步|85°后态数/250；100m/s后态数/250|航向净/总转向°；平均绝对目标航向误差°|
|---|---|---|---|---|---|---|
|0|7744.193→9972.435|11477.364；9142.488|100/146.533/231.975|76.331/85；46|205；98|-528.681/823.215；90.888|
|1|9902.977→11153.782|11234.163；9538.994|100/152.853/241.881|75.985/85；48|203；88|-547.453/809.895；86.453|
|2|8143.016→11462.800|11840.599；9804.143|100/157.085/249.058|76.154/85；48|203；82|-532.876/803.816；99.796|

全局记录的gamma_clips=611、speed_clips=268、nonfinite_states=0。上表逐机“状态在界限”计数来自实际state，不能混称为逐机裁剪事件（v1未逐机记录）。0个点完成。

平均执行归一化控制分别约 `[.492493,.519775,.463273]`、`[.492101,.518369,.463701]`、`[.491772,.517491,.461351]`；对应法向负载3.70619/3.70961/3.69080。bank虽很小，法向投影远高于水平配平所需1，按实际动力学推动γ快速增大。γ接近85°后，`1/cosγ`放大偏航，且切向约.55小于sinγ≈1，使速度下降到100。轨迹末高度已超11km，空间距离恶化，progress=-4.501313/-5.443115/-3.685104、efficiency各-2.5。

这条“执行法向过高→爬升触界→偏航放大/减速→远离任务”的物理解释由实际控制和状态支持。动作自身却**没有**端点饱和，说明不能把“状态触界”写成“Beta动作饱和”。它与当前v2的trim中心控制明显不同；不能据此宣称20万步还在复现旧初值问题。

## 3. 诊断判定表

|轨迹现象/问题|数值证据|原因判断|是否已证实|下一步验证方法|
|---|---|---|---|---|
|用户20万步Best/Latest均未整局成功|用户提供0/50，完成率.67%/2%，本地无对应模型/轨迹|只能确认报告描述的结果，不能确定机理|外部结果待本地复核|取得checkpoint/provenance、原seed清单、相同配置轨迹|
|历史v1爬升而非导航|三机法向均值约3.7，γ最大85°，末z>11km；距离恶化|实际负载驱动爬升，极限几何放大偏航并减速|在该v1记录中已证实|不要移植归因；先看20万步法向/γ时间序列|
|v2短模型接近直飞且航向纠偏弱|2s净转向约-.010至-.029°，bank均值绝对值<.006rad|仅1–3次更新、近配平控制；不够检验长程导航|行为已证实，学习原因/外推未证实|200000步同seed完整250步，检查误差与bank符号/量级|
|v2规则导航可三维到达|179步5/5，五事件，γ约-18.6至8.9°，三机路程约10km|该局具有可行控制；动力学没有阻止导航|该局已证实，非全场景可达性定理|同一真实checkpoint配置、同初始状态配对规则策略|
|奖励计算疑似错位/重复|11份、6859个UAV转移槽重算，progress残差≤1.4e-17，其余0，无重复点事件|所查记录与当前规范对齐|已证实（有限样本）|对用户原始200000步轨迹运行同一只读重算|
|均值动作与训练采样响应不同|trim初始Beta的Ecosμ=.744614；γ=0时期望dγ约-.638°/s，均值动作dγ=0|非线性动力学使“均值配平”不能保证采样飞行配平|初始化数学性质已证实；是否影响20万步未知|原模型原始α/β，另行受控随机/均值策略录制；本轮未运行随机模型评价|
|密集progress替代了真正完成|225m/s直飞一步约+.1025净reward，完成才有+10与终奖；Smoke只有progress|策略可能优先早期局部收益/探索未触达完成|假设，需要进一步实验验证|完整轨迹检查先减距后偏离/过冲、完成附近动作与return分项|
|时序尺度可能不匹配|γ^100≈.0090，γ^200≈8.12e-5；原始目标常在数km外|长期完成奖励受折扣抑制，但dense reward也可能足够|数值已证实；失败因果未证实|先统计第一次接近300m/事件时间及对应优势，不直接改γ|
|90D未包括时间|完全相同90D与相同动作，可得终奖20或19（诊断时钟0/24）|有限时域价值发生时间别名；截断自举目标需澄清|信息缺失已证实；对失败贡献未知|单独定义剩余时长及截止语义，再决定短实验|
|best可能不优于latest的新50局|best按固定5局验证三元组选择；用户新结果latest完成率更高|验证选择偏差/不同seed/稀有事件均可能|没有证据表明best文件选错|同50新seed重新比对，两模型分别打印step/update，不混用role|
|原始Beta分布未记录|本地JSON只有physical_actions和state/reward|不能恢复idle前候选动作，也不能恢复α/β/采样概率|已证实缺字段|用新增`--policy-diagnostics`生成全新轨迹；旧文件不补写|

## 4. 只读工具的范围

```bash
conda activate uav
python -m tools.trajectory_audit /absolute/path/trajectory.json --output /absolute/path/new_audit.md
```

工具只读JSON，不导入env、torch或训练器、不加载模型、不步进仿真；输出默认stdout，`--output`以独占创建模式拒绝覆盖已有文件，也拒绝写回输入。它统计逐UAV实际位置路径、固定点/每步分配距离、速度/γ/ψ/heading error、执行控制与反归一化、转向、逐机数值事件、奖励重算、任务切换/完成/故障行。统计控制饱和定义为归一化值距端点≤1%范围，与状态clip分开。

原轨迹没有Beta参数时明确显示missing；执行控制的反映射不冒充原策略动作。新诊断字段齐全时另外报告原始归一化动作、α/β、logp/entropy，并只对**有效且有任务**转移核算策略动作→物理控制残差；idle覆盖单独排除。均值评价动作标为mean，不冒充训练随机Beta样本。

## 5. Ubuntu 完整录制命令：不训练、不覆盖旧数据

本机WSL权限错误，因此以下为用户在实际有200000步模型的Ubuntu端执行的命令。只需输入仓库路径、原实验目录及seed。使用模型自带环境配置，不能用默认3U3P替换原实验。若在WSL，可输入仓库路径 `/mnt/c/Users/HPK/Desktop/1_uav`；远程Ubuntu应输入实际路径。

默认只录制**一个种子的Best/Latest/规则导航三局**，先核验身份和行为。seed应填原50局清单中的一个；留空采用30000000，这是新的配对诊断种子，不冒充原50局复现。输入原seed遇训练种子隔离错误时须检查原评价独立性，不能去掉检查。

```bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uav
read -r -p 'Ubuntu仓库绝对路径: ' REPO
cd "$REPO"
read -r -p '含best.pt/latest.pt的原200000步实验目录: ' RUN
RUN=$(realpath "$RUN")
test -f "$RUN/best.pt"
test -f "$RUN/latest.pt"
read -r -p '原评估种子中的一个 [默认30000000]: ' SEED
SEED=${SEED:-30000000}
mkdir -p outputs
AUDIT=$(mktemp -d "$PWD/outputs/paper_trajectory_audit.XXXXXX")

python - "$RUN" "$AUDIT" <<'PY'
import json, sys
from pathlib import Path
import torch, yaml
from algorithm.common.runtime import require_cuda
from algorithm.common.checkpoint import read_checkpoint
require_cuda('cuda')
print('Python:',sys.executable,'GPU:',torch.cuda.get_device_name())
run, out = map(Path,sys.argv[1:])
best, latest = [read_checkpoint(run/(role+'.pt')) for role in ('best','latest')]
assert latest['sampled_steps']==200000, '不是所述200000步latest，请核对实验目录'
assert best['env_config']==latest['env_config'], 'best/latest环境不同，不能配对'
assert best['train_config']==latest['train_config'], 'best/latest算法配置不同'
with (out/'env.yaml').open('x',encoding='utf-8') as f:
    yaml.safe_dump(latest['env_config'],f,allow_unicode=True)
identity={role:{k:d[k] for k in ('sampled_steps','updates','training_episode_count','environment_version','algorithm_version')}
          for role,d in [('best',best),('latest',latest)]}
with (out/'identity.json').open('x',encoding='utf-8') as f:
    json.dump(identity,f,ensure_ascii=False,indent=2)
print(json.dumps(identity,ensure_ascii=False,indent=2))
PY

sha256sum "$RUN/best.pt" "$RUN/latest.pt" > "$AUDIT/checkpoint_sha256.txt"
for ROLE in best latest; do
  python -m tools.record_episode --checkpoint "$RUN/$ROLE.pt" \
    --device cuda --seed "$SEED" \
    --policy-diagnostics --output "$AUDIT/${ROLE}_${SEED}"
  python -m tools.trajectory_audit "$AUDIT/${ROLE}_${SEED}/trajectory.json" \
    --output "$AUDIT/${ROLE}_${SEED}/audit.md"
done
python -m tools.record_episode --policy navigation --env-config "$AUDIT/env.yaml" \
  --device cuda --seed "$SEED" --output "$AUDIT/navigation_${SEED}"
python -m tools.trajectory_audit "$AUDIT/navigation_${SEED}/trajectory.json" \
  --output "$AUDIT/navigation_${SEED}/audit.md"
sha256sum --check "$AUDIT/checkpoint_sha256.txt"
printf '新诊断文件位于: %s\n' "$AUDIT"
```

产生新模型身份文件、三个trajectory.json、三个audit.md和三个replay.html，原模型/原outputs子目录不变。录制模型强制CUDA；只读JSON统计不需要GPU计算。新数据只描述确定性均值评价，不能代表训练时的随机探索轨迹。

若确需复核原50局统计，且确认原种子恰为连续区间，可在同一shell继续以下命令；只使用原seed_base，不能事后挑选易成功种子。CLI评价只录第一局轨迹，所以逐架分析仍需上面的独立录制。导航50局按同一模型配置、同一seed范围计算。以下为可选命令，本轮未执行：

```bash
read -r -p '原50局连续seed_base: ' SEED_BASE
python -m algorithm.evaluate_mappo --checkpoint "$RUN/best.pt" --device cuda \
  --seed-base "$SEED_BASE" --episodes 50 --output "$AUDIT/eval50_best"
python -m algorithm.evaluate_mappo --checkpoint "$RUN/latest.pt" --device cuda \
  --seed-base "$SEED_BASE" --episodes 50 --output "$AUDIT/eval50_latest"
python - "$AUDIT" "$SEED_BASE" <<'PY'
import json, sys
from pathlib import Path
from env.config import EnvConfig
from algorithm.common.runtime import require_cuda
from tools.audit_navigation import run_episode
require_cuda('cuda')
out=Path(sys.argv[1]); base=int(sys.argv[2])
ec=EnvConfig.load(out/'env.yaml')
rows=[run_episode(ec,base+i,'navigation') for i in range(50)]
summary=dict(seed_base=base,episodes=50,
             success_count=sum(int(r['success']) for r in rows),
             completion_rate=sum(r['completion_rate'] for r in rows)/50,
             per_episode=[{k:r[k] for k in ('seed','success','completion_rate','completed_waypoints','episode_length')} for r in rows])
with (out/'eval50_navigation.json').open('x',encoding='utf-8') as f:
    json.dump(summary,f,ensure_ascii=False,indent=2)
print('navigation:',summary['success_count'],'/50;',summary['completion_rate'])
PY
```

如果原50局不是连续种子，不能用这一段声称严格复核，需用原seed清单逐项调用评价。

## 6. 对20万步失败的当前回答及下一步

1. **已证实**当前实现与论文研究任务/观测语义/rollout协议不同，不能用论文成功率为导航预期背书；PPO/GAE核心数学检查及短回归目前未发现确证Bug。
2. **已证实**旧v1有大法向爬升/偏航失败链；v2短模型是近直飞，规则控制该局可成功。三者文件身份、seed和配置不同，均不能给200000步失败作替代归因。
3. **尚未证实**20万步的失败具体原因。完成稀疏与折扣、随机/均值动作差异、长期偏航或过冲、时间表征缺失及验证选模偏差都是待区分的解释，不是已发现的算式Bug。也不能因为某一loss下降就说导航学会了，或因为负熵就说训练崩溃。

最值得做的1–3项工作：

1. 先取得真实模型身份与原50局seed清单，用上面的同配置配对三局确认实际飞行现象；从已失败seed中再补两例，避免只挑代表性截图。
2. 对新轨迹逐机检查误差→原始Beta动作→执行控制→位置/γ/ψ→progress/事件这一条时序链，区分“没有转向”“错误法向”“随机动作失稳”“接近后过冲/重分配”。若需要解释训练探索，还需另行采集明确标为stochastic的轨迹；本轮不自动扩展实验。
3. 只有前两项定位后，才设计单一变量的时间表征/截止bootstrap短诊断，或其他被轨迹直接指向的问题。不得同时更改Reward、gamma、任务条件或自动延长200000步。

本轮验证共41项针对性单元/极短集成检查通过；新增记录开关不改变状态、奖励、评估结果或CUDA RNG。另以既有2步Smoke模型录制并审计2步，三机策略到执行控制残差均0；Ubuntu命令通过Bash及内嵌Python语法检查。所有既有outputs和两个原始PDF共89个文件SHA256在结束时复核不变，不改模型或历史轨迹。

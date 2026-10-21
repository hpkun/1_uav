# Coop3D v2.0 日志、指标和浮点回归维护报告

日期：2026-10-21（Asia/Shanghai）。本次是日志与诊断维护，没有改变环境或算法版本身份，没有运行正式长训练或自动延长已有训练。

## 实际修复

1. **Actor loss 指标含义错误**：原实现实际 backward 的是 `policy - entropy_coef*entropy.mean()`，记录的却是 `policy.item()`。现在同时记录 `policy_loss` 和 `actor_total_loss`，`actor_loss` 作为真实 `actor_total_loss` 的兼容别名。critic_loss、entropy、approx_kl、clip_fraction、gradient_norm 保持原计算。
2. **跨平台浮点回归过严**：递归compare对浮点数组使用assert_allclose，默认rtol=atol=1e-12，浮点标量使用相同容差。整数、bool、分配ID、事件结构/数量和终止状态继续严格检查，字典键和序列形状/长度也严格比较；浮点NaN/Inf即使两侧相同也失败。失败包含字段路径、实际误差及容差。
3. **终端难以阅读**：独立progress模块使用标准库格式化两行TRAIN进度、独立EVAL和最终DONE摘要，没有ANSI、光标控制或额外依赖；保留机器可读JSONL和CSV。
4. **恢复后日志/模型冲突风险**：增加只读预检，先检查日志与checkpoint的sampled_steps、update、重复/倒序、目标更晚checkpoint及best来源，再进行任何目录写入、模型复制或trainer启动。冲突明确拒绝，原文件不动；缺失/落后日志明确提示缺口，不补造数据、不自动截断。

## 学习行为和模型兼容性

保持不变：Actor/Critic网络及维度、动作接口、环境动力学与五项Reward、GAE和PPO公式、Adam参数、TrainConfig持久化字段、版本身份和Checkpoint内容协议。

验证依据：

- 修改前update函数冻结在测试fixture中，以同一随机种子和相同rollout执行旧/新更新，Actor/Critic参数及两个Adam状态逐元素完全相等。
- 拦截真实Tensor.backward，实际Actor损失的批次均值与新actor_total_loss一致。该测试取entropy_coef=.07以明确区分两种loss：policy_loss=0.12449807735780875，actor_total_loss=0.31314245114723843。
- 去除纯日志记录/返回语句后，新旧update的计算AST完全相同；finite_selected、clean_global、gae、初始化、tensor和collect的AST均未改变。
- 87个受保护文件SHA-256一致，包含全部环境源码、网络、TrainConfig、Checkpoint/协议源码、原始回归fixture及本次开始前已有outputs。历史日志和模型未重写、未覆盖。
- 已有真实v2 latest.pt成功加载并恢复到同配置trainer，参数完全一致；已有run的日志预检无冲突。没有恢复环境采样或继续旧run。

**本工作区未找到用户提及的200000步实验目录。** 实际读取并验证的是仓库已有v2历史Smoke目录，包含3条更新记录，checkpoint为25 joint transitions。不能据此声称已审计那一份200000步模型或其训练效果；新工具可在Ubuntu上直接读取该真实目录。

## 历史loss字段的处理

修复前v2 JSONL中的actor_loss实际是policy_loss，旧报告中的相应表格也应按此理解。历史文件全部原样保留。离线工具依据新字段是否存在判断语义：

|数据|policy_loss|actor_total_loss|actor_loss|
|---|---|---|---|
|旧v2记录|从旧actor_loss读取|未记录，显示N/A|原始内容不变，解释为policy|
|新记录|纯PPO clipped surrogate|实际含熵项backward损失|与actor_total_loss一致的别名|

没有使用当前entropy_coef反推并伪造旧实验的总损失。不同大小小批量仍按原有等权方式求指标均值。Beta微分熵可为负，总损失可能大于policy loss，这不是新的符号错误。

## 真实终端输出示例

以下来自本次唯一完成的新目录CUDA Smoke，不是硬编码的200000步展示数字。它只使用1环境、2步rollout、1个PPO epoch，执行2个joint transitions和1次真实更新；独立评估n=1、成功率0，不能标记为已收敛。

```text
[EVAL] Steps 2 | Episodes n=1 | Model current
       Success 0.0% (0/1) | Completion 0.0% | Completed waypoints 0.0/3
       Mean Return 0.207 | Mean Episode Length 2.0 | BEST UPDATED
[TRAIN] 2/2 (100.0%) | Update 1 | Episodes 1 | Speed 2 joint steps/s | ETA 00:00:00 | Train: Success 0.0% | Completion 0.0% | Return 0.184
        Task: Distance 8,474 m | Distance reduction +306 m | PPO: Policy 0.0000 | Actor total 0.0027 | Critic 0.0100 | KL 0.00000 | Clip 0.00% | Entropy -2.69 | Flight: Speed clipping 0.0% | Gamma clipping 0.0%
[DONE] Sampled steps 2 | PPO updates 1 | Training episodes 1
       BEST: n=1 | Success 0.0% | Completion 0.0% | Completed 0.0/3 | Return 0.207
       LATEST: n=1 | Success 0.0% | Completion 0.0% | Completed 0.0/3 | Return 0.207
       best.pt: C:\Users\HPK\Desktop\1_uav\outputs\log_maintenance_checks\smoke_checked\best.pt
       latest.pt: C:\Users\HPK\Desktop\1_uav\outputs\log_maintenance_checks\smoke_checked\latest.pt
       training_metrics.jsonl: C:\Users\HPK\Desktop\1_uav\outputs\log_maintenance_checks\smoke_checked\training_metrics.jsonl
       evaluation_history.csv: C:\Users\HPK\Desktop\1_uav\outputs\log_maintenance_checks\smoke_checked\evaluation_history.csv
       run_summary.json: C:\Users\HPK\Desktop\1_uav\outputs\log_maintenance_checks\smoke_checked\run_summary.json
```

普通进度按原console_interval_sampled_steps触发并保留最终输出；独立评估按原调度触发，使用evaluate返回的结果，显示实际n、成功数/总数、完成点数/总点数和BEST UPDATED/UNCHANGED。没有训练回合结束时训练Success/Completion/Return为N/A，不以0%代替缺失。

Reward保留3位、loss 4位、KL 5位、entropy 2位，比例转为百分数，距离用m、speed用整数joint steps/s，ETA为hh:mm:ss。速度是当前恢复会话新增步数/会话墙钟时间，包含评估与保存耗时，未把过去采样计入速度。展示函数不会修改原始指标或随机数状态。

新Smoke机器记录中，policy_loss=3.377596727e-07，actor_total_loss=0.002691790462，actor_loss=0.002691790462。终端舍入不会改变JSONL数值。

辅助验证脚本首次启动漏了Windows spawn主入口保护，在worker正式运行和环境采样前中止；已补上保护，首次尝试的少量配置/空CSV保留在新的检查目录，未删除任何历史实验。完成的一次Smoke位于smoke_checked；没有再执行额外长训练。

## 只读离线报告工具

新增 `tools/report_training.py`：只读取training_metrics.jsonl、evaluation_history.csv、run_summary.json、run_config.json，不加载模型、不启动环境、不依赖torch或CUDA，不产生输出文件（默认写stdout）。

显示规模和版本、采样步数/update、独立评估表、best/latest及实际记录路径、首10/尾N次优化均值、任务距离/训练回合趋势、限幅率、最后rollout的三维Beta动作mean/std及五项Reward。均值按update等权；短历史首尾窗口重叠会提示。缺失字段显示N/A，不推断不存在的模型指标或评估样本数。

旧日志读取测试和实际旧v2读取均通过；新格式Smoke日志也已由同一工具读取。子进程测试禁止导入torch、env、algorithm.mappo后仍能完成CLI报告，确认它不触发模型或CUDA。

诊断包括非有限数据、长期零成功率/完成率不增长、目标距离逐次增长、持续高限幅率、异常KL和裁剪前梯度范数；至少3条记录、80%超过阈值才提示持续偏高。默认阈值为γ/速度裁剪10%、KL .03、梯度范数5.0。所有提示都是排查线索，不是自动失败或收敛判定；小样本n始终明确显示。

## 中断恢复与历史安全

- 日志或summary晚于断点、重复/倒序记录、更晚latest/best、错误best-score来源：抛出ResumeLogConflict，在任何覆盖/追加前停止。
- 日志落后或缺失：只读预检提示历史缺口，后续从断点继续，缺口不伪造。空CSV才创建表头，有有效历史的CSV不会重建。
- 默认不自动备份后裁剪、删除或覆盖冲突；选择一致的checkpoint与日志来源后才允许继续。若用户需要回退，应先单独保全原目录并审核恢复方案。
- best/latest独立评价，最终summary补充best实际sampled_steps和update来源。这些是运行摘要字段，不是TrainConfig或Checkpoint持久化协议变更。
- 新formatter、ETA、只读预检不使用随机数。测试验证展示前后Python/NumPy/Torch RNG不变。

## 测试和执行结果

- 完整测试：**150 passed，0 failures，0 errors**，约8.50s；原109项保留，新增41项测试。
- 浮点测试覆盖1e-14噪声可通过、1e-6真实差异失败、ID/事件/奖励/布尔变化失败、NaN/Inf失败；原12场景873步参考轨迹全部通过。没有重生成fixture、没有改动力学。
- 日志测试覆盖loss/backward一致、纯policy与total分别记录、旧格式解读、N/A、TRAIN/EVAL分离、最佳提示、ETA/速度、非有限输出防护、CPU离线读取、恢复重复/未来/缺失/落后/来源冲突及完全相同的参数更新。
- compileall、git diff --check通过；Git仅有Windows行尾转换提示。
- 当前机器PowerShell/uav、CUDA RTX5060 Laptop GPU。未在本轮实际Ubuntu中运行测试；已在本机模拟跨平台浮点量级并验证参考轨迹，Ubuntu仍应执行下面的回归命令。
- 没有修改、恢复或延长用户的200000步训练，也没有百万级采样。

证据目录：`outputs/log_maintenance_checks/`：pytest.xml、focused_tests.xml、final_audit.json、compatibility_audit.json、historical_v2_report.txt、new_metrics_report.txt、console_smoke_checked.txt及新Smoke目录。

## Ubuntu可直接执行的离线命令

将run-dir替换为实际200000步实验目录；此命令不需要CUDA：

```bash
conda activate uav
python -m tools.report_training --run-dir outputs/你的训练目录 --last-updates 10
```

本仓库已有历史记录的可运行示例：

```bash
python -m tools.report_training --run-dir outputs/coop3d_v2/smoke --last-updates 10
python -m pytest -q tests/test_environment_regression.py tests/test_regression_tolerance.py
```

没有自动提交Git或推送远端。具体200000步实验的收敛情况未在本工作区核实。

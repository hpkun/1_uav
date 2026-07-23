# Reward Definition

## 1v1

`zheng_series_1v1` 使用明确命名的论文分段距离、高度、角度、速度、稀密和事件分量。`debug_linear_distance_reward` 与 `debug_linear_height_reward` 只供诊断。旧别名 `piecewise_distance_reward`、`height_reward` 暂为兼容保留，调用时发出 `DeprecationWarning`，且不再静默指向 debug 函数。

## 2v2 dense reward

每架红机对所有存活蓝机计算态势分量并取最佳配对。原始 dense reward 经公开的有效/正贡献/近零分支分配；终局奖励另行加入。`MultiAgentRewardBreakdown` 分开记录 dense、event、terminal、profile、team base、allocation factor、health、contribution 与 survival component。

## 多机终局奖励的核验状态

项目提供 `project_balanced`，完整保留原项目比例公式，并在原文核验完成前作为安全默认。配置中的 `win_weights`、`lose_weights` 与 `draw_reward` 是项目设定，不是论文公开参数；代码要求三个权重非负且和在容差内等于 1。

当前源码还包含名为 `paper_2024_exact` 的候选转写，但截至本次审计，Zheng、Wei、Duan（2024）出版商全文受订阅限制，SciEngine PDF 又被 WAF 拦截，无法从原文逐项核对式（22）—（25）的括号、分母和符号。因此该候选实现**尚未获得原文验证，不得在论文或正式实验中声称为精确复现**。需要用户提供可读取的论文 PDF 后完成最终核对；这是有意保留的科学审计阻塞项，而不是用记忆补公式。

已从任务给定信息确认的团队总量只有：

```text
r_win_all  = r_win0  * N * (1 + (max_steps - current_step) / max_steps)
r_lose_all = r_lose0 * N * (0.8 + 0.2 * (max_steps - current_step) / max_steps)
```

其中 `N` 为红机数；`current_step` 为终局决策步；`r_win0/r_lose0` 为项目配置。失败分配的反向量定义已确认：

```text
beta_prime_i = max(beta) - beta_i + 1
B_prime_i    = B0 - B_i + 10
```

候选代码处理 `beta_sum=0`、生命值和为 0、单机/全机存活、第 1/最后一步、权重非法与平局，但这些数值保护不能替代原公式核验。正式核验完成后，应在本文件逐符号列出式（22）—（25）并用独立手算测试锁定。

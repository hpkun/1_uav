# Reward Definition

## 1v1 profiles

`zheng_series_1v1` 是正式默认模式：2023 年角度、精确距离、精确高度、速度、稠密组合和终局奖励，加上 2024 年同系列论文明确列出的事件奖励数值。`debug_linear` 只保留旧连续线性距离/高度函数供调试。

距离奖励返回 `distance_approach` 与 `distance_piecewise`，`distance` 是两者之和。高度奖励严格使用附件给出的二次系数和端点。`RewardBreakdown.reward_profile` 明确记录来源模式。

## 2v2

对每个红机，态势奖励取其与全部存活蓝机配对奖励的最大值；配对奖励为 `0.15*angle + 0.60*distance + 0.10*speed + 0.15*height`，四个分量均使用论文函数。事件奖励使用 2024 表格值。

`raw_dense = situation + event`，随后执行 Algorithm 2。正贡献、接近零和失效智能体使用公开分支；存活且小于等于 -0.01 的缺失分支按项目约定保留 raw 值。终局奖励使用剩余生命值和贡献分数，并对零生命值和零贡献做保护。

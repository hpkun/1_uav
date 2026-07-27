# Environment Learnability Checklist

训练前后均应检查：

- 局部观测和全局状态的平均/最大饱和率，并定位饱和字段；
- dense、event、terminal 奖励各分量尺度，以及 terminal reward 占总回报比例；
- 随机策略、规则策略和学习策略的撞地率、超时率与 episode length；
- 动作熵、15 类动作频率和是否发生动作塌缩；
- value explained variance、value/return 尺度及是否发散；
- PPO clip fraction、approx KL、ratio、梯度范数是否有限；
- 对 straight、pursuit 和 random 基线的胜率；
- 相同独立评估 seed 上 initial、last、best 的配对差异；
- 多训练 seed 的均值、标准差、95% 置信区间；
- 对称性报告中的阵营胜率差、撞地差、配对结果一致率；
- 是否出现 NaN/Inf、无可用动作行、reset state 被误用于 truncated bootstrap。

任何单个 seed 的胜利轨迹、短期 loss 下降或规则对手胜率都不能单独证明环境可学习或算法收敛。

## Fixed 3v3 V2 reward-ordering gate

Before treating any 300k-scale fixed homogeneous 3v3 V2 run as learnability evidence, run `scripts/diagnose_3v3_reward_ordering.py` on `head_on_mirrored_jitter_v2`.

- Red rule pursuit is used to test whether active combat is reachable through the real environment dynamics, attack, and reward interfaces.
- Red straight and random baselines test whether the current reward accidentally favors delay, straight flight, or random behavior over active engagement.
- Learned Actor checkpoints, when provided, are loaded actor-only and compared with the rule policies on the same paired seeds.
- Reward components are accumulated across the whole episode, not sampled from the final step.
- Current reward coefficients and terminal formulas are not changed by this diagnostic.
- Timeout survivor-count wins, elimination wins, and draws must remain separate in reports.
- Interrupted runs can still be diagnosed from `metrics.csv` and `evaluations.csv` with `scripts/analyze_mappo_run.py`.
- Short training probes are meaningful only after fixed-policy reward ordering is understood.

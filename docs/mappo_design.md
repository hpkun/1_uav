# MAPPO Design

本项目的 MAPPO 是同构 1v1/2v2 的前馈基线，不是 Zheng、Wei、Duan（2024）图网络的逐层复刻。它不包含 GRU、注意力、异构策略、网络迁移或零样本迁移。

## CTDE 与参数共享

- 分散执行：共享 Actor 只读取单机局部观测；1v1 为 11 维，2v2 为 28 维。
- 集中训练：共享 Critic 读取全局状态与 agent identity one-hot；1v1 输入为 10+1，2v2 为 40+2。
- agent ID 只进入 Critic。Actor 不读取 ID，因此部署和后续零样本讨论不依赖固定身份编码。
- Actor：`LayerNorm -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(15)`。
- Critic：`LayerNorm -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Linear(1)`，对各 agent 的 one-hot 输入分别求值。
- 线性层使用正交初始化；隐藏层 gain 为 `sqrt(2)`，Actor 输出层 gain 为 `0.01`。

## Rollout、GAE 与 mask

缓冲区采用 `[T, num_envs, num_agents, ...]`；观测、全局状态和 value 额外保存 `T+1`。`actor_active_mask` 仅允许存活 agent 参与策略和熵损失。`critic_mask` 在团队回合继续时保持有效，因此 Critic 能学习 agent 死亡后的后续分配奖励。

普通步的 TD 残差为 `delta = r + gamma * V(next) - V(current)`。真正 `terminated` 时 bootstrap 为 0；时间截断 `truncated` 使用自动 reset 之前 terminal state 的 value。两者均切断 GAE 时间递推，避免跨回合污染。

## PPO

Actor 使用 clipped surrogate objective、available-action mask、active mask 和 entropy regularization。Critic 使用 agent-conditioned value、clipped value target 和 Huber loss。优势只在有效 Actor 样本上归一化。ValueNormalizer 维护 returns 的运行均值/方差，并随 checkpoint 保存。梯度执行全局范数裁剪，更新和日志均检查 NaN/Inf。

该基线刻意保持编码器简单，后续算法应在相同环境、采样器、评估种子和统计协议下替换 Actor/Critic 编码器。

# Literature Basis

2023 年同系列论文提供了三自由度过载质点模型、15 动作、角度奖励、距离奖励完整分段解析式、高度奖励完整分段解析式、速度奖励、稠密组合和 1v1 终局奖励。本版本已按附件给出的端点不等式和系数直接实现，不再把距离或高度解析式标为缺失。

2024 年文献为 Zheng、Wei、Duan，*UAV swarm air combat maneuver decision-making method based on multi-agent reinforcement learning and transferring*，Science China Information Sciences 67, 180204，DOI `10.1007/s11432-023-4088-2`。它提供实体块、态势奖励组合、事件奖励表、稠密奖励分配 Algorithm 2 和多机终局奖励结构。论文正式实验规模为同构 3v2；当前 2v2 是对相同实体定义的项目适配。出版商元数据已核对，但式（22）—（25）全文当前不可访问，因此精确终局公式仍处于待 PDF 核验状态。

预测对手的四个子威胁函数仍未完整公开，因此 `PursuitOpponent` 仅是带边界安全项的透明几何规则，不是论文预测威胁规则的严格复现。

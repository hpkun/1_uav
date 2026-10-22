# 第四节外部基线可复现条件核查

核查日期：2026-10-22。本轮没有实现或运行GA-MCTS、CA MADDPG AFF、LIORL，也没有生成图7、8的本地基线成绩。

## 来源与可获得性

| Xu引用 | 原始论文及入口 | 本次可获得内容 | 结论 |
|---|---|---|---|
| [18] GA-MCTS | J. Jian等，Decision-Making Method of Multi-UAV Cooperate Air Combat Under Uncertain Environment，IEEE JMASS 5(3):138–148，2024；[DOI](https://doi.org/10.1109/JMASS.2024.3378726) | 已从Xu参考文献核对论文身份；尝试打开原始DOI，本工具未能获取正文。检索得到摘要的第三方索引，但未将其用作完整算法实现依据 | 状态具体字段、维数、机动动作集、区间不确定性定义、GA算子/参数、MCTS展开/回传/预算均未完成原文核实 |
| [20] CA MADDPG AFF | J. Wu等，A Context-Aware Feature Fusion Method for Multi-UAV Cooperative Air Combat，IEEE TITS 26(5):7197–7210，2025；[DOI](https://doi.org/10.1109/TITS.2025.3530463) | 已从Xu参考文献核对论文身份；原始DOI正文未能获取。外部索引显示CAAFF名称，但不能据此认定已掌握CA MADDPG AFF完整实现 | 原始状态/动作、上下文历史长度、特征融合网络及其与MADDPG的连接、损失与训练细节仍需正文/作者实现 |
| [21] LIORL | L. Sun等，Autonomous UAV Maneuvering Decisions by Refining Opponent Strategies，IEEE TAES 60(3):3454–3467，2024；[原始IEEE页面](https://ieeexplore.ieee.org/document/10423189/)；[作者机构记录](https://shi.buaa.edu.cn/qiuhuaxin/en/lwcg/196540/content/30946.htm) | 搜索可读的IEEE原始摘要明确描述limited-imitation offline RL从既有交战数据细化对手策略，随后训练己方策略。直接打开正文被反自动化页面阻断；未获得离线数据集 | 状态/动作精确编码、离线损失/限制条件、数据覆盖与划分、后续RL配置未能完整核实；缺少原始数据尤其关键 |

Xu第IV.A明确将GA-MCTS作为在线规划方法，另外两个作为可训练基线；还明确说明其LIORL比较实现未使用不可获得的原始离线预训练数据。若以后实现，必须分别标明“原始方法”与“Xu在当前环境中的适配版本”，不能混为一谈。

## 可用实现检索

使用GitHub连接器检索仓库：`"GA-MCTS" UAV`、`"CAAFF" air combat`、`"LIORL" UAV` 均返回0个匹配仓库；宽泛LIORL搜索只出现同名个人仓库，未发现可验证的论文作者实现。也按完整论文标题及算法名检索Web。这个结果表示**本次没有找到可核验实现**，不表示证明公开实现绝不存在。

没有把普通MADDPG当作CA MADDPG AFF，没有用简单GA+任意启发式搜索冒充GA-MCTS，也没有省略离线机制后将普通RL命名为原始LIORL。

## 后续需要的材料

三篇原始全文及附录；与原文匹配的作者代码/版本；精确状态和动作空间；训练或规划预算；LIORL原始离线交战数据及预处理。获得这些材料后再决定是否具备忠实实现条件。本轮APH-MAPPO自身实验的配置、训练、评估和图表工具不依赖这些尚缺的基线。

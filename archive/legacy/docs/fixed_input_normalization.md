# 固定尺度网络输入实验（2026-10-26）

新增 `headon_xu_norm_3v3`。与 `headon_xu_3v3` 的环境配置完全相等，训练配置唯一差异为 `observation_normalization: fixed_scale_v1`。仍为 head_on、Xu paper Reward；APH、Blue、武器、动力学、观测维度、网络规模及初始化、PPO/GAE保持不变。

归一化仅在 Actor.distribution 和 ProtocolCritic.forward 调用 MLP 前执行，共用 `algorithm/mappo/networks.py:normalize_inputs`：

```text
[1, 400, pi, pi, pi, 10000, 400, pi, pi, pi, 10000, 10000]
```

12维输入逐元素除以上述向量；60维重复五次。无裁剪、运行统计、可学习参数或新增网络层。死亡/补零槽缩放后仍全零。环境输出和Rollout存储保持原始物理量；采样、旧logp重算、PPO前向、当前/后继Critic价值、独立评估和模型加载统一经过模型入口，避免双重缩放。高度/距离尺度是工程实验参数，不是Xu公开方法。

旧配置默认 `none`，直接返回原tensor，不做算术。模型state_dict键和权重结构不变。新字段进入训练配置、实验manifest、config.json和checkpoint；启动日志显示input_norm。现有严格配置恢复检查拒绝none/fixed_scale_v1双向混用，旧checkpoint缺少字段时解释为none。固定向量由带版本的fixed_scale_v1标识，具体数值同时保存在训练假设中；后续如改变向量应增加模式版本。

实验沿用headon_experiment分类，paper_experiment=false，不进入Xu原论文37项计划。默认新根目录为outputs/headon_norm_2500或outputs/headon_norm_5000，另按实验/seed分目录；已有数据覆盖保护不变。

## 实测验收

WSL Ubuntu，conda uav，CUDA。重点新增3项测试通过；相关网络/PPO/诊断/配置回归45项通过，共48项。未运行全量历史测试。修改模块和测试Python编译及git diff --check通过。

种子1000，head_on同一真实初始状态；两模型未训练，Actor/Critic初始权重逐值相等。Actor使用三个存活Red观测，Critic使用对应一条60维集中输入；饱和定义abs(Tanh输出)≥0.99：

| 第一层 | 原始输入 | fixed_scale_v1 |
| --- | --- | --- |
| Actor | 100.00000% | 0% |
| Critic | 99.21875% | 0% |

此为未训练网络的固定状态功能对照，不是历史2500回合模型的重新测量。既有PPO前向钩子仍记录实际归一化后的Tanh激活；本次短程更新中Actor/Critic两层饱和率也均为0，所有激活有限。

归一化模型对距离+1000米的确定性动作变化为[-7.30157e-6,-3.87430e-6,-2.26498e-6]；对六个角度字段分别+0.2弧度的变化为[-1.82688e-5,+2.45571e-5,+1.60933e-6]。均为非零响应。150组随机Beta动作和3组确定性动作有限，映射后的mu/zeta/normal_load全部位于原范围。未调整输出权重或Beta初始化来放大响应。

唯一CUDA训练冒烟显式缩小到max_steps=2、horizon=2、1 PPO epoch、2完成回合，使用真实10个spawn worker；共12联合步、一次PPO更新，再独立评估2局，numerical_failure=0。正常训练目录和正式配置未改。真实回合边界半Rollout checkpoint恢复后继续更新，Actor/Critic与未中断结果逐值相等；final加载后的确定性动作相等，跨模式恢复被拒绝。此规模仅用于功能验收，不评价胜率提升。

初次测试因pytest验收父目录不存在而未能创建临时目录；补建目录后全部重点测试通过，没有修改算法来规避失败。

实测JSON与短程模型位于：

`outputs/fixed_norm_acceptance_20261026/pytest/test_cuda_input_response_diagn0/acceptance.json`

## 用户自行运行正式2500回合实验

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/1_uav
python -m algorithm.experiments train --experiment headon_xu_norm_3v3 --seed 1 --episodes 2500 --root outputs/headon_norm_2500
```

正式协议仍为10环境、horizon=1500，其余MAPPO参数不变；结束自动独立评估200局。建议对照headon_xu_3v3，观察Tanh饱和率、PPO梯度、独立胜率及真实击落/存活等指标，不能用本次功能冒烟推断训练效果。本轮未启动正式2500或5000回合训练。

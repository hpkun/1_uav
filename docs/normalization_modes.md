# Normalization Modes

所有归一化先在未裁剪值上计算 `saturated_mask = abs(value) > 1`。`paper_linear` 默认不裁剪但仍报告越界；`symmetric_training` 映射并裁剪到 `[-1,1]`。死亡补零槽位不计饱和。2v2 局部观测按 agent 返回 count/ratio/mask/names，全局状态返回对应标量诊断；MAPPO 每个 rollout 记录平均与最大饱和率。

`paper_linear` 的绝对 `relative_yaw` 先转为 `[0,2*pi)`，reference 为 `2*pi`；`pi` 映射为 0，`7*pi/4` 映射为 0.75。`symmetric_training` 保留有符号的相对 yaw，正负小角度映射符号相反。heading difference 与 flight-path difference 的 reference 仍为 `pi`。

`paper_linear` 用于论文变量复核：`a*value/reference-b`，默认不裁剪；相对方位角先映射到 `[0,2*pi)` 并以 `2*pi` 为参考。该形式不被宣称为神经网络训练的最优选择。

`symmetric_training` 是默认训练模式。有符号量（坐标差、速度差、俯仰/方位/航向/航迹角差、生命值差）使用 `value/reference`；非负量（高度、距离、攻击/逃逸角、生命值）使用 `2*value/reference-1`；最后裁剪到 `[-1,1]`。

两种模式共用显式 `FeatureSpec` 表，因此 Actor、Critic、2v2 局部观测和全局状态不会依赖隐含索引猜测语义。

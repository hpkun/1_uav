# Normalization Modes

`paper_linear` 用于论文变量复核：`a*value/reference-b`，默认不裁剪；相对方位角先映射到 `[0,2*pi)` 并以 `2*pi` 为参考。该形式不被宣称为神经网络训练的最优选择。

`symmetric_training` 是默认训练模式。有符号量（坐标差、速度差、俯仰/方位/航向/航迹角差、生命值差）使用 `value/reference`；非负量（高度、距离、攻击/逃逸角、生命值）使用 `2*value/reference-1`；最后裁剪到 `[-1,1]`。

两种模式共用显式 `FeatureSpec` 表，因此 Actor、Critic、2v2 局部观测和全局状态不会依赖隐含索引猜测语义。

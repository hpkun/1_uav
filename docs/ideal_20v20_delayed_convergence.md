# 20v20 延迟收束 Intruder 参考轨迹

该轨迹只由独立 20v20 公共环境生成；共享的 6v6/15v15 `sample_level2_bezier_plan` 不变。所有 Intruder 仍以同一保护区域中心为目标，`preserve_start_altitude` 时目标高度等于各自起始高度。轨迹预生成，与 Defender 状态无关。

形成中心取 20 个名义起点的均值。共享纵向单位轴从形成中心指向保护目标，横向轴是其 XY 平面正交向量。将各实际起点相对保护中心分解为 `long0` 与 `lane0`。三次 Bézier 控制点的纵向位置分别为 `2/3 × long0` 和 `1/3 × long0`；横向位置分别为 `0.90 × lane0 + noise1` 和 `0.65 × lane0 + noise2`。两个横向噪声独立采样，范围为 `±min(12 m, 0.03 × 起终点弦长)`。两个垂向噪声独立采样，范围为 `±min(6 m, 0.015 × 起终点弦长)`。控制点抽样使用独立的 seed、实体编号和重试序号，不改变原有 start jitter 的随机数消费顺序。

路径元数据标识为 `20v20_wedge_delayed_convergence_v1`，记录共同目标、控制点、初始 lane、保持系数及实际噪声。此分布是项目自己的场景构造，**不是 Zuo 论文提供的参数**。新场景分别使用新的 nearest 和 Zuo 环境协议；旧记录仍按原协议回放。

轨迹审计入口：

```bash
python -m tools.ideal_20v20_intruder_path_audit --seed 17 --out outputs/ideal_guidance/20v20_delayed_convergence_path_audit_seed17
```

审计输出 `summary.json`、`intruder_reference_xy.png` 和 `intruder_reference_3d.png`。跨度与相关性按各路径真实弧长行进时间的 0、1/3、2/3 采样，而不是按均匀 Bézier 参数采样。

新 20v20 协议的录像默认仅展示实际执行轨迹。使用 `--reference-lines on` 可叠加完整参考虚线；`--reference-lines off` 总是隐藏，旧协议的 `auto` 保持原来显示参考线的行为。HTML 与 MP4 使用相同的开关；隐藏参考线不改变轨迹数据或插值。

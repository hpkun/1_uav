# UAV Env

一个基于 15 种离散机动的轻量级三维无人机近距空战环境骨架。项目采用三自由度过载质点模型和 `src` 布局，所有内部角度使用弧度，位置、速度和时间单位分别为米、m/s 和秒。

当前阶段仅完成基础工程骨架、离散动作库、几何工具和动力学内核。攻击与概率毁伤、奖励、规则对手、观测构造以及完整的单/多智能体 Gymnasium 环境均尚未实现；相应位置以 `TODO` 或 `NotImplementedError` 明确标识。

## 安装与验证

需要 Python 3.10 或更高版本。

```bash
python -m pip install -e .
pytest
python scripts/smoke_test.py
```

还可运行：

```bash
python scripts/validate_actions.py
python scripts/validate_dynamics.py
```

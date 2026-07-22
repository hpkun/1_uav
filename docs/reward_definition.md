# Reward Definition

`RewardBreakdown` 保存四个稠密原始项以及所有直接相加的奖励项。`total` 的直接加项是 `dense + advantage + attack_area + hit + destroy + being_hit + being_destroyed + boundary + terminal`；angle、distance、height、speed 是构成 dense 的诊断值，不重复相加。

角度奖励为 `((pi-attack)/pi) * ((pi-escape)/pi)`。距离项使用实验约定中的连续分段函数；高度项使用四个高度断点形成连续梯形；速度项严格使用附件给出的速度比区间。稠密奖励为：

```text
(0.15*angle + 0.60*distance + 0.15*height + 0.10*speed - 1) * 0.05
```

事件值为：己方攻击区 +0.3、敌方攻击区 -0.3、命中 +0.8、击毁 +1.5、被命中 -0.9、被击毁 -1.6、己方边界违规 -0.5。己方优势区使用给定的距离/角度公式，敌方优势区为 -1.0。

终局胜利奖励为 `5 + 3*(max_steps-current_step)/max_steps + 6*remaining_health/initial_health`；失败为 -15。默认 `draw_as_loss: true`。

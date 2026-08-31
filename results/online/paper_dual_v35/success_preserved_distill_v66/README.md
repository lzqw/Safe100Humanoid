# v66 成功状态保护蒸馏

v66 在 v65 的 deterministic-mean CBF 介入局部蒸馏上，增加完整成功 episode
的 local reference KL（`beta=4`），尝试在学习安全 correction 时显式保护已经会
上楼的状态。actor 仍完全禁用 PPO/entropy 梯度，只保留介入状态的 25% residual
teacher、全局 moving KL（`beta=0.5`）和成功轨迹 local KL。

训练从同一个 v60 checkpoint 出发，使用 F2 单调 18.4 cm 台阶、斜坡 x-z CBF、
256 环境、4×1024 steps、每轮刷新 DR25、actor LR `1e-7`。4080 上总训练时间为
**119.21 秒**。

## 训练结果

第 N 轮 rollout 在更新前执行，因此严格评估第 N−1 轮 checkpoint：

| Rollout 轮次 | 实际评估 checkpoint | Filtered 成功率 | 安全目标距离（更新前→后） |
|---:|---:|---:|---:|
| 1 | base | **357/547 (65.27%)** | 0.165287→0.165147 |
| 2 | round 1 | 351/558 (62.90%) | 0.169045→0.168887 |
| 3 | round 2 | 346/551 (62.79%) | 0.165986→0.165798 |
| 4 | round 3 | 353/576 (61.28%) | 0.167236→0.167046 |

实现按预期工作：每轮 actor PPO transition 数均为 0；成功 episode transition
占 56.80%–60.05%，local KL loss 非零；安全目标距离也在每次更新后略微下降。
但是三个可对齐评估的 post-update checkpoint 全部低于 base，而且随更新总体退化。

因此 v66 在训练门槛直接拒绝，不运行额外 deterministic filter-off gate，也不选择
或上传 checkpoint。最后生成的 `round_04.pt` 没有下一轮 rollout 与其对齐；为遵守
“不要做很多校验”的要求，没有为这个已呈退化趋势的更新追加评估。其 checkpoint
与 actor 哈希仍完整记录在 `training_summary.json` 中。

## 文件与溯源

- source commit：`f1e6b33a5118815d1896b18a452c4145393d236f`
- base checkpoint SHA-256：
  `f00e3a56276f629504234a20b40c124ee43a2f4d145cb143b3b2899acc024b27`
- final（未入选）checkpoint SHA-256：
  `229d96816028b7d9c121c90a23b6f0480c6742dd64f1029241a4eb38872a9f11`
- `training/training_summary.json`：完整配置、最终哈希和训练摘要。
- `training/round_metrics.{json,csv}`：四轮完整指标。
- `decision_summary.json`：机器可读门槛结论。

结论是：成功状态 local KL 足以限制数值漂移，却不能把 filtered CBF correction
转化成更高的任务成功率。当前最佳模型保持不变，v66 仅作为负结果与算法审计保留。

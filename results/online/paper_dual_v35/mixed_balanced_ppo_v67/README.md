# v67 混合过滤分组优势 PPO

v67 针对 fully-filtered PPO 的训练/部署分布差异，把 128 个并行环境固定分成
64 个 filter-on 和 64 个 filter-off 环境，并在每轮交换两组。critic 仍使用全部
transition；actor 的 PPO advantage 则分别在 on/off 组内归一化为零均值、单位
方差，使过滤轨迹的回报尺度不会淹没真实部署轨迹。

训练从当前 405-D F2 filter-off 最佳 base 出发，使用 `raw_moderate` 双奖励、
斜坡 x-z CBF、4×1024 steps、actor LR `1e-6`、moving KL `0.5`。4080 总训练
时间为 **142.68 秒**。

## 训练结果

第 N 轮 rollout 在更新前执行，严格评估第 N−1 轮 checkpoint：

| Rollout | 实际 checkpoint | Filter off | Filter on | 总成功率 |
|---:|---:|---:|---:|---:|
| 1 | base | 92/132 (69.70%) | 91/130 (70.00%) | 69.85% |
| 2 | round 1 | 85/133 (63.91%) | 89/136 (65.44%) | 64.68% |
| 3 | round 2 | **91/126 (72.22%)** | 102/142 (71.83%) | **72.01%** |
| 4 | round 3 | 78/128 (60.94%) | **102/131 (77.86%)** | 69.50% |

实现审计通过：每轮 on/off 各有 65,536 个 transition，归一化后两组 advantage
均值约为 0、标准差约为 1；执行动作路由误差为 0。但四轮 actor 更新的 gradient
clipped fraction 都是 1.0。round 2 一度把 off 提高 2.53 pp，随后 round 3
出现明显分裂：on 提高而 off 降低。这说明平衡标量 advantage 的尺度还不足以
处理 filtered/unfiltered 策略梯度方向冲突。

最佳对齐 checkpoint 仍只有 72.22% filter-off，未达到预设 75% 训练门槛，
因此没有运行独立 deterministic filter-off gate，也没有上传 checkpoint。最后的
`round_04.pt` 没有下一轮对齐 rollout，同样不作为候选。

## 文件与溯源

- source commit：`707ffc92cfb73616b2abfb61e5611478f36a6f9e`
- base checkpoint SHA-256：
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`
- best aligned（已拒绝）round-2 checkpoint SHA-256：
  `5352dea634bed77c01c673a410c3da98776d1860dcb5a7e2d8df028eee559b07`
- final（未对齐）checkpoint SHA-256：
  `b18f3c4f44f1746ac1ca5eb4f482f9f03dcf17965f92462d1116f078a125c85b`
- `training/training_summary.json`：完整配置、哈希和训练摘要。
- `training/round_metrics.{json,csv}`：四轮完整指标。
- `decision_summary.json`：机器可读门槛结论。

下一步不再让 filtered transition 的 PPO 回报直接更新 nominal actor：filter-off
组负责部署 PPO，filter-on 组只提供 deterministic-mean CBF teacher，critic 继续
共享全部数据。

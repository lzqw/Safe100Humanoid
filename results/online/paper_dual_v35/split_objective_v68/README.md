# v68 混合过滤分流 actor 目标

v68 延续 v67 的 64/64 filter-on/off 混合 rollout，但不再把过滤轨迹的回报归因
给 nominal action。filter-off transition 只负责 PPO actor 梯度；filter-on 环境中
发生 CBF 介入的状态只提供 deterministic-mean CBF teacher；critic 与 moving KL
仍使用全部 transition。两组 advantage 继续分别归一化。

训练从当前 405-D F2 filter-off 最佳 base 出发，使用 `raw_moderate` 双奖励、
斜坡 x-z CBF、4×1024 steps、actor LR `2.5e-7`、moving KL `0.5`，未开启 DR。
RTX 4080 SUPER 总训练时间为 **132.06 秒**。

## 训练结果

第 N 轮 rollout 在更新前执行，严格评估第 N−1 轮 checkpoint：

| Rollout | 实际 checkpoint | Filter off | Filter on | 总成功率 |
|---:|---:|---:|---:|---:|
| 1 | base | 73/131 (55.73%) | 104/140 (74.29%) | 65.31% |
| 2 | round 1 | 84/131 (64.12%) | 91/139 (65.47%) | 64.81% |
| 3 | round 2 | **92/132 (69.70%)** | 96/138 (69.57%) | **69.63%** |
| 4 | round 3 | 82/130 (63.08%) | **102/136 (75.00%)** | 69.17% |

实现路由与配置审计符合预期：每轮 50% transition 进入 PPO actor 目标；teacher
只使用 filter-on 环境中的介入状态，占全部 transition 的 4.82%–5.02%；critic
使用全部数据。teacher 目标距离每轮都有小幅下降，moving forward KL 保持在
`1.31e-4`–`2.44e-4`。

但四轮 actor 更新的 gradient-clipped fraction 都是 1.0，pre-clip 最大梯度为
5.71–7.32。最佳对齐 checkpoint 的 filter-off 只有 69.70%，低于预设 75%
训练门槛，也低于 v67 的 72.22%；下一轮再次出现 on/off 分裂。因此没有运行独立
deterministic filter-off gate，也没有上传 checkpoint。最后的 `round_04.pt` 没有
下一轮对齐 rollout，同样不作为候选。

## 文件与溯源

- source commit：`4c9b6fdb0785d5b8ef83bf9fa9507255cc2b23e9`
- base checkpoint SHA-256：
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`
- best aligned（已拒绝）round-2 checkpoint SHA-256：
  `db0de699163e87f209f73063a2c6c803df2afff50e9e2c32e8f8aaf4bb092622`
- final（未对齐）checkpoint SHA-256：
  `56f7a0709bccdb66d6b5ff9d63e9a8e3110bdd98c47577ab86716f616d8b0d62`
- 4080 原始输出目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/split_objective_v68_4c9b6fd_s201352615`
- `training/training_summary.json`：完整配置、哈希和训练摘要。
- `training/round_metrics.{json,csv}`：四轮完整指标。
- `decision_summary.json`：机器可读门槛结论。

下一步应处理 PPO 与 CBF teacher 的梯度冲突和持续全量裁剪，而不是增加同配置的
重复验证。

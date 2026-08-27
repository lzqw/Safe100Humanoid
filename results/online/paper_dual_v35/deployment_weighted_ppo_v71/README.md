# v71 25/75 deployment-weighted paper-style CBF-dual PPO

v71 完全移除 supervised CBF teacher，回到论文式训练期 safety filtering、bounded
CBF-dual reward 与 PPO。与 v67 的 50/50 mixed execution 不同，本次只有 25%
环境执行 filter，75% 环境执行 nominal action；on/off advantage 继续分别归一化，
critic 使用全部 transition。目标是让 actor 的大多数数据来自真实 filter-off
deployment kernel，同时保留少量安全过滤轨迹。

训练从当前 405-D F2 filter-off 最佳 base 出发，使用 `raw_moderate` 双奖励、
斜坡 x-z CBF、4×1024 steps、actor LR `1e-6`、moving KL `0.5`，未开启 DR。
RTX 4080 SUPER 总训练时间为 **135.31 秒**。

## 训练结果

第 N 轮 rollout 在更新前执行，严格评估第 N−1 轮 checkpoint：

| Rollout | 实际 checkpoint | Filter off | Filter on | 总成功率 |
|---:|---:|---:|---:|---:|
| 1 | base | **130/196 (66.33%)** | 50/67 (74.63%) | **68.44%** |
| 2 | round 1 | 129/200 (64.50%) | 41/64 (64.06%) | 64.39% |
| 3 | round 2 | 122/194 (62.89%) | 49/66 (74.24%) | 65.77% |
| 4 | round 3 | 125/200 (62.50%) | **52/66 (78.79%)** | 66.54% |

所有更新后的 filter-off 均低于本次 base 66.33%，最佳更新后 checkpoint 只有
64.50%，没有达到 75% 门槛。round 3 更出现 on 78.79%、off 62.50% 的显著
分裂；把 filtered 环境降到 25% 并未阻止策略继续依赖安全过滤。因此没有运行独立
deterministic filter-off gate，也没有上传 checkpoint。

执行审计符合配置：每轮 32/128 环境 filter-on，实际执行比例精确为 0.25，CBF
介入占全部 transition 的 2.32%–2.57%，teacher loss 和 teacher transition 均为
0。四轮 actor gradient-clipped fraction 仍全部为 1.0，pre-clip 最大梯度达到
11.16–14.40，说明 8 个 Adam minibatch step 的噪声/幅度仍是更直接的优化瓶颈。

## 文件与溯源

- source commit：`03cf19c6e7e625c5bd788892e7c07e0291a14899`
- base checkpoint SHA-256：
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`
- best updated aligned（已拒绝）round-1 checkpoint SHA-256：
  `ef824cd5b3a06fd1cf9a9b3eb61447dc4c95073e1aa892ffaee922c98d887767`
- final（未对齐）checkpoint SHA-256：
  `7c7cfa053683c86c0022f4f1280759a5b1f88125d6506298ff22aa0f41c9e083`
- 4080 原始输出目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/deployment_weighted_ppo_v71_03cf19c_s201352618`
- `training/training_summary.json`：完整配置、哈希和训练摘要。
- `training/round_metrics.{json,csv}`：四轮完整指标。
- `decision_summary.json`：机器可读门槛结论。

下一步不再调整 filter/teacher 比例，而是把每轮 8 个 Adam actor step 改为一次
保留完整 batch 方向的保守更新，减少逐 minibatch 符号缩放和持续全局裁剪。

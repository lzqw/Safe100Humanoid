# v70 投影后 CBF teacher 梯度范数平衡

v70 在 v69 的任务优先梯度投影之后，把保留下来的 filter-on CBF teacher 梯度
缩放到 filter-off PPO+KL deployment 梯度范数的 0.5，并把放大倍数硬限制为 4。
这样既不重新引入与 deployment 相反的分量，也让 teacher 在全局裁剪前获得稳定
份额。critic 仍使用全部 transition。

训练从当前 405-D F2 filter-off 最佳 base 出发，使用 `raw_moderate` 双奖励、
斜坡 x-z CBF、64/64 mixed execution、4×1024 steps、actor LR `2.5e-7`、
moving KL `0.5`，未开启 DR。RTX 4080 SUPER 总训练时间为 **138.08 秒**。

## 训练结果

第 N 轮 rollout 在更新前执行，严格评估第 N−1 轮 checkpoint：

| Rollout | 实际 checkpoint | Filter off | Filter on | 总成功率 |
|---:|---:|---:|---:|---:|
| 1 | base | **94/131 (71.76%)** | 93/135 (68.89%) | **70.30%** |
| 2 | round 1 | 84/131 (64.12%) | **98/135 (72.59%)** | 68.42% |
| 3 | round 2 | 84/129 (65.12%) | 91/143 (63.64%) | 64.34% |
| 4 | round 3 | 89/129 (68.99%) | 88/134 (65.67%) | 67.30% |

所有更新后的对齐 checkpoint 都低于本次 base filter-off 71.76%；最佳更新后结果
仅为 round 3 的 68.99%，并且没有任何一轮达到 75% 门槛。因此没有运行独立
deterministic filter-off gate，也没有上传 checkpoint。最后的 `round_04.pt` 没有
下一轮对齐 rollout，同样不作为候选。

## 优化诊断

| Round | 平均放大 | 触发 4× 上限 | 实际 teacher/deployment 比 | Teacher target distance |
|---:|---:|---:|---:|---:|
| 1 | 3.30× | 25.0% | 0.497 | 0.14540 → 0.14421 |
| 2 | 2.71× | 0% | 0.500 | 0.14285 → 0.14199 |
| 3 | 3.11× | 12.5% | 0.488 | 0.16360 → 0.16238 |
| 4 | 2.57× | 0% | 0.500 | 0.14544 → 0.14426 |

范数控制按设计执行，teacher target distance 的下降也比 v69 更大，但部署成功率
反而下降。说明 teacher 弱并不是主要原因；即使先做欧氏空间冲突投影，放大该局部
CBF action target 仍会破坏无过滤任务行为。四轮合并梯度依然 100% 触发全局裁剪。

## 文件与溯源

- source commit：`03cf19c6e7e625c5bd788892e7c07e0291a14899`
- base checkpoint SHA-256：
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`
- best updated aligned（已拒绝）round-3 checkpoint SHA-256：
  `e9730933da52f2e11b42f323c4c90a7d5ed8a39fdad430d123edd95341f5265b`
- final（未对齐）checkpoint SHA-256：
  `5c4ae78817ee9b6f69af6d2451ae588ae4a42792b87d69026821ff14f78190f0`
- 4080 原始输出目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/norm_balanced_surgery_v70_03cf19c_s201352617`
- `training/training_summary.json`：完整配置、哈希和训练摘要。
- `training/round_metrics.{json,csv}`：逐轮、逐 minibatch 完整指标。
- `decision_summary.json`：机器可读门槛结论。

下一步停止放大 supervised CBF teacher，回到论文式 CBF-dual PPO，并降低 filtered
环境占比，使大多数 actor 数据来自真实 filter-off deployment kernel。

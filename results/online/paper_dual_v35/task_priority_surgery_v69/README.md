# v69 任务优先 CBF teacher 梯度投影

v69 在 v68 的 mixed filter actor 目标分流上加入全参数梯度诊断与任务优先投影。
filter-off PPO 加全局 round-reference KL 构成受保护的 deployment 梯度；filter-on
介入状态构成 CBF teacher 梯度。两者点积为负时，只移除 teacher 中与 deployment
相反的分量；方向一致时保持 teacher 不变。critic 继续使用全部 transition。

训练从当前 405-D F2 filter-off 最佳 base 出发，使用 `raw_moderate` 双奖励、
斜坡 x-z CBF、64/64 mixed execution、4×1024 steps、actor LR `2.5e-7`、
moving KL `0.5`，未开启 DR。RTX 4080 SUPER 总训练时间为 **131.98 秒**。

## 训练结果

第 N 轮 rollout 在更新前执行，严格评估第 N−1 轮 checkpoint：

| Rollout | 实际 checkpoint | Filter off | Filter on | 总成功率 |
|---:|---:|---:|---:|---:|
| 1 | base | 90/135 (66.67%) | 99/135 (73.33%) | 70.00% |
| 2 | round 1 | 82/126 (65.08%) | 97/134 (72.39%) | 68.85% |
| 3 | round 2 | 82/129 (63.57%) | 90/141 (63.83%) | 63.70% |
| 4 | round 3 | **92/127 (72.44%)** | 97/134 (72.39%) | **72.41%** |

最佳对齐 checkpoint 的 filter-off 为 72.44%，低于预设 75% 训练门槛；它仅比
v67 的 72.22% 高 0.22 pp，而且来自不同 stochastic rollout seed，不能视为可靠
提升。因此没有运行独立 deterministic filter-off gate，也没有上传 checkpoint。
最后的 `round_04.pt` 没有下一轮对齐 rollout，同样不作为候选。

## 梯度诊断

| Round | 冲突 minibatch | 平均 cosine | Teacher 梯度保留 | Deployment / teacher norm |
|---:|---:|---:|---:|---:|
| 1 | 62.5% | -0.0960 | 98.33% | 5.94 / 1.10 |
| 2 | 37.5% | +0.0442 | 99.99% | 6.02 / 1.10 |
| 3 | 50.0% | -0.0597 | 99.20% | 5.37 / 1.08 |
| 4 | 25.0% | +0.0043 | 99.27% | 5.53 / 1.11 |

投影实现按预期工作：冲突 minibatch 的投影后点积约为 0，执行动作路由误差也为
0。但 teacher 平均保留 98.3%–100%，说明相反方向分量不是主要瓶颈。更明确的
瓶颈是 deployment 梯度约为 teacher 的 5 倍，合并梯度四轮仍 100% 触发全局裁剪，
使 teacher 在实际 actor step 中持续偏弱。

## 文件与溯源

- source commit：`2c60a251d2ce462b39c176129156a4b10fe4e83e`
- base checkpoint SHA-256：
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`
- best aligned（已拒绝）round-3 checkpoint SHA-256：
  `edff41ebb9f5a978dc0411176d6c6ccd8c12d408629b86ced59d221b1d9471fc`
- final（未对齐）checkpoint SHA-256：
  `e0ce364e74360f22a22bed05f66dd71d67a0f05d81b5f3720222df7a046d2280`
- 4080 原始输出目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/task_priority_surgery_v69_2c60a25_s201352616`
- `training/training_summary.json`：完整配置、哈希和训练摘要。
- `training/round_metrics.{json,csv}`：逐轮、逐 minibatch 完整指标。
- `decision_summary.json`：机器可读门槛结论。

下一步是在冲突投影之后按 deployment 梯度范数对 teacher 做有上限的比例平衡，
使安全监督在被全局裁剪前获得稳定份额，而不是继续调标量学习率或重复评估。

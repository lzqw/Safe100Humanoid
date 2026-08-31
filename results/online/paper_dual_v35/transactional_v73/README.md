# v73 自适应 full-batch SGD 与 rollout 事务回滚

v73 从 v72 最佳 aligned round-2 checkpoint 继续训练。每个 full-batch SGD
proposal 在下一轮 rollout 中接受或拒绝；filter-off 下降时同时恢复 actor、critic、
actor/critic optimizer，并把 actor LR 乘 0.5。KL 几何还会把下一步限制到 moving
forward KL `1e-5` 附近。训练保持论文式 25% filter-on / 75% filter-off CBF-dual
PPO 数据流，128 environments、4×1024 steps、无 teacher、无 DR。RTX 4080 SUPER
总训练时间为 **99.27 秒**。

## 对齐训练结果

第 N 轮 rollout 评估 `round_(N-1).pt`；若候选被拒绝，当前 round checkpoint
保存恢复后的 accepted state。

| Rollout | 候选状态 | Filter off | Filter on | LR | 结论 |
|---:|---|---:|---:|---:|---|
| 1 | v72 round-2 base | 125/194 (64.43%) | 47/69 (68.12%) | `5e-5` | 建立 anchor |
| 2 | proposal 1 | 120/194 (61.86%) | 49/68 (72.06%) | `5e-5` | rejected，完整回滚 |
| 3 | 同一 accepted base 重试 | 133/195 (68.21%) | 43/68 (63.24%) | `2.5e-5` | 产生 proposal 2 |
| 4 | proposal 2 | **137/197 (69.54%)** | **48/63 (76.19%)** | `2.5e-5` | accepted / selected |

回滚机制按设计阻止了 61.86% 候选继续训练。减小步长后，proposal 2 相对紧邻的
同一 base 重试提高 **1.34 pp**；若按预先登记的 round-1 anchor 则提高 5.11 pp。
但同一个 base actor 在 round 1 和 round 3 分别得到 64.43% 与 68.21%，说明非配对
stochastic rollout 本身约有数个百分点波动，当前 raw-rate acceptance 仍只是训练
启发式，最终模型必须通过独立 deterministic gate。

selected filter-off 为 **69.54%**，低于 75% 训练门槛，也低于 v72 的单次最高
71.58%。因此没有追加独立评估，当前正式最佳模型不变。四轮 moving forward KL
分别为 `9.17e-6`、`9.29e-6`、`1.95e-6`、`2.51e-6`，表明小步长目标准确执行。

## 文件与溯源

- source commit：`241bfb1abd0f6816a500f9fefb3194b2794cce69`
- base checkpoint SHA-256：
  `3285223174b01c97009db54361042c4c3d2d87054ca2156c84769f6d13ceccbc`
- selected aligned `round_03.pt` SHA-256（未上传模型）：
  `c8d09065c3c3a5409c446f5eb895cb01c550bb5e919534cb0152e6471e269ab9`
- selected actor SHA-256：
  `a1d6ab2c124f92b30fb4e84a2403dddd7ad20affd576cc880bd5277d1a986c82`
- final unaligned checkpoint SHA-256：
  `2c5a4d0c60850c8f813bb3821b1adfd4bed79635b9ce9e23360239d79875e49a`
- 4080 原始输出目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/transactional_v73_241bfb1_s201352620`
- `training/training_summary.json`：完整配置、选择、回滚和 checkpoint 哈希。
- `training/round_metrics.{json,csv}`：四轮完整指标。
- `decision_summary.json`：机器可读训练门槛结论。

下一步从 selected checkpoint 继续更小的 `1.25e-5` full-batch step，并把结果与
同一 seed 的起点 rollout 分开解释，避免把 rollout 方差当成算法提升。

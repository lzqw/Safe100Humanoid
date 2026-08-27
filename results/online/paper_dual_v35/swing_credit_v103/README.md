# v103 bounded full-swing safety credit

结论：v103 在论文 Eq. (27) 双奖励之外增加一个跨完整 swing 的有界 intervention
credit。实现和尺度均按设计工作，但最佳对齐 filter-off 只有
**131/193 = 67.88%**，相对本次 v79 base 提高 1.55 pp，未达到 75%，因此拒绝。

## 动机与实现

历史 v72/v79 的 `pre_intervention_weight` 实际固定为 0；已有
`pre_intervention_cost_mean` 只是诊断，没有进入训练 reward。v103 是一个显式独立算法，
在 Eq. (27)+GAE 之外加入早期时序 credit：

- 向前观察 50 control steps，即 1.0 秒，覆盖完整 swing/预摆动阶段。
- 对连续 intervention 使用 `max` 而不是逐帧求和，credit 严格不超过 1。
- 时间衰减 `0.95`，reward 权重 `0.01`。
- 实测附加惩罚均值约 `0.00121/step`，小于 Eq. (27) 双奖励约
  `0.00281–0.00287/step`，没有压过 nominal reward。

commit `6dd6923` 提供有界 credit 数学与配置；commit `6377037` 用独立
`PaperSwingCreditV103PPO` 显式接入 full-batch PPO，所有历史路径仍保持 weight 0。
正式环境的两项定向测试通过。第一次启动被历史构造器在 rollout 前拒绝，optimizer step
为 0；随后通过独立类重新启动正式训练。

## 训练结果

训练从 v79 round 3 开始，使用 current CBF、25% filter-on / 75% filter-off、
Eq. (27) unit-balanced reward、128 environments、4×1024 steps、full-batch SGD 和
actor LR `1e-4`。

| 对齐 rollout | 实际 checkpoint | Filter off | Filter on | Swing penalty/step |
|---:|---:|---:|---:|---:|
| 1 | v79 base | 130/196 (66.33%) | 50/69 (72.46%) | 0.001207 |
| 2 | round 1 | 127/192 (66.15%) | **48/65 (73.85%)** | 0.001203 |
| 3 | round 2 | **131/193 (67.88%)** | 44/67 (65.67%) | 0.001215 |
| 4 | round 3 | 129/193 (66.84%) | 45/68 (66.18%) | 0.001212 |

训练耗时 100.98 秒。四轮 credit max 均为 1.0，action routing error 均为 0。
最佳 round-2 checkpoint 低于训练 gate，也低于 v79 的历史正式结果 72.02%；因此没有
运行额外独立 gate，也没有把 rejected checkpoint 二进制提交到 GitHub。

## 结论

把安全介入信号扩展到完整 swing 能稳定地产生有界、同量级梯度，但仍未解决 outcome
gradient 的跨 rollout 不一致。下一步需要让 actor 直接获得论文任务所依赖的 next-riser
几何/clearance reference，并进行更长的 filtered-execution 训练，而不是继续调整 credit
窗口或权重。

完整训练数据位于 [`training/`](training/)，机器可读决策和 checkpoint 哈希分别位于
[`decision_summary.json`](decision_summary.json) 与
[`checkpoint_index.json`](checkpoint_index.json)。

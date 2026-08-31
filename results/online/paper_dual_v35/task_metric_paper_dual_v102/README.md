# v102 task-metric CBF + paper-dual PPO

结论：v102 把 v100 的 task-metric CBF 真正接入论文 Algorithm 1 风格的“过滤执行、
task/CBF 双奖励、PPO 更新”闭环。最佳对齐 filter-off checkpoint 相对本次 v79 base
提高 **1.50 pp**，但只有 **133/197 = 67.51%**，未达到 75%，因此拒绝且不追加 gate。

## 实现修复

第一次启动在 rollout 审计时发现：历史 `TaskMetricVelocityCbfAction` 只读取全局
`cfg.enabled`，没有读取每环境 `runtime_filter_mask`，导致 25/75 mixed execution 实际上
对全部环境执行过滤。该次在 optimizer step 前终止，没有候选更新。

commit `a0611231ae9fd9be4b748785de18611d2bd4109a` 修复为逐环境路由，并同步修正
executed action、margin、correction 与 intervention telemetry。针对性测试 9/9 通过；
正式 v102 四轮的 `executed_action_routing_max_abs_error` 均为 0。

## 论文式训练设置

- 输入 checkpoint：v79 round 3，SHA-256
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`。
- F2 18.4 cm，128 environments，4×1024 steps。
- 25% task-metric CBF filter-on、75% filter-off，逐轮旋转环境分组。
- Eq. (27) foot-task proximity、0.1 margin 单位桥接、next-riser clearance reference。
- 分组 advantage normalization、full-batch SGD、actor LR `5e-5`、moving KL `0.5`。

| 对齐 rollout | 实际 checkpoint | Filter off | Filter on | Moving KL |
|---:|---:|---:|---:|---:|
| 1 | v79 base | 134/203 (66.01%) | 41/67 (61.19%) | 8.34e-6 |
| 2 | round 1 | 132/199 (66.33%) | 45/68 (66.18%) | 1.11e-5 |
| 3 | round 2 | 127/195 (65.13%) | 42/67 (62.69%) | 9.41e-6 |
| 4 | round 3 | **133/197 (67.51%)** | **46/69 (66.67%)** | 1.29e-5 |

训练总耗时 94.20 秒。round 3 是本次最佳对齐候选，但它既低于 75% gate，也低于
v79 的历史正式结果 139/193（72.02%）。round 4 是未对齐的最后一次 update，不参与选择。

## 决策

task-metric teacher 的 PPO 闭环能够带来小幅运行内恢复，但没有复现 v79 的大幅提升，
也没有提高最终安全 teacher ceiling。v102 拒绝，不上传模型二进制；全局选择仍为 v79。
这也说明下一步不应继续做静态 metric 或 residual 幅度搜索，而应改善跨整段 swing 的
安全目标与 episode-level task credit。

完整训练汇总和逐轮指标位于 [`training/`](training/)，机器可读决定位于
[`decision_summary.json`](decision_summary.json)，远端 checkpoint 路径和哈希位于
[`checkpoint_index.json`](checkpoint_index.json)。

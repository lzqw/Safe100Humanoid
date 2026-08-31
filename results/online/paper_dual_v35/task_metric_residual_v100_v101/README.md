# v100–v101 task-metric CBF residual

结论：task-metric CBF 在同 seed 的 filter-on 对照中略优于当前 Euclidean CBF，且降低了
脚尖峰值接触力；但用它训练的成功轨迹 residual 在独立 filter-off gate 中只有
**40/64 = 62.50%**。v101 因此拒绝，全局选择仍为 v79。

## v100：先提高 teacher 的 task compatibility

冻结 v79，在相同 seed `201353200`、各 64 个 F2 18.4 cm 回合上比较两个 runtime
filter。`c012` 使用 stance leg 与 hip roll/yaw 的 2x 对角 task metric，并设置
`top_clearance=0.03`。

| Filter | 成功 | Mean reached riser | Toe force peak | Mean correction norm |
|---|---:|---:|---:|---:|
| Current Euclidean | 44/64 (68.75%) | 8.000 | 1561.15 | 1.2626 |
| Task metric `c012` | **46/64 (71.88%)** | **8.125** | **1225.35** | 1.4687 |

`c012` 多成功 2 个 episode，并把脚尖峰值接触力降低约 21.5%，因此被用作 v101 的
filtered-execution teacher。它本身仍低于 75% 目标。

## v101：成功 filtered episode 的 residual DAgger

v101 冻结 v79，只训练 28,364 参数 residual head。输入是 v79 的 128-D hidden state、
10-D persistent next-riser geometry 和 12-D nominal action；训练执行 task-metric CBF
安全动作，并只保留最终成功 episode 中的 `safe - nominal` 教师修正。部署评估完全关闭
runtime filter。

| 轮次 | Filter-on 成功 | 本轮 teacher transitions | Teacher distance after |
|---:|---:|---:|---:|
| 1 | 82/128 (64.06%) | 3,153 | 0.16994 |
| 2 | 84/128 (65.63%) | 2,198 | 0.11402 |
| 3 | 86/128 (67.19%) | 2,041 | 0.09468 |
| 4 | 76/128 (59.38%) | 1,678 | 0.08661 |

四轮总计保留 9,070 条 teacher transitions，训练耗时 68.65 秒。拟合误差持续下降，
但最后一轮 filter-on task success 回落，说明 correction fit 与 episode success 仍不一致。

同一 256-env screen 按每档 64 回合比较部署 residual scale：

| Scale | Filter-off screen |
|---:|---:|
| 0 | 45/64 (70.31%) |
| 0.025 | 48/64 (75.00%) |
| 0.05 | **49/64 (76.56%)** |
| 0.10 | 44/64 (68.75%) |

按预注册规则选择 `0.05x`。它在全新 seed `201353300` 的唯一 gate 中仅得到
**40/64 = 62.50%**，没有通过 75% 门槛，所以不追加 rollout，v101 正式拒绝。

## 文件与溯源

- [`v100_filter_screen/`](v100_filter_screen/)：两个 CBF 的逐回合 CSV 与汇总 JSON。
- [`v101_training/`](v101_training/)：运行配置、逐轮摘要和训练汇总。
- [`v101_calibration/`](v101_calibration/)：scale screen、独立 gate 及逐回合数据。
- [`decision_summary.json`](decision_summary.json)：机器可读的核心结果与最终决定。
- [`checkpoint_index.json`](checkpoint_index.json)：输入、候选哈希及 4080 路径。

源代码 commit 为 `586d58ccd8674deecad1f28c6cb2ff5d9abd93c0`。被拒绝的模型二进制不提交
GitHub，但仍保留在 4080 的记录路径中。

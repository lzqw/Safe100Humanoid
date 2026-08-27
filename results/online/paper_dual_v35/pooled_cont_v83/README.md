# v83 新 seed microbatch continuation

v83 从 v82 selected `round_03.pt` 出发，在新 seed `201352626` 上继续相同
unit-balanced mixed-execution 训练。由于启动时其他 GPU 任务使可用显存只有约 5 GB，actor
使用四个等权 gradient chunks；每轮仍只执行一次全局裁剪和一次 SGD step。4 rounds 总耗时
**149.15 秒**，没有 OOM。

| Rollout | Filter off | Filter on | 决策 |
|---:|---:|---:|---|
| 1（base） | 247/396（62.37%） | 89/135（65.93%） | anchor |
| 2 | 266/384（69.27%） | 101/132（76.52%） | accepted |
| 3 | 252/389（64.78%） | 91/138（65.94%） | rejected / rollback |
| 4（同 accepted actor 重试） | 254/395（64.30%） | 102/134（76.12%） | pooled evidence |

单轮峰值为 69.27%，但同一 accepted actor 的 rollout 2 和 4 合并后为
**520/779（66.75%）**。这比单轮峰值低 2.52 pp，且远低于 75% gate，说明继续重复相同
unit-balanced continuation 的收益不可靠。没有运行额外 deployment gate。

## 溯源

- training source commit：`d74dbc9`
- input checkpoint SHA-256：
  `b483e95ab345b7936d7b0f8b360e758a01ccd96d77d1ac08f3a915714001c16a`
- selected aligned checkpoint：`round_01.pt`
- selected checkpoint SHA-256：
  `09330d3a5a7ae2baa2d84437f0633f0038128674d6761d1d495d57bb593aae56`
- selected actor SHA-256：
  `219878640bc9de3f659e062dcb3c1389293603c47ef6a9d95e30f0cc0c60a99a`
- final unaligned checkpoint SHA-256：
  `47e5e45bf3bda630e349943cce8825a3ce9dc7b430f84dc0a875c838ff22ba7a`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/pooled_cont_v83_d74dbc9_s201352626`

完整 `round_metrics.json`、CSV 和 `training_summary.json` 位于 `training/`。


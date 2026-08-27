# v74 同 seed 超小步长 continuation

v74 检查 v73 最后一轮生成但尚未对齐评估的 proposal。训练重新使用 v73 的 seed
`201352620`，因此首轮与 v73 round-1 具有相同初始化随机序列；随后只做一次
`1.25e-5` full-batch SGD proposal。128 environments、2×1024 steps 在 RTX 4080
SUPER 上耗时 **46.96 秒**。

| Rollout | Filter off | Filter on | moving KL | 结论 |
|---:|---:|---:|---:|---|
| 1，v73 unaligned final | 125/195 (64.10%) | 50/71 (70.42%) | `1.92e-7` | 建立 anchor |
| 2，超小步长 proposal | 121/196 (61.73%) | 51/68 (75.00%) | `6.31e-7` | rejected / rollback |

首轮 filter-off 与 v73 同 seed 原始起点的 125/194（64.43%）基本相同，说明 v73
最后一个 unaligned update 没有产生可复现的提升。新的超小步长 proposal 又下降
2.37 pp 并被完整回滚；KL 已降到 `1e-7` 量级，继续缩小相同方向的更新没有价值。
本次未达到 75% training gate，不追加独立评估或上传 checkpoint，当前最佳不变。

## 溯源

- source commit：`241bfb1abd0f6816a500f9fefb3194b2794cce69`
- input checkpoint SHA-256：
  `2c5a4d0c60850c8f813bb3821b1adfd4bed79635b9ce9e23360239d79875e49a`
- input actor SHA-256：
  `020c7d399b10a83ae9b272d82b166cdd9502b5a9f0dab3f673631c2cadb07f5c`
- 4080 原始输出目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/matched_continuation_v74_241bfb1_s201352620`
- `training/` 保存完整 JSON/CSV 指标；`decision_summary.json` 保存机器可读结论。

下一步停止同方向缩步长，改测论文原始训练分布：所有训练环境执行 safety filter，
但 PPO 仍存 nominal policy action，并使用 task/CBF reward 更新 actor。

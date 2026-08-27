# v75 论文式 100% safety-filtered full-batch PPO

v75 恢复论文的核心训练分布：所有环境执行 CBF-filtered action，但 PPO transition
仍保存 nominal policy action，并使用 task/CBF dual reward。actor 每轮只做一次
full-batch SGD；filter-on 对齐 rollout 用于事务式接受/回滚。训练从 v72 aligned
round-2 actor 开始，128 environments、4×1024 steps、LR `5e-5`、无 teacher、
无 DR。RTX 4080 SUPER 总耗时 **89.67 秒**。

## 结果

| Rollout | 实际 actor | Filter on | LR | moving KL | 结论 |
|---:|---|---:|---:|---:|---|
| 1 | base | 194/273 (71.06%) | `5e-5` | `9.35e-6` | 建立 anchor |
| 2 | proposal 1 | 191/271 (70.48%) | `5e-5` | `1.15e-5` | rejected / rollback |
| 3 | 同一 base 重试 | 190/266 (71.43%) | `2.33e-5` | `1.98e-6` | 产生 proposal 2 |
| 4 | proposal 2 | 185/277 (66.79%) | `2.33e-5` | `1.94e-6` | rejected / rollback |

两个更新后的 actor 均低于起点并被完整回滚，最佳 accepted actor 仍是 base。因为
100% 训练过滤，本次没有 filter-off rollout；filter-on 未达到 75% training gate，
所以不追加独立 filter-off 评估，也不上传 rejected checkpoint。

本次严格审计通过：每轮 runtime filter fraction 均为 1，CBF 介入占 transition 的
约 9.7%–10.1%，约为 25% mixed execution 时的四倍；
`executed_action_routing_max_abs_error=0` 且
`policy_storage_max_abs_error=0`，证明 simulator 执行 filtered action、PPO 保存
nominal action。失败因此不是数据路由错误，而是当前 dual reward 产生的 PPO 梯度
在这个已训练 actor 附近仍未提高任务成功率。

## 文件与溯源

- source commit：`7810d2f5718088205f0f02a62f1707605039358a`
- base checkpoint SHA-256：
  `3285223174b01c97009db54361042c4c3d2d87054ca2156c84769f6d13ceccbc`
- base/selected actor SHA-256：
  `672dfdb3af5f19c870313183b461e90cdd4198bff59d41060ce1553eb24fea32`
- final rollback checkpoint SHA-256：
  `f9149bd245ba80171f5a722d33e1031e785cdc22415f1780536281af0aa17f4d`
- 4080 原始输出目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/full_filter_v75_7810d2f_s201352622`
- `training/` 保存完整配置、哈希、逐轮 JSON/CSV；`decision_summary.json` 保存门槛结论。

下一步不再调整 filter fraction，而是核对论文 Eq. (22)–(23) 与当前
`margin_weight=1`、`intervention_weight=10`、`sigma=0.5` 的 reward 数值分解，
确认任务梯度是否被 action-proximity 项主导。

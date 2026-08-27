# v65 介入局部安全蒸馏

v63/v64 表明强 Eq. (27) safety reward 通过普通 PPO advantage 传播时会损伤
18.4 cm 任务策略。v65 因此把两种 actor 梯度解耦：训练 rollout 仍执行 CBF，
critic 仍观察完整 dual reward；actor 禁用 PPO/entropy 梯度，只在 frozen
deterministic policy mean 确实触发 CBF 的状态上学习 25% residual correction，
并用全局 round-reference KL 限制漂移。

训练使用 v60 对齐 checkpoint、斜坡 x-z clearance、persistent next-riser
脚部 reference、128 环境、4×1024 steps、每轮刷新 DR25，actor LR 为 `1e-7`。

## 训练结果

| Rollout 轮次 | 实际评估 checkpoint | Filtered 成功率 | 安全目标距离（更新前→后） | KL |
|---:|---:|---:|---:|---:|
| 1 | base | 182/274 (66.42%) | 0.16863→0.16766 | 1.07e-4 |
| 2 | **round 1** | **192/270 (71.11%)** | 0.16764→0.16670 | 9.46e-5 |
| 3 | round 2 | 176/276 (63.77%) | 0.16536→0.16434 | 1.19e-4 |
| 4 | round 3 | 168/283 (59.36%) | 0.16496→0.16399 | 1.06e-4 |

每轮 actor PPO transition 数精确为 0，安全 teacher transition 约 16.5k，证明
梯度隔离按预期工作。安全目标距离每轮都下降，但连续蒸馏仍逐渐损伤任务性能；
因此按 checkpoint 对齐规则只选择 `round_01.pt`，它在下一组 DR 上比 base
rollout 高 4.69 pp。

## 单次纯无过滤 gate

只对入选 checkpoint 做一次 deterministic fixed-environment gate：

| Context | Runtime filter | Seed | Episodes | Success | Fall |
|---|---|---:|---:|---:|---:|
| F2，18.4 cm | off | 201352512 | 128 | **76/128 (59.38%)** | 52/128 (40.62%) |

该结果低于预设 75% 门槛，v65 拒绝。没有追加 filter-on、基线重测或更多 seed，
也不将训练时的 stochastic filtered 71.11% 当作部署成功证据。

## 同步修复

首个 256-env run 发现同一步可能同时触发 `reached_top` 和 `fell_over`，导致
旧 teacher outcome mask 重叠。commit `b7ecf95` 明确采用 deployment success
语义：到顶优先，只将未到顶的 fall 归为失败。随后 256-env 重跑因服务器上其他
进程占用约 10 GB 显存而 OOM；最终 128-env run 完整结束。OOM 不计作算法结果。

## 文件与溯源

- source commit：`b7ecf953f8e9b0445b0b31ef3c203c32a081b1e6`
- base checkpoint SHA-256：
  `f00e3a56276f629504234a20b40c124ee43a2f4d145cb143b3b2899acc024b27`
- selected checkpoint SHA-256：
  `4b69938604fd8235ce50401e5ed07fdcda6cd3ab5977adc41ce329249e73eb1c`
- selected actor SHA-256：
  `a25b36f03a4a3e7c2b2433908fbe3d3ac530dc360933234d7c4e344f79c14436`
- `training/round_01.pt`：入选但最终拒绝的精确 checkpoint。
- `training/training_summary.json`、`round_metrics.{json,csv}`：完整训练记录。
- `gate/filter_off_summary.json`、`filter_off_episodes.csv`：128-episode gate。
- `decision_summary.json`：机器可读结论。

下一步需要让 intervention-local 安全更新同时显式保护成功任务状态，而不是继续
累积同一种全介入蒸馏。

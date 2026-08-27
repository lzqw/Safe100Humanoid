# Paper-dual 当前结果索引（2026-08-27）

当前目标是 F2 18.4 cm、部署时 filter-off 成功率至少 75%。截至本次记录，目标尚未达到，
当前最好的训练内对齐结果仍是 v79 的 **139/193（72.02%）**。

| 版本 | 关键结论 | GitHub 目录 |
|---|---|---|
| v79 | 从 61.11% 提升到 72.02%，但低于 75% 门槛；仍是当前最佳 | [`mixed_unit_balanced_v79/`](mixed_unit_balanced_v79/) |
| v80 | pooled base 为 68.04%，proposal 为 68.32%，仅 +0.28 pp，判定统计持平 | [`mixed_cont_v80/`](mixed_cont_v80/) |
| v81 | 256/192 environments 都在首次 actor full-batch update OOM；0 次 optimizer step，无候选模型 | [`pooled_scaling_v81_oom/`](pooled_scaling_v81_oom/) |
| v82 | microbatch 修复后 256 environments 完整运行；filter-off 63.22%→69.37%，仍未过 gate | [`pooled_microbatch_v82/`](pooled_microbatch_v82/) |
| v83 | 新 seed continuation 的单轮峰值 69.27%；同 actor pooled 520/779=66.75%，未过 gate | [`pooled_cont_v83/`](pooled_cont_v83/) |
| v84 | Eq. (27) margin 0.1→0.25 后峰值仅 67.59%，第 4 轮回落并 rollback | [`mid_margin_v84/`](mid_margin_v84/) |
| v85 | Eq. (23) proximity 1→2 后单轮 69.49%；同 actor pooled 526/777=67.70% | [`proximity_v85/`](proximity_v85/) |
| v86 | 100% filtered execution 的 base pooled 789/1094=72.12%；两个 proposal 均回落 | [`full_filter_v86/`](full_filter_v86/) |
| v87 | 四个 256-env PPO delta 共识；方向 cosine 0.017，独立 off gate 173/256=67.58% | [`consensus_v87/`](consensus_v87/) |

v79 最佳 checkpoint 保留在 4080：
`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/mixed_unit_balanced_v79_d65f0b6_s201352619/round_03.pt`，
SHA-256 为 `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`。
因为它没有通过 75% gate，模型二进制未提交到 Git；配置、逐轮结果、哈希和选择依据均已提交。

v82/v83 已证明 microbatch 路径稳定：每轮多个等权 backward chunks、一次全局梯度裁剪、
一次 SGD step，256-environment 训练没有再发生 OOM。v83 在新 seed 上先升后退，rollback
后的 accepted actor 两次 rollout 合并仅 66.75%，说明同配置 continuation 已进入噪声平台。
v87 将有效 PPO batch 扩到 1,048,576 transitions。四个 delta 的平均 pairwise cosine 仅
0.017，共识范数也只有单成员平均的 51.3%；独立 filter-off gate 仍只有 67.58%。这证明
reward-only PPO 的局部梯度主要是 rollout 噪声，继续增加同类样本不会解决问题。下一步改为
成功 filtered trajectory 的 safe-action 自模仿：只克隆真正到顶 episode 中执行过的安全
动作，并保留 PPO、moving KL 和事务 rollback。


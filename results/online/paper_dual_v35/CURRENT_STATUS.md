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

v79 最佳 checkpoint 保留在 4080：
`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/mixed_unit_balanced_v79_d65f0b6_s201352619/round_03.pt`，
SHA-256 为 `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`。
因为它没有通过 75% gate，模型二进制未提交到 Git；配置、逐轮结果、哈希和选择依据均已提交。

v82/v83 已证明 microbatch 路径稳定：每轮多个等权 backward chunks、一次全局梯度裁剪、
一次 SGD step，256-environment 训练没有再发生 OOM。v83 在新 seed 上先升后退，rollback
后的 accepted actor 两次 rollout 合并仅 66.75%，说明同配置 continuation 已进入噪声平台。
v86 使用论文的 100% filtered execution，同时 PPO 存 nominal action，但两个 proposal
都不如未更新 base；base 的两次 filter-on rollout pooled 为 72.12%。下一步用四个独立
256-environment rollout 从同一个 v79 actor 产生 PPO delta，再平均 delta 后评估。这把有效
batch 扩到约一百万 transitions，更接近论文的大规模并行训练并降低单 seed 梯度噪声。


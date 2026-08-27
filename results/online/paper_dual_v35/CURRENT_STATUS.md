# Paper-dual 当前结果索引（2026-08-28）

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
| v88 | 成功轨迹 sampled-safe-action 模仿的两个 proposal 均回落；base pooled 754/1078=69.94% | [`success_imitation_v88/`](success_imitation_v88/) |
| v89 | 成功且干预状态的 deterministic safe mean 模仿仍回落；base pooled 775/1092=70.97% | [`success_intervention_v89/`](success_intervention_v89/) |
| v90 | 25% bounded residual 的本机 32-env 筛选中两个 proposal 均回落；不正式放大 | [`success_residual_v90/`](success_residual_v90/) |
| v91 | 64-env 放大后两个 residual-only proposal 均低于 pooled base 194/269=72.12%，路线拒绝 | [`success_residual_only_v91/`](success_residual_only_v91/) |
| v92 | 5-D geometry adapter 的 full-batch SGD 方向正确；untouched filter-off 47/64=73.44%，差 1 episode | [`observable_rescue_sgd_v92/`](observable_rescue_sgd_v92/) |
| v93 | 16-D 左右脚/phase 条件化 adapter 离线方向改善，但唯一 untouched filter-off 仅 45/64=70.31%，拒绝 | [`conditional_geometry_sgd_v93/`](conditional_geometry_sgd_v93/) |

v79 最佳 checkpoint 保留在 4080：
`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/mixed_unit_balanced_v79_d65f0b6_s201352619/round_03.pt`，
SHA-256 为 `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`。
因为它没有通过 75% gate，模型二进制未提交到 Git；配置、逐轮结果、哈希和选择依据均已提交。

v82/v83 已证明 microbatch 路径稳定：每轮多个等权 backward chunks、一次全局梯度裁剪、
一次 SGD step，256-environment 训练没有再发生 OOM。v83 在新 seed 上先升后退，rollback
后的 accepted actor 两次 rollout 合并仅 66.75%，说明同配置 continuation 已进入噪声平台。
v87 将有效 PPO batch 扩到 1,048,576 transitions。四个 delta 的平均 pairwise cosine 仅
0.017，共识范数也只有单成员平均的 51.3%；独立 filter-off gate 仍只有 67.58%。这证明
reward-only PPO 的局部梯度主要是 rollout 噪声，继续增加同类样本不会解决问题。v88 随后
对成功 filtered trajectory 做 sampled-safe-action 自模仿，但成功轨迹中只有约 8.5%–8.8%
的 transition 真正被 CBF 修正，其余目标主要是 stochastic exploration noise；两个 proposal
均被 rollback。下一步只训练“成功且实际 CBF 干预”的 transition，并改用同状态
deterministic safe mean 作为低噪声目标。

v89 完成了上述低噪声路由：teacher transition 只占 5.26%–5.55%，但完整 deterministic
safe-mean correction norm 高达 0.512–0.524，两个 proposal 仍回落。下一步保留该成功干预
gate，只应用论文 A2 的 25% bounded deterministic residual，避免直接克隆过大的完整修正。

v90 将目标距离按预期缩到 0.128–0.133，但小批量 actor gradient norm 仍达 7.43–9.42
且全部被裁剪，两个 proposal 均退化。这轮仅是 4080 被占用时的本机方向筛选，不是正式
gate；结论是下一步必须从 actor 更新中移除 noisy PPO/entropy 梯度，只保留成功干预
bounded residual 与 moving reference KL。

v91 完成了该隔离：四轮 actor PPO transition count 均为 0，gradient norm 降至
0.118–0.132 且不再裁剪。但 64-env 放大中 `1.25e-4` 与 `6.25e-5` proposal 分别只有
71.32% 和 71.21%，都低于 pooled base 72.12%，说明 32-env 的 +0.32 pp 是 rollout 方差。

v92 随后补充 5 个可部署 CBF geometry 输入，只从 54 个 matched-rescue episode 学习，
并用一次 full-batch SGD 避免 Adam 的逐坐标方向扭曲。离线 correction cosine 达 0.591，
KL 为 5e-5；唯一 untouched deterministic filter-off gate 为 47/64=73.44%，距离 75%
只差 1 个 episode，但仍按门槛拒绝。下一步应把几何 residual 按 swing side / barrier
phase 条件化，避免相反的局部修正被单一全局 adapter 平均掉。

v93 完成了上述条件化：将 5-D geometry 展开为 left/right × unsafe/safe 的 16-D
可部署特征，只训练 8,192 个新增首层权重。一次 full-batch SGD 将离线 correction cosine
提高到 0.641，distance 从 0.135701 降到 0.135405，且 KL 仅 3.02e-5；但唯一全新 seed
的 deterministic filter-off gate 只有 45/64=70.31%。因此该结构虽改善 teacher 对齐，
仍未改善最终成功率，候选已拒绝，未追加更多 rollout，正式最佳继续保留 v79。

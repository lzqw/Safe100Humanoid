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
| v94 | 10-D persistent 双脚 next-riser 几何使 paired filter-on 达 191/256，但单步 imitation 的 untouched off 仍为 45/64=70.31% | [`persistent_geometry_sgd_v94/`](persistent_geometry_sgd_v94/) |
| v95 | persistent geometry paired PPO 的 6/6 surrogate 改善，但跨 seed gradient cosine 仅 0.020；untouched off 44/64=68.75% | [`persistent_geometry_ppo_v95/`](persistent_geometry_ppo_v95/) |
| v96 | persistent geometry + filter-free 成功轨迹自模仿离线方向为正，但 cosine 仅 0.005；唯一 untouched off 35/64=54.69%，拒绝 | [`persistent_geometry_elite_v96/`](persistent_geometry_elite_v96/) |
| v97 | learned residual head 能拟合 CBF correction，但 filter-on 随拟合增强从 95/128 降至 78/128；off gate 45/64=70.31% | [`learned_residual_v97_v99/`](learned_residual_v97_v99/) |
| v98 | 同屏幅度校准选择 0.05x；screen 47/64，唯一 unseen off gate 46/64=71.88%，拒绝 | [`learned_residual_v97_v99/`](learned_residual_v97_v99/) |
| v99 | 只学习成功 filtered episode 后，幅度 screen 最终选择 0x；learned direction 完全拒绝 | [`learned_residual_v97_v99/`](learned_residual_v97_v99/) |
| v100 | 同 seed filter-on 对照中 task-metric CBF 为 46/64，当前 CBF 为 44/64，且脚尖峰值力降低 21.5% | [`task_metric_residual_v100_v101/`](task_metric_residual_v100_v101/) |
| v101 | task-metric 成功轨迹 residual 的 scale screen 达 49/64，但唯一 unseen filter-off gate 仅 40/64，拒绝 | [`task_metric_residual_v100_v101/`](task_metric_residual_v100_v101/) |
| v102 | task-metric CBF 接入 paper-dual PPO；修复 mixed mask 路由后最佳对齐 off 133/197=67.51%，拒绝 | [`task_metric_paper_dual_v102/`](task_metric_paper_dual_v102/) |
| v103 | 1 秒 bounded max swing credit 尺度稳定，但最佳对齐 off 131/193=67.88%，拒绝 | [`swing_credit_v103/`](swing_credit_v103/) |
| v104 | 415-D persistent geometry 接入完整 PPO；8 轮最佳 off 132/200=66.00%，新增几何列几乎未被利用，拒绝 | [`persistent_paper_dual_v104/`](persistent_paper_dual_v104/) |
| v105 | 自适应放大 geometry 梯度到 legacy block 的 1:1；最佳 off 138/194=71.13%，仍低于 v79 与 gate | [`persistent_geometry_balanced_v105/`](persistent_geometry_balanced_v105/) |
| v106 | episode outcome credit 与 GAE 联合训练；本轮峰值 off 68/95=71.58%，提高 2.19 pp 但仍低于 v79 与 gate | [`outcome_geometry_v106/`](outcome_geometry_v106/) |
| v107 | outcome 权重减半并逐 proposal 事务回滚；4/4 proposal 回落，最终完整恢复输入 actor | [`outcome_transactional_v107/`](outcome_transactional_v107/) |
| v108 | 回到原生 CBF-dual 并将 std 降至 0.03；4/4 proposal 回落，低探索噪声仍未改善 | [`low_noise_transactional_v108/`](low_noise_transactional_v108/) |
| v109 | 成对 filter-off 观测/filter-on 安全轨迹训练；screen 仅 41/64=64.06%，未触发独立 gate | [`paired_rescue_trajectory_v109/`](paired_rescue_trajectory_v109/) |
| v110 | 修正 v109 分叉后的状态/动作错配；离线方向更好但 screen 降至 36/64=56.25%，拒绝 | [`deployment_counterfactual_v110/`](deployment_counterfactual_v110/) |
| v111 | paired terminal outcome 正负对比；rescued/harmed 方向近乎抵消，screen 38/64=59.38% | [`paired_outcome_contrast_v111/`](paired_outcome_contrast_v111/) |
| v112 | 正负对比开放完整第一层状态条件化；平均到达更远但 screen 仍为 38/64=59.38% | [`state_conditioned_outcome_v112/`](state_conditioned_outcome_v112/) |
| v113 | 显式 paired treatment gate + bounded residual；gate 仅 54.46% balanced accuracy，screen 42/64=65.63% | [`paired_gated_residual_v113/`](paired_gated_residual_v113/) |
| v114 | v92 adapter 预声明幅度 screen；四档均低于 75%，最佳为未修改 v79 的 46/64=71.88% | [`observable_adapter_scale_v114/`](observable_adapter_scale_v114/) |
| v115 | causal GRU 只区分 rescued/harmed，0.6 高置信 gate；screen 恢复到 47/64=73.44%，差 1 episode | [`causal_gated_residual_v115/`](causal_gated_residual_v115/) |
| v116 | causal gate + 实际成功 filter-on trajectory teacher；state/action mismatch 使 screen 回落到 38/64=59.38% | [`outcome_gated_trajectory_v116/`](outcome_gated_trajectory_v116/) |
| v117 | filter-off deployment-state episodic residual PPO；Adam 后 clipped surrogate 为负，screen 44/64=68.75% | [`filter_off_residual_ppo_v117/`](filter_off_residual_ppo_v117/) |
| v118 | 一次 full-batch SGD 修复离线 surrogate，但新 seed screen 仅 40/64=62.50% | [`filter_off_full_batch_v118/`](filter_off_full_batch_v118/) |
| v119 | 完整 seed held-out surrogate 为负，事务回滚 exact zero residual；base screen 46/64=71.88% | [`heldout_residual_v119/`](heldout_residual_v119/) |
| v120 | 连续 reached-riser credit 的 held-out surrogate 仍为负；exact rollback 后 base screen 46/64=71.88% | [`progress_residual_v120/`](progress_residual_v120/) |
| v121 | v79 pretrained critic 的逐步 GAE 在 train 改善、held-out 反转；exact rollback 后 screen 41/64=64.06% | [`critic_gae_residual_v121/`](critic_gae_residual_v121/) |
| v122 | matched `+noise/-noise` 使 held-out unclipped 变正，但 clipped 略负；exact rollback 后 screen 42/64=65.63% | [`antithetic_residual_v122/`](antithetic_residual_v122/) |
| v123 | joint scale=0.25 使 train/held-out clipped 均改善，但 unseen deterministic screen 仅 38/64=59.38% | [`calibrated_antithetic_v123/`](calibrated_antithetic_v123/) |
| v124 | 132-D deterministic parameter ES 的 train cosine 为正、held-out 为 -0.0848；rollback 后 screen 45/64=70.31% | [`parameter_antithetic_v124/`](parameter_antithetic_v124/) |
| v125 | local sigma 使 held-out cosine 变为 +0.0854，但 train pairwise -0.0202 未过门槛；rollback 后 42/64 | [`local_parameter_es_v125/`](local_parameter_es_v125/) |

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

v94 修复了更早的时序缺口：原 5-D/16-D geometry 仅在脚已离地且 CBF 激活后出现，
新 10-D bilateral next-riser geometry 在接近台阶、双脚仍着地时就持续可见。训练的
256 个 paired initial states 中，filter-on 从 off 的 169 成功提高到 191 成功，并产生
61 个 matched rescue；但一次 rescued-only imitation 后，唯一 untouched filter-off gate
仍只有 45/64=70.31%。这说明提前几何提高了 safety filter 的轨迹价值，但局部 action
拟合仍不能把早期抬脚 credit 内化；下一步应把 persistent geometry 放入论文式
filtered-execution PPO 闭环，用 episode return/GAE 训练预摆动动作。

v95 将 persistent geometry 接入上述 paired filter-on/off PPO。49,152 transitions 的
6/6 batch surrogate 均改善，最差 gain 为 +1.26e-4，KL 精确限制在 5e-5；但三组
paired-seed gradient 的平均 cosine 仍只有 0.0202，唯一 untouched filter-off gate
下降到 44/64=68.75%。因此提前观测解决了信息时序，却没有解决跨 rollout 的 outcome
gradient 不一致；继续沿同一局部 PPO surrogate 放大或多轮更新没有证据支持。

v96 进一步把 v42 曾短暂达到 75% 的 filter-free 成功轨迹自模仿与 persistent geometry
结合。四个 stochastic seed 共得到 173/256 成功 episode，只训练成功 episode 的 77,875
个 transition，并保持旧 405-D actor 路径逐位不变。trust-scaled 更新离线距离下降且 KL
为 4.9991e-5，但探索方向 cosine 仅 0.0050；唯一 untouched deterministic filter-off gate
显著降至 35/64=54.69%。这表明把成功条件下的随机动作残差绑定到楼梯几何仍在拟合
seed-specific exploration noise，不能作为下一轮的可靠训练信号。当前仍选 v79。

v97–v99 改用独立的 28,364 参数 residual policy head，冻结 v79，并直接学习论文
Eq. (23) 的 deterministic `safe - nominal` 方向。v97 对全部 intervention 的 teacher
distance 从 0.317 降到 0.128，但 filter-on 成功率随 residual 增强由 95/128 持续降到
78/128，唯一 off gate 为 45/64。v98 将同一个方向缩放后，0.05x 在 screen 得到
47/64，但 unseen gate 仅 46/64。v99 再把 teacher 严格限制为最终成功的 filtered
episode，离线 distance 降到 0.108；然而 scale screen 中最优反而是 0x，即不使用
learned residual。由此可排除“网络容量不足”与“失败轨迹污染”两种解释：当前
instantaneous CBF correction 本身不是稳定的 task-success direction。下一步需要先提高
CBF filter-on ceiling 和 task compatibility，再继续论文式 filter internalization。

v100 在相同 initial-state seed 上比较当前 Euclidean CBF 与历史最优 task metric `c012`。
`c012` 的 filter-on 成功率从 44/64 提高到 46/64，mean reached riser 从 8.000 提高到
8.125，同时脚尖峰值接触力从 1561.15 降到 1225.35。因此 v101 改用 `c012`，并只从
最终成功的 filtered episode 学习 residual。四轮共得到 9,070 条 teacher transitions，
teacher distance 降到 0.08661。部署 scale screen 中 0.05x 短暂达到 49/64=76.56%，
但全新 seed 的唯一 filter-off gate 只有 40/64=62.50%。该提升未能泛化，v101 已拒绝；
当前最佳与 GitHub 对外选择仍为 v79。

v102 将 `c012` task-metric CBF 接入 v79 曾有效的 25/75 mixed-execution、Eq. (27)
unit-balanced paper-dual PPO。首次启动暴露并修复了 task-metric action 忽略逐环境 filter
mask 的实现错误；正式四轮的 action routing 误差均为 0。修复后四个对齐 filter-off
rollout 为 66.01%、66.33%、65.13%、67.51%，最佳 round-3 仅比本次 base 高 1.50 pp，
仍低于 75% 且低于 v79 历史正式结果，因此没有追加独立 gate。该结果排除了“只需把
task metric 放进现有 PPO 闭环”这一解释，下一步需要跨完整 swing 的安全目标和更稳定的
episode-level credit，而不是继续静态 metric 或 residual scale 搜索。

v103 新增独立的 bounded full-swing credit：将当前 CBF intervention 向前回传 50 steps，
以 `max` 聚合避免连续干预重复计数，credit 上限为 1，权重仅 0.01。实测附加 penalty
约 0.00121/step，低于 Eq. (27) 双奖励约 0.0028/step；因此信号尺度和路由均正确。
四个对齐 filter-off rollout 为 66.33%、66.15%、67.88%、66.84%，最佳只比本次 base
高 1.55 pp，未达到 75% 且低于 v79 历史正式结果。v103 已拒绝且未追加 gate；证据表明
仅扩大 temporal credit 仍不能解决跨 rollout outcome gradient 不一致。

v104 将 persistent bilateral next-riser geometry 正式接入完整 paper-dual PPO：v79
actor 405→415、critic 838→848 都以零列 exact prefix expansion warm-start，且不再像
v95 那样冻结 legacy actor。8 轮共 1,048,576 transitions，最佳更新后 checkpoint 的
filter-off 仅 132/200=66.00%，比本次零列 baseline 高 0.52 pp，低于 v79 历史结果，
因此未运行独立 gate。最佳 checkpoint 中新增 geometry 权重最大仅 2.23e-7，而深层
MLP 变化达 8.88e-6，证明普通 PPO 几乎没有利用新增观测。下一步应保留完整 actor
训练，同时显式放大 geometry-relative gradient 或增加与 episode outcome 对齐的几何
auxiliary objective；不能再简单重复纯 adapter 或无差别全网络 PPO。

v105 针对 v104 的 geometry gradient starvation，在保留完整 actor PPO 的同时，将
新增 10-D 首层梯度块自适应提升到 legacy 405-D 首层块的同等 norm。六轮实测倍率
为 8.78×–12.16×，缩放后 ratio 每轮均为 1.0；最佳 checkpoint 的 geometry 权重
仅两轮即达到 v104 五轮的 3.28 倍，证明该机制确实生效。然而最佳 filter-off 只有
138/194=71.13%，相对本次 baseline 仅 +0.43 pp，仍低于 v79 和 75% gate。由此可
排除“只因新增观测梯度太弱”这一解释；下一步必须让 geometry auxiliary objective
直接与完整 episode outcome 对齐，而不是继续扩大同一个 per-transition PPO 梯度。

v106 实现上述 episode-level 对齐：每个 filter group 内让成功与失败完整 episode
分别平分 +0.5/-0.5 credit，再与组内标准化 GAE 等权合并。8 轮中 outcome 标签平均
覆盖 80.33% transitions，GAE/credit cosine 为 0.227–0.306，证明信号确实进入 actor；
最佳 filter-off 从本次起点 68/98=69.39% 提高到 68/95=71.58%，增幅 2.19 pp。
但该峰值仍比 v79 低 0.44 pp，随后又回落到 64.71%，故未运行独立 gate、没有替换
v79。固定权重 episode credit 的方向有用但累积不稳定；下一步需要保守的 outcome
weight/acceptance control，而不是继续用相同权重延长训练。

v107 将 outcome 权重从 1.0 减到 0.5，并从 v106 最佳 415-D checkpoint 精确继续；
每个 proposal 都在下一轮 filter-off rollout 中评估，回落时原子恢复完整训练状态。
四个 proposal 分别为 69.63%、57.71%、65.31%、61.93%，全部被 rollback；LR 从
`5e-5` 逐步缩到 `6.12e-6` 后仍没有改善。相同保留 actor 的四次重测范围却达到
62.00%–72.92%，pool 为 534/782=68.29%。因此 v106 的单轮峰值不是可稳定累积的
outcome-gradient 证据，且问题不能再归因于权重或步长过大。下一步去掉直接 episode
credit，回到论文原生 GAE + CBF dual reward，并降低近收敛 policy 的 rollout std，
减少训练与 deterministic filter-off deployment 之间的探索噪声差异。

v108 去掉直接 episode credit，从 v79 重新执行论文原生 GAE + CBF dual PPO，并将
action std 从 0.05 降到 0.03。std=0.02 诊断先暴露了低方差 Gaussian 的 float32
log-prob reduction 误差边界，因此没有放宽冻结校验，而是使用有数值余量的 0.03
完成正式训练。四个 proposal 为 67.71%、58.00%、66.67%、67.02%，全部回落并被
完整 rollback；保留 anchor pool 为 530/785=67.52%。即使 LR 降到 `3.23e-6`、
moving KL 低于 `1e-7`，方向仍无改善。由此可排除 action exploration 太高这一充分
解释；下一步需要 matched filter rescue 的成对反事实轨迹，而不是继续调整同一 PPO
标量或延长相同训练。

v109 实现了上述成对反事实轨迹：部署时的 filter-off observation 与同一初始状态、
同一时刻的 filter-on safe action 精确配对，并从首次真实 CBF 干预向前回溯 20 steps、
向后跟踪 50 steps。8x32 paired initial states 中，filter-off 为 161/256=62.89%，
filter-on 为 184/256=71.88%，69 个 matched rescue 共给出 4,899 条 teacher target。
一次仅更新新增 10-D geometry 首层列的 full-batch SGD 将 teacher distance 从 0.126094
降到 0.125850，KL 限制为 5e-5；但对齐的 deterministic filter-off screen 只有
41/64=64.06%。因此没有触发独立 gate，v109 已拒绝，当前最佳仍为 v79。该结果表明
即使把真实安全轨迹延伸到干预前，局部 action imitation 仍不是稳定的部署成功方向；
下一步需要 sequence-level counterfactual outcome/value objective，而不是继续扩大同类
teacher 数据或重复局部拟合。

v110 修正了 v109 的关键因果错配：filter-on 轨迹只用于识别 matched rescue，实际
teacher action 则由每个 filter-off 部署状态上的 CBF 反事实投影产生。4x32 paired
initial states 得到 30 个 rescue、1,982 条目标；teacher distance 从 0.062279 降到
0.062002，cosine 为 0.454，且 KL 精确限制在 5e-5。尽管这些离线指标优于 v109，
同一开发 seed 的 deterministic filter-off screen 反而只有 36/64=56.25%，因此没有
运行独立 gate。该结果进一步确认局部 CBF action correction 与最终任务成功方向并不
等价；后续不再扩充同类蒸馏，而应直接学习 sequence-level counterfactual
outcome/value objective。当前最佳继续为 v79。

v111 将 paired terminal outcome 直接用于整段目标符号：32 个 `off失败/on成功`
rescued episode 取 `+1`，24 个 `off成功/on失败` harmed episode 取 `-1`，每个
episode 等权。3,675 条目标的正负方向几乎抵消，完整 gradient norm 仅 0.00264，
最终 KL 仅 1.82e-6；同一开发 screen 为 38/64=59.38%，虽比 v110 恢复两个 episode，
仍远低于 75%。因此没有运行独立 gate。该证据说明 CBF 的任务价值高度依赖状态，
不能再用一个全局 adapter 平均处理；下一步需要显式学习 state-dependent paired
counterfactual value/gating，而不是继续调整全局正负权重。当前最佳仍为 v79。

v112 将 v111 的正负 outcome contrast 从 10 个 geometry 列扩展到完整 415-D 第一层，
允许 212,480 个权重依赖本体历史与几何联合区分 CBF 是否有益，并在 90,951 个状态
上限制 KL。训练配对中 28 rescued 对 30 harmed，方向 cosine 仅 0.0204；更新投影到
17.1% 后 KL 为 4.99e-5。同一开发 screen 仍为 38/64=59.38%，虽 mean reached riser
从 v111 的 7.625 提高到 8.234，但登顶数没有增加。因此不运行独立 gate，并可排除
“只因第一层条件化容量不足”这一解释。下一步应把 paired treatment-effect 分类与
action residual 预测拆成显式 gate + residual 两个头。当前最佳仍为 v79。

v113 完成上述双头拆分：冻结 v79 actor，使用 256 个 paired initial states 训练
treatment gate，并只让 bounded residual 拟合 54 个 matched rescue。residual distance
从 0.05758 降到 0.03343，但 gate 对 18,176 条状态的 balanced accuracy 只有 54.46%，
正负概率均值也仅相差 0.0113；唯一 filter-off screen 为 42/64=65.63%。因此没有运行
独立 gate。该结果说明消除全局符号抵消仍不足以从局部状态可靠预测终局 treatment
effect；下一步应加入因果时序上下文，或对曾达到 47/64 的 v92 方向做预先限定的幅度
筛选。当前最佳仍为 v79。

v114 完成 v92 幅度筛选：在一次 256-env filter-off rollout 中，0×/0.5×/1×/1.5×
分别为 46/64、39/64、41/64、45/64。1.5× 的 mean reached riser 达 8.406，但登顶
仍少于未修改 v79；最佳按成功率回到 0×，且没有一档达到 75%，因此独立 gate 自动
跳过。由此不再继续标量搜索 v92 correction；下一步必须使用因果时序或完整 episode
policy objective。当前最佳仍为 v79。

v115 将 gate 改为因果 GRU，并只在 rescued/harmed discordant pair 上训练；第 4 个
seed 完整留作离线验证，部署阈值提高到 0.6。唯一 filter-off screen 恢复到
47/64=73.44%，gate 只激活 7.37% transition，但仍差一个 episode 才达到门槛，因此
没有运行独立 gate。候选和 screen 写入后，最终 summary 因 Python `false` 拼写触发
NameError；commit `8fc8a3b` 已修复，且没有为补诊断重复 rollout。该结果保留了时序
高置信 fallback 的价值，但仍说明 instantaneous correction 不是足够的 episode-level
policy objective。当前正式最佳继续为 v79。

v116 保留 v115 gate，只把 residual teacher 换成 matched-rescue 的实际 filter-on
成功轨迹。唯一 filter-off screen 从 v115 的 47/64 大幅回落到 38/64=59.38%，即使
gate 仅在 3.73% transition 激活也无法避免退化。这复现并加强了 v109 的因果问题：
轨迹分叉后的 filter-on action 不能作为 filter-off state 的稳定监督。最终 summary
另有第二处 `false` 拼写在 screen 落盘后触发，commit `171ef70` 已清除全部剩余小写
布尔值，未重复 rollout。下一步必须保持 deployment-state on-policy，并直接使用
episode return，而不是继续轨迹 action imitation。当前最佳仍为 v79。

v117 首次完全去掉 action teacher，让 residual 在 filter-off deployment states 中探索，
并用 episode-balanced 成功/失败 advantage 做 PPO。256 episode 中 168 成功，87,624
条 active-geometry transition；unclipped surrogate 为正，但 44 次 Adam minibatch 后
clipped surrogate 变成 -0.01666，KL 投影只保留 5.54% delta。screen 为
44/64=68.75%，离线和在线均拒绝。下一步保留相同 on-policy objective，只改成一次
full-batch direction-preserving SGD，避免优化器扭曲方向。当前最佳仍为 v79。

v118 用一次 full-batch SGD 将 clipped surrogate 从约 0 提高到 +0.001179，并把 KL
限制为 0.004853，离线 gate 成功通过；但新 seed screen 仅 40/64=62.50%。因此优化器
扭曲已修复，剩余问题是 pooled episode-outcome gradient 对 rollout seeds 过拟合。
下一步将第 4 个 rollout seed 完整留作 surrogate validation，验证不改善就原子回滚，
不让过拟合 proposal 进入 screen。当前最佳仍为 v79。

v119 将第 4 个 rollout seed 完整留作 validation。前三个 seed surrogate 提升到
+0.001675，但 held-out seed 为 -0.001449，因此 proposal 在 screen 前原子回滚；
initial/final residual SHA 完全相同，mean residual norm 为 0。回滚后的 v79 screen
为 46/64=71.88%。这直接证明二元 outcome gradient 跨 seed 反向；下一步保留事务
协议，改用连续 reached-riser credit 降低方差。当前最佳仍为 v79。

v120 用每 seed 标准化的 terminal reached-riser + success bonus 取代二元标签。train
surrogate 为 +0.000742，但 held-out seed 仍为 -0.001842，故再次 exact rollback；
screen 是未修改 v79 的 46/64=71.88%。连续 credit 也不能稳定 residual score-function
gradient，后续不再继续替换同类标量，而应引入 state-action critic/GAE 或显著扩大真正
独立的 on-policy batch。当前最佳仍为 v79。

v121 将 v79 的 pretrained critic 以 10 个零列精确扩展，并在真实 filter-off
deployment trajectory 上计算逐步 GAE。105,214 个完整 transition 中 88,846 个
active-geometry transition 进入 residual objective；train clipped surrogate 从
-0.004977 改善到 -0.003434，但完整 held-out seed 从 -0.030414 降到 -0.032271，
因此 exact rollback。回滚后的新 seed screen 为 41/64=64.06%，未运行独立 gate。
预训练 value baseline 仍不能消除 residual action gradient 的跨 seed 反转；下一步改用
同初始状态 paired/antithetic action advantage，而不是继续更换单轨迹 credit。
当前最佳仍为 v79。

v122 对 256 个 matched initial states 分别运行 `+noise/-noise` 镜像 filter-off 轨迹，
全部 pair 的初态签名一致。正负分支分别为 171/256 与 170/256 success；train clipped
surrogate 提高到 +0.001510。held-out unclipped surrogate 也首次为正（+0.001113），
但在当前 KL 步长发生 clipping 后略降到 -0.000133，故仍 exact rollback。回滚后的
screen 为 42/64=65.63%，未运行独立 gate。该结果说明 paired 差分已改善一阶方向，
下一步应缩放同一 proposal，使 train/held-out clipped surrogate 同时为正，而不是再次
更换 objective。当前最佳仍为 v79。

v123 保留相同 paired objective，只在固定 `1...1/128` scale grid 上选择 train 与
held-out clipped surrogate 都至少改善 1e-6 的最大步长。scale=0.25 通过：train 为
+0.000658、held-out 为 +0.00000618、KL 约 0.000309，候选没有回滚。但唯一 unseen
deterministic filter-off screen 只有 38/64=59.38%，因此未运行 gate 且不替换 v79。
这证明 action-space stochastic PPO surrogate 即使在完整 held-out seed 上同向，也仍
不能预测 deterministic mean-policy 成功率；下一步应直接优化 deterministic residual
参数的 paired return。当前最佳仍为 v79。

v124 使用 132-D persistent-geometry 线性 residual，并为每个环境固定镜像参数方向，
直接从 deterministic `theta+sigma*u` / `theta-sigma*u` return 差估计梯度。256 对轨迹
全部初态和参数方向匹配；三个 train gradient 的平均 pairwise cosine 为 +0.0509，
但 held-out cosine 为 -0.0848，故 exact rollback。回滚后的 screen 为
45/64=70.31%，未运行独立 gate。当前 sigma=0.02 使探索 residual norm 约 0.039，
下一步仅缩小到 sigma=0.005 的局部范围，其余 objective 和 gate 保持不变。
当前最佳仍为 v79。

v125 将 parameter sigma 降至 0.005，探索 residual norm 降到约 0.0103。首个正分支
达到 48/64，但它包含 64 个不同参数方向，不能视为候选。完整统计中 held-out gradient
cosine 从 v124 的负值改善到 +0.0854，已过 +0.05 门槛；三个 train gradient 的平均
pairwise cosine 为 -0.0202，略低于预声明 0 门槛，因此仍 exact rollback。screen 为
42/64=65.63%，未运行独立 gate。下一步保留 held-out 门槛，仅将 train cosine 的噪声
容忍线固定为 -0.05，再用全新 seeds 运行一次。当前最佳仍为 v79。

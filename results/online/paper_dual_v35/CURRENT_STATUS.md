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

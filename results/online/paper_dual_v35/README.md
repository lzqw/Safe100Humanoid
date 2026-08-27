# v35 Paper-Aligned Dual CBF-RL Pilot

## 当前最佳结果：A2×4 → A1×4

第一阶段 ablation 后冻结 `raw_moderate` reward，并将前四轮 A2 residual
teacher 与后四轮 A1 full-action teacher 串联。三个 context 分别从同一 base
checkpoint 独立训练；每个 context 只做一次独立种子的 64-episode on/off pilot。

| Context | Stair profile | Filter on | Filter off | Off - on |
|---|---|---:|---:|---:|
| F1 | uniform 18.0 cm, 9 risers | **78.12% (50/64)** | **76.56% (49/64)** | -1.56 pp |
| F2 | uniform 18.4 cm, 9 risers | 70.31% (45/64) | 68.75% (44/64) | -1.56 pp |
| F3 | nonuniform 17.6–18.4 cm, 11 risers | 65.62% (42/64) | 59.38% (38/64) | -6.25 pp |
| Macro / micro mean | 3 contexts, 192 episodes/condition | **71.35% (137/192)** | **68.23% (131/192)** | **-3.12 pp** |

该 staged actor 在 F1 同时超过第一阶段最高 on 和最高 off，并把三 context
平均 on/off gap 收窄到 3.12 pp，说明训练时过滤得到的行为已经较强地内化到
filter-free actor 中。它是当前 v35 best result，但仍不是最终论文级复现：F2/F3
绝对成功率未达到 75%，每个条件只有一个 64-episode pilot，而且 A1 阶段最终
moving KL 为 0.67–0.71，表明 full-action teacher 更新过强。

后续已经分别测试 soft-A1 与 stair-height curriculum。前者降低 KL 但降低任务
成功率；后者把 F2 filter-on 提高到 81.25%，但没有同步改善 filter-off。因此
staged actor 仍是当前综合结果，下一步是把 curriculum 的 A2 阶段接到固定目标
高度的 A1 consolidation，而不是增加相同配置的重复评估。

### Soft-A1 follow-up（已拒绝）

按实测梯度比例把 A1 teacher weight 从 `0.1` 降到 `0.0075` 后，F2 第 5
轮 KL 从原方法的 `0.2214` 降至 `0.00497`，最终 KL 从 `0.7107` 降至
`0.00606`。但同一训练/评估 seed 下，filter-on/off 只有
62.50%/64.06%，低于 current best 的 70.31%/68.75%。该方向按预设 gate
停止，没有运行 F3；详见 [`soft_a1_f2_summary.json`](soft_a1_f2_summary.json)。
这说明单纯抑制策略移动能修复优化诊断，却不能提升困难台阶的任务能力，下一步
转向 stair-height curriculum；该后续实验结果见下一节。

### Stair-height curriculum follow-up（仅改善 filter-on）

F2 使用 13.0 cm 到 18.4 cm 的五级高度课程，训练期间仍执行 CBF-filtered
action，teacher schedule 保持 A2×4 → A1×4。最终在固定 18.4 cm F2、同一独立
评估 seed 的 64 episodes 上得到：

| F2 run | Filter on | Filter off | Off - on | Training time |
|---|---:|---:|---:|---:|
| fixed-height staged（当前综合最佳） | 70.31% (45/64) | **68.75% (44/64)** | -1.56 pp | 231.0 s |
| five-level height curriculum | **81.25% (52/64)** | 67.19% (43/64) | -14.06 pp | 233.8 s |

课程训练将 filter-on 提高 10.94 pp，并使其超过 75% gate；但 filter-off 降低
1.56 pp，on/off gap 扩大到 14.06 pp，所以没有把它选为最终 actor。结果表明
课程提高了安全过滤器辅助下的目标高度通过能力，但高台阶上的 CBF correction
还没有充分内化。下一次算法实验将使用 curriculum A2 warm-up，随后只在固定
18.4 cm 目标高度做 A1 consolidation。精确 provenance 与逐轮诊断见
[`height_curriculum_f2_summary.json`](height_curriculum_f2_summary.json)。

### Fixed-target continuation follow-up（前一阶段的 F2 Pareto 最优）

课程训练第 4 轮 A2 checkpoint 被精确 SHA-256 锁定后，用作三个固定 18.4 cm
F2 continuation 的共同起点。每个 candidate 只运行一次训练和一次预设 gate；
明显失败时早停。

| Continuation from curriculum round 4 | Filter on | Filter off | Final KL | Decision |
|---|---:|---:|---:|---|
| A1 Gaussian-NLL ×4 | **81.25% (52/64)** | **68.75% (44/64)** | 0.6518 | 当时的 F2 Pareto best，仍未达 off≥75% |
| sampled-action A2 η=1 ×4 | 79.69% (51/64) | 51.56% (33/64) | 0.00056 | rejected |
| deterministic-mean CBF A2 η=1 ×4 | 65.62% (42/64) | not run | 0.00067 | on gate 早停 |

A1 continuation 保持课程模型的 81.25% filter-on，同时将 filter-off 从 67.19%
提高到 68.75%，因此成为当时 F2 的 Pareto 最优；它相对原 fixed-height staged
F2 则是 on +10.94 pp、off 持平。结果仍不足以宣称达到论文级效果，因为
filter-off 未达到 75%，paired gap 仍为 12.50 pp，而且 A1 更新的 KL/clip 仍过高。

两个 A2 follow-up 证明“优化稳定”不等于“安全行为内化”：把 sampled correction
扩大到 η=1 会显著降低 off；进一步对 frozen deterministic mean 做同状态 shadow
CBF projection 后，训练数据流误差虽为 0，filter-on 仍明显退化，因此按门槛不再
运行 off。代码保留该机制用于审计，但不选择其 actor。完整结果、模型哈希及逐轮
诊断见 [`continuation_f2_summary.json`](continuation_f2_summary.json)。

### Unshielded counterfactual + on-policy mean DAgger（当前 F2 off 最佳）

为缩小训练时过滤与部署时无过滤的分布差异，后续 rollout 直接执行 nominal
unshielded action；CBF 不再替换训练动作，而是在同一状态上投影 deterministic
policy mean，作为反事实监督。所有候选仍只做一次预设训练 gate，只有通过 gate
的 actor 才运行同一个 seed 的 64-episode paired on/off 评估。

| Candidate | Filter on | Filter off | Off - on | Decision |
|---|---:|---:|---:|---|
| 50/50 curriculum–A1 model soup | 65.62% (42/64) | not run | — | on gate rejected |
| unshielded A0，selected round 3 | 76.56% (49/64) | 70.31% (45/64) | -6.25 pp | next-stage base |
| all-intervention mean-CBF DAgger，η=0.25 ×3 | **79.69% (51/64)** | **71.88% (46/64)** | -7.81 pp | **current F2 off best** |
| failure-only mean-CBF teacher ×3 | 78.12% (50/64) | 70.31% (45/64) | -7.81 pp | rejected |

当前最佳相对 fixed-target A1 将 filter-off 从 44/64 提高到 46/64（+3.12 pp），
并把 gap 从 12.50 pp 缩到 7.81 pp；相对 unshielded A0，它的 on/off 分别提高
3.12/1.56 pp。它仍不满足目标：filter-off 距 75% gate 还差 2 个成功 episode，
且目前只有一个评估 seed，因此不宣称达到论文级复现。

额外两轮、强 reward、2048-step double batch、较小 rollout noise，以及仅按失败
episode 过滤 teacher label 均未改善训练 gate 或 paired 结果，已停止，避免消耗
4080 做低价值重复验证。精确训练配置、逐轮指标、checkpoint/actor SHA-256 与这些
负结果见 [`unshielded_f2_summary.json`](unshielded_f2_summary.json)。

#### Failure-focused actor ablation（已拒绝）

为直接保护成功轨迹，又尝试只让完整失败 episode 进入 PPO 与 mean-CBF teacher；
成功 episode 只保留 round-reference KL，critic 仍使用全部 transition。两种损失
归一化都从当前最佳 checkpoint 独立训练 3 轮：

| PPO mask normalization | Round 1 | Round 2 | Round 3 | Decision |
|---|---:|---:|---:|---|
| failed-transition mean | 68.46% | 68.61% | 68.70% | rejected，全部 actor step gradient-clipped |
| full-rollout mean | 66.67% | 65.89% | 64.71% | rejected，成功率持续下降 |

第一种做法因失败 transition 只占约 20% 而将 PPO 梯度隐式放大约 5 倍；第二种
修正尺度后虽降低了梯度范数，仍未恢复任务成功率。这说明问题不是简单的成功样本
覆盖失败信号；该方向已按训练 gate 停止，未追加 paired 评估，当前最佳不变。

#### Checkpoint alignment + success-local KL（已拒绝）

最新审计确认训练过程的时序为：第 N 轮 rollout 在该轮更新之前完成，因此它评估
的是第 N−1 轮 checkpoint。训练器现已显式记录 rollout actor SHA-256、checkpoint
轮次及 `rollout_precedes_update`，后续不再用同编号的 pre-update rollout 为
post-update checkpoint 排名。

| Audit / candidate | Filter on | Filter off | Off - on | Decision |
|---|---:|---:|---:|---|
| 原 mean-CBF run 的 aligned round-2 checkpoint | 79.69% (51/64) | 59.38% (38/64) | -20.31 pp | rejected |
| success-local KL β=2、LR=5e-6，selected round 1 | 68.75% (44/64) | 70.31% (45/64) | +1.56 pp | rejected |
| success-local KL β=2、LR=1e-6 | not run | not run | — | rollout gate rejected |

原 run 的第 3 轮 stochastic rollout 成功率为 73.68%，它实际对应 round-2
checkpoint；该 checkpoint 的 deterministic filter-off 只有 38/64，说明训练
rollout 不能单独用于筛选部署 actor。随后在当前最佳 checkpoint 上加入完整成功
episode 的 local reference KL：LR=5e-6 时 aligned rollout 一度由 67.42% 升至
77.60%，但选中的 round-1 actor 在 deterministic paired 评估中只有 44/64 on、
45/64 off；LR=1e-6 时 aligned rollout 仅由 61.65% 升至 63.43%，因此未评估。

本节两组新 paired 评估内部使用相同初始状态签名
`8ea76a17dff5c8248231bbf34d984be2bb9506415c0830f3b6d8560d14fda1ee`；它与历史
current-best 评估的 `3f8c032be07f7741ddf1da02b72cc65ba59e4bc8f61227d09dc99de9582fa40b`
不同，所以只比较各自内部的 on/off，不宣称跨历史结果的逐 episode 配对。当前
F2 off 最佳仍为 46/64，距离 75% gate 还差 2 个成功 episode。

#### Stable reset + shielded/unshielded bridge（未改善 filter-off）

评估器现在在 runner 构造及 checkpoint 加载后，紧邻最终 episode reset 重新设置
底层环境 seed，避免模型初始化消耗全局 RNG 后改变测试 episode。新协议在相同
seed `201350902` 下固定得到初始状态签名
`bc4c4cd08aef0d22b69dbf985eaef78a226c402428c72b1ea431593dda42b22c`；所有后续
candidate 都与该固定基线比较。

| Stable-reset F2 run | Filter on | Filter off | Off - on | Decision |
|---|---:|---:|---:|---|
| 当前最佳 actor 的固定基线 | 75.00% (48/64) | **68.75% (44/64)** | -6.25 pp | future comparison baseline |
| 成功 shielded episode 的 CBF 蒸馏，round 1 | **84.38% (54/64)** | 67.19% (43/64) | -17.19 pp | rejected |
| 50/50 shielded/unshielded filter-dropout 蒸馏 | not run | not run | — | rollout gate rejected |

第一项新方法只在完整 reached-top 的 shielded episode 上学习 deterministic-mean
CBF correction，并关闭 PPO/entropy actor 梯度；它把 filter-on 提高 6 个 episode，
但 filter-off 降低 1 个，说明更新仍依赖过滤后的状态分布。第二项因此让 32 个环境
中的 16 个执行 CBF、16 个执行 nominal action，并在下一轮交换两组。运行时审计
确认 filter fraction 精确为 0.5、executed-action routing 误差为 0；但 rollout 从
43/66（65.15%）降至 39/67（58.21%），故不再做 paired 评估。固定-reset 协议下
filter-off 距 75% gate 为 4 个 episode；历史未固定 reset 的最高记录仍是 46/64，
两者不做逐 episode 混合比较。

#### Matched filter-rescue off-state distillation（已拒绝）

为避免 shielded 成功轨迹中混入本来就能无过滤成功的样本，v36 用独立训练 seed
对相同初始状态分别执行 filter-on/off，只保留“on 成功、off 失败”的 episode；
监督状态和 CBF correction 均取自对应的 unshielded 失败轨迹。受共享 GPU 显存
限制，本次短 pilot 使用 32 个环境，识别到 5 个 filter-rescued episode、238 个
CBF teacher transition，并用全数据 reference KL 约束一次 actor 更新。

| Stable-reset F2 candidate | Training pair on/off | Teacher distance | Held-out filter off | Decision |
|---|---:|---:|---:|---|
| matched rescue v36 | 24/32 vs 25/32；5 rescued | 0.14530 → 0.14089 | 43/64（67.19%） | rejected |
| 固定基线 | — | — | **44/64（68.75%）** | retained |

候选在独立 seed `201350902`、固定初始状态签名
`bc4c4cd08aef0d22b69dbf985eaef78a226c402428c72b1ea431593dda42b22c`
上比基线少 1 个成功 episode，因此按部署 gate 停止，没有追加 filter-on 或多 seed
验证。候选 actor SHA-256 为
`ef07a91527c376d4d2baaa563a770725329d1ca4f3763487147eb346811d1f7e`；
训练、配对样本与逐 episode 结果见
[`rescue_distill_v36/`](rescue_distill_v36/)。

#### Matched-rescue shielded distillation + trust region（独立 seed 未泛化）

v36 的逐 episode 对比显示，它虽然救回 12 个基线失败 episode，却破坏了 13 个
原成功 episode；平均 forward-KL `0.00455` 对当前 humanoid policy 仍然过大。
v37 因此改用更贴近论文训练分布的 filter-on 成功 rescue 轨迹作为 teacher 状态，
并在更新后用二分参数投影把数据集平均 KL 硬限制为 `2.5e-4`。32-env 短 pilot
识别出 7 个 matched rescue episode 和 291 个 teacher transition；未投影更新的
KL 为 `0.004827`，最终只保留 `22.75%` 参数位移，实际 KL 为 `0.00024986`。

| Stable-reset F2 | Filter on | Filter off | Off - on | Decision |
|---|---:|---:|---:|---|
| 固定基线，adaptive seed `201350902` | 75.00% (48/64) | 68.75% (44/64) | -6.25 pp | retained |
| matched-rescue shielded v37，同 seed | **81.25% (52/64)** | **75.00% (48/64)** | -6.25 pp | provisional gate passed |
| 固定基线，untouched seed `201350912` | not run | 67.19% (43/64) | — | confirmation baseline |
| matched-rescue shielded v37，同 untouched seed | not run | 67.19% (43/64) | — | rejected; no improvement |

候选在已反复用于 gate 的 seed `201350902` 上相对基线的 filter-on/off 各净增
4 个成功 episode，并首次达到 `filter-off >= 75%`。但在预先未使用的 seed
`201350912` 上，候选与基线同为 43/64（67.19%），没有泛化提升，因此停止该 seed
的 filter-on 并不选择该 actor。actor SHA-256 为
`fdc872355f38a31f7c35d16e1ee5f151d156ae2907cf2f5cf8492f7d18891733`。
完整训练摘要、比较摘要、adaptive paired on/off 与 untouched filter-off 逐 episode
证据见
[`rescue_shielded_v37/`](rescue_shielded_v37/)。

#### Multi-seed rescue aggregation + off-success trust anchor（已拒绝）

v38 将 3 个训练 seed、每个 32 个环境的 matched-rescue shielded 数据合并，得到
23 个 rescue episode 和 860 个 teacher transition。KL 投影不再使用 teacher
状态，而是在 30,797 个“原策略 filter-off 成功”状态上限制平均 forward-KL，
希望直接保护既有成功行为。训练耗时 97.6 秒，最终 KL 为 `0.00024972`。

在较难 gate seed `201350912` 上，固定基线 filter-off 为 43/64（67.19%），v38
只有 38/64（59.38%），净少 5 个成功 episode，因此立即停止且不运行 filter-on。
这说明问题不只是单 seed 标签过拟合；当前 supervised correction 的聚合方向本身
不能可靠替代论文中由 on-policy return 决定的策略更新。完整证据见
[`multi_seed_rescue_v38/`](multi_seed_rescue_v38/)。

#### Paper-style filtered PPO + bounded dual reward continuation（已拒绝）

v39 完全移除显式 teacher，只保留论文的训练时 CBF filtering、PPO 与 bounded
negative-margin/action-distance reward；从固定基线以 `2e-6` actor LR 训练 4 轮。
第 4 轮 pre-update rollout 达到 76.81%，按 checkpoint alignment 对应 round-3
actor，训练耗时 115.2 秒。

但该 round-3 actor 在 hard seed `201350912` 的 filter-off 只有 33/64
（51.56%），比同 seed 基线 43/64 少 10 个成功 episode，因此不运行 filter-on。
这直接表明训练 rollout 的提升依赖运行时 filter，未被 nominal actor 内化；仅延长
相同 filtered PPO 方向不符合部署目标。完整证据见
[`filtered_ppo_v39/`](filtered_ppo_v39/)。

#### Last-layer-only rescue distillation（已拒绝）

v40 复用 v38 的冻结多 seed 数据，不再进行 rollout，只更新 actor 最后一个线性层
的 1,548 个参数。离线训练耗时 9.6 秒，off-success 状态上的 KL 为 `0.0002234`，
低于预设上限且无需参数缩放。但 hard seed `201350912` 的 filter-off 仍只有
36/64（56.25%），比基线少 7 个成功 episode。

因此 v36–v38 的失败不是单纯由全网络表征漂移造成；CBF correction 的 supervised
方向本身无法稳定提高 filter-free task return。该路线至此停止，证据见
[`last_layer_rescue_v40/`](last_layer_rescue_v40/)。

#### Filter-free elite self-imitation（单 gate seed 成功，独立 seed 未通过）

v41 不再拟合反事实 CBF correction，而是在 3 个新训练 seed 上用 std=0.05 的
当前策略执行无过滤探索，收集 96 个 episode，其中 72 个成功；只把成功 episode
的实际执行动作作为 deterministic actor 的监督。4-minibatch 更新使全数据 elite
loss 略升，因此未评估。v42 复用完全相同的数据，只做一次 full-batch 输出层更新，
elite Smooth-L1 从 `0.02121596` 降至 `0.02121466`，KL 为 `9.996e-5`。

| v42 stable-reset F2 seed | Filter on | Filter off | Off - on | Result |
|---|---:|---:|---:|---|
| gate `201350912` | 76.56% (49/64) | **75.00% (48/64)** | -1.56 pp | gate passed；off 比基线 +5 |
| untouched `201350922` | not run | 68.75% (44/64) | — | stopped below gate |

v42 首次在较难 seed 上同时达到 on/off 75% 且 gap 仅 1.56 pp，证明成功动作的
filter-free self-imitation 比直接 CBF 蒸馏更接近论文部署目标；但 untouched seed
仍未达到绝对 gate，所以尚不选择为最终稳健 actor。完整训练、paired gate 与
untouched 逐 episode 证据见
[`elite_self_imitation_v41_v42/`](elite_self_imitation_v41_v42/)。

#### Six-seed elite self-imitation（有改善，仍低于 gate）

v43 新增 3 个训练 seed 的 96 个无过滤探索 episode（64 个成功），并与 v41
数据合并为 6 个 seed、80,044 transitions，其中 60,976 个来自成功 episode。
只对 actor 最后一层做一次 full-batch 更新；elite Smooth-L1 从
`0.02121494` 降至 `0.02121404`，reference KL 为 `9.991e-5`，训练耗时 10.3 秒。

在 v42 未通过的 stable-reset seed `201350922` 上，无过滤成功率从 44/64 提高到
46/64（71.875%），但仍比 75% gate 少 2 个 episode。因此按预设早停，不运行
filter-on 或额外 seed，也不选择该 actor。代码提交、训练摘要及逐 episode 证据见
[`elite_self_imitation_v43_6seed/`](elite_self_imitation_v43_6seed/)。

把同一 elite 更新的 KL 上限从 `1e-4` 提高到 `2e-4` 后，该 seed 降至
42/64，证明不是继续增大成功轨迹拟合步长即可解决。

#### Episode-balanced outcome PPO（最佳点 47/64，仍未通过）

v44 把六个训练 seed 的成功/失败 episode 做中心化 PPO 优势，每个 episode 等权，
并用逐坐标 trimmed mean 聚合 seed gradient。离线 6/6 seed surrogate 均改善，
但 seed gradient 平均余弦只有 `-0.0032`。Adam 候选在 gate seed `201350922`
达到 48/64，却在 untouched seed `201350932` 从 base 的 46/64 降到 37/64，
说明 Adam 第一步的逐坐标符号缩放放大了不一致梯度。

v45 改用保留梯度幅度的 SGD trust-region step。KL=`2.5e-5` 时，同一独立对照从
46/64 小幅提高到 **47/64（73.44%）**；将 KL 增至 `5e-5` 后反而降到 42/64。
因此 47/64 是该路线最佳点，但仍比 75% gate 少一个 episode，按计划停止且不跑
filter-on。完整离线诊断、base 对照和逐 episode 证据见
[`outcome_ppo_v44_v45/`](outcome_ppo_v44_v45/)。

## 方法与第一阶段 F1 ablation

本目录发布 v35 第一阶段 F1 pilot 的关键结果。该阶段依据
[CBF-RL](https://arxiv.org/abs/2510.14959) 的训练时安全过滤与双重奖励，
并参考作者公开的
[navigation demo](https://github.com/lzyang2000/cbf-rl-navigation-demo)
把策略动作与安全动作之间的距离改到 actor 原生动作坐标中计算。

三组训练均从同一个 838-D privileged-critic checkpoint 开始，在 RTX 4080
上使用 64 个并行环境、8 轮、每轮 1024 steps。训练种子为 `201350101`；
随后用独立种子 `201350901` 做 64 个 deterministic episodes 的 F1 评估。
训练 rollout 始终执行 CBF-filtered action。

| Run | Teacher | Margin / imitation weight | Filter on | Filter off | Off - on | Training time |
|---|---|---:|---:|---:|---:|---:|
| `raw_moderate` | A0, no teacher | 1 / 10 | 71.88% (46/64) | 65.62% (42/64) | -6.25 pp | 231.3 s |
| `raw_demo` | A0, no teacher | 10 / 100 | **76.56% (49/64)** | 57.81% (37/64) | -18.75 pp | 206.7 s |
| `raw_moderate_A1` | A1, corrective full-action teacher | 1 / 10 | 67.19% (43/64) | **76.56% (49/64)** | +9.38 pp | 217.5 s |

第一阶段核心观察：A1 corrective teacher 将相同 `raw_moderate` reward 的 filter-off
成功率从 65.62% 提高到 76.56%（+10.94 pp），说明安全动作确实更多地被策略
内化；但 filter-on 成功率同时从 71.88% 降到 67.19%（-4.69 pp）。作者 demo
的强权重则提高了 filter-on 表现，却显著扩大 on/off 差距。因此目前没有一个
方案同时保持较高的 filter-on 与 filter-off 成功率，所以该 ablation 本身没有
winner。后续 staged experiment 解决了 F1 矛盾，并已冻结扩展到 F2/F3。

## Published evidence

- [`pilot_summary.json`](pilot_summary.json): protocol、精确指标、代码与模型哈希。
- [`eval_summary.csv`](eval_summary.csv): 三个 actor 的 filter-on/off 独立评估。
- [`round_metrics.csv`](round_metrics.csv): 24 个训练轮次的 rollout 与 CBF 诊断。
- [`staged_summary.json`](staged_summary.json): 当前最佳 staged 方法的三 context 汇总。
- [`staged_eval_summary.csv`](staged_eval_summary.csv): staged 方法的六个 on/off 结果。
- [`staged_round_metrics.csv`](staged_round_metrics.csv): staged 方法全部 24 轮及 KL 诊断。
- [`soft_a1_f2_summary.json`](soft_a1_f2_summary.json): 低 KL 但成功率退化的 F2 负结果。
- [`height_curriculum_f2_summary.json`](height_curriculum_f2_summary.json): F2 五级高度课程的配对评估、模型哈希与逐轮诊断。
- [`continuation_f2_summary.json`](continuation_f2_summary.json): 三个固定目标 continuation 的 gate、早停决定与 provenance。
- [`unshielded_f2_summary.json`](unshielded_f2_summary.json): 无过滤 rollout、反事实 mean-CBF DAgger、模型哈希、paired 评估及训练 gate 负结果。
- [`rescue_distill_v36/`](rescue_distill_v36/): matched filter-rescue 训练摘要、配对结果及留出 filter-off 逐 episode 证据。
- [`rescue_shielded_v37/`](rescue_shielded_v37/): trust-region 训练、adaptive paired gate 与失败的 untouched-seed 泛化确认。
- [`multi_seed_rescue_v38/`](multi_seed_rescue_v38/): 三训练 seed 聚合、off-success trust anchor 及失败的 filter-off gate。
- [`filtered_ppo_v39/`](filtered_ppo_v39/): 论文式 filtered PPO continuation、checkpoint 对齐及失败的 hard-seed filter-off gate。
- [`last_layer_rescue_v40/`](last_layer_rescue_v40/): 输出层局部更新、离线 KL 审计及失败的 hard-seed gate。
- [`elite_self_imitation_v41_v42/`](elite_self_imitation_v41_v42/): 无过滤成功轨迹收集、full-batch 更新、成功的 paired gate 与失败的 untouched-seed 确认。
- [`elite_self_imitation_v43_6seed/`](elite_self_imitation_v43_6seed/): 六训练 seed 合并、KL 放大负结果及稳定 seed gate。
- [`outcome_ppo_v44_v45/`](outcome_ppo_v44_v45/): episode-balanced outcome PPO、Adam 泛化失败、SGD trust-region 最佳点与独立 base 对照。

大型 `.pt` checkpoint 没有提交进 Git；其 SHA-256 已写入
`pilot_summary.json`，原始 checkpoint 和逐 episode 文件保留在 4080 artifact
目录中。这个结果包只包含足以审计当前判断的聚合记录，不是多种子正式评估。

## Reproduction boundary

- Base checkpoint SHA-256:
  `cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8`
- A0 source commit: `e532d606c78a42839bf78221a37d6a020aac878a`
- A1 source commit: `c767e7b9df913dedea93c3715e0f8c422301f218`
- Staged source commit: `b21eceecc10ec651a4d3c45fe9675a89f00c1410`
- Reward implementation: `src/tasks/stairs_cbf/paper_dual_v35.py`
- Trainer: `experiments/scripts/refine_paper_dual_v35.py`
- Target contexts: F1, F2, F3 as listed in the staged table
- Original/staged runs enable the runtime filter during training. The newest
  unshielded runs execute nominal actions during training and use CBF only for
  same-state counterfactual labels; paired evaluation still toggles the filter.

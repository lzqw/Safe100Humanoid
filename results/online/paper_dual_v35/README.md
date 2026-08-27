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

### Fixed-target continuation follow-up（新的 F2 Pareto 最优）

课程训练第 4 轮 A2 checkpoint 被精确 SHA-256 锁定后，用作三个固定 18.4 cm
F2 continuation 的共同起点。每个 candidate 只运行一次训练和一次预设 gate；
明显失败时早停。

| Continuation from curriculum round 4 | Filter on | Filter off | Final KL | Decision |
|---|---:|---:|---:|---|
| A1 Gaussian-NLL ×4 | **81.25% (52/64)** | **68.75% (44/64)** | 0.6518 | F2 Pareto best，仍未达 off≥75% |
| sampled-action A2 η=1 ×4 | 79.69% (51/64) | 51.56% (33/64) | 0.00056 | rejected |
| deterministic-mean CBF A2 η=1 ×4 | 65.62% (42/64) | not run | 0.00067 | on gate 早停 |

A1 continuation 保持课程模型的 81.25% filter-on，同时将 filter-off 从 67.19%
提高到 68.75%，因此成为当前 F2 的 Pareto 最优；它相对原 fixed-height staged
F2 则是 on +10.94 pp、off 持平。结果仍不足以宣称达到论文级效果，因为
filter-off 未达到 75%，paired gap 仍为 12.50 pp，而且 A1 更新的 KL/clip 仍过高。

两个 A2 follow-up 证明“优化稳定”不等于“安全行为内化”：把 sampled correction
扩大到 η=1 会显著降低 off；进一步对 frozen deterministic mean 做同状态 shadow
CBF projection 后，训练数据流误差虽为 0，filter-on 仍明显退化，因此按门槛不再
运行 off。代码保留该机制用于审计，但不选择其 actor。完整结果、模型哈希及逐轮
诊断见 [`continuation_f2_summary.json`](continuation_f2_summary.json)。

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
- Runtime filter is enabled during training and toggled only for the paired evaluation.

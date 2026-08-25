# v32 Long-Horizon CBF-Protected Continual Refinement

## 中文摘要

v32 保持 v31 A2 的 Actor/Critic、CBF、reward、raw-action PPO、soft moving-KL 与 CBF-guided correction 不变。三个 per-context 策略从各自 v31 A2 round 8 继续到固定 round 24，并比较恒定学习率与后期衰减；另一个 mixed 策略从共同 base 开始，用每轮轮换的 22/21/21 个 F1/F2/F3 环境训练到固定 round 24。没有 candidate、性能/KL gate、best checkpoint 或结果依赖重跑。

| method | mean CBF-on success | mean CBF-off success | D0 success | interventions/riser |
|---|---:|---:|---:|---:|
| v31_A2 | 71.09% | 55.47% | 91.41% | 6.226 |
| LongConstant | 70.57% | 58.79% | 92.45% | 6.119 |
| LongDecay | 69.66% | 59.31% | 93.23% | 6.196 |
| Mixed | 71.16% | 58.07% | 89.84% | 6.129 |

平均 CBF-on success 最高的是 **Mixed**（71.16%）。Long-Constant、Long-Decay、Mixed 相对 v31 A2 round 8 的平均变化分别为 -0.0052、-0.0143、+0.0007；正向 context 数分别为 2/3、1/3、2/3。

本结果只回答长期仿真在线交互是否继续提高成功率、是否方向一致、D0 是否保持以及 CBF 依赖是否变化。所有 paired 95% CI 都是描述性结果，不是 gate。真机仍应保留 base+CBF 安全基线。

## English summary

v32 leaves the v31 A2 networks, CBF, reward, raw-action PPO, soft moving-KL, and corrective objective unchanged. Six per-context continuations compare constant and decayed learning rates from v31 round 8 through the unconditional round-24 policy. One base-initialized mixed policy trains for 24 rounds with an exactly balanced rotating F1/F2/F3 allocation. No candidate search, performance/KL gate, best-checkpoint selection, or outcome-dependent rerun is used.

The highest three-context mean CBF-on success belongs to **Mixed** at 71.16%. Context-wise changes, D0 retention, CBF-off behavior, intervention rate, correction norm, would-intervene fraction, nominal violations, returns, falls, and reached-riser metrics are published in the formal JSON files.

Protocol source commit: `6e2257f8cc109a3cee6bf563ac8014bf8a3fc926`. `SHA256SUMS` binds every compact publication file; `external_artifacts_manifest.json` binds the external checkpoints and raw telemetry.

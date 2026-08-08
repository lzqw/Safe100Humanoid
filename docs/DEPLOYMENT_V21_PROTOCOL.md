# v21: One-Deployment Local Matched-Success Preservation

## 当前证据状态 / Current evidence status

This document records the prospective v21 design. The first corrected,
prospectively frozen `L_dev` base-only sweep completed all 12 candidates and
stopped because none satisfied the calibration gates. A subsequent non-formal
base-only range pilot completed all 12 families: `C_dev`, `L1`, `L3`, `C1`,
`C2`, `C4`, and `C5` produced qualifiers, while `L_dev`, `L2`, `L4`, `L5`, and
`C3` did not. Pilot 2 then found qualifiers for all five failed families, but
the only `L2` qualifier sat exactly on two gates and was not treated as robust.
Pilot 3 isolated `L2` and found one qualifier with strict margin on every
scaled gate. The replacement 512-episode calibration then froze `L_dev`,
`C_dev`, and `L1`, but cleanly stopped at `L2`: all 12 candidates remained too
easy and none reached 100 falls. Pilot 4 then tested a deterministic bilateral
hip-yaw actuator zero offset. It produced a sharp difficulty cliff but no
qualifier: the matched-success point had only 53.57% lateral purity, while the
closest mechanism point was already below 70% success and still below 80%
purity. Pilot 5 returned to command-side yaw and found three adjacent high-end
near misses: one was a single fall short, one was 0.69 percentage points short
on purity, and one was three falls short. Pilot 6 doubled the sample size and
varied only persistent yaw bias; two mechanism-pure candidates missed the
scaled fall gate by two and three falls. Pilot 7 added a fixed low-amplitude
yaw-pulse carrier; candidates that reached the fall-count gate lost lateral
purity, and pooled purity fell below the pulse-free pilot. Pilot 8 then tested
strong pulse-free yaw bias from 0.65 to 0.95. None of its 12 candidates
qualified; the closest missed target purity by 0.75 percentage points and the
pooled purity was only 72.28%. This closes the tested command-magnitude axis.
Pilot 9 fixed a modest 0.34 yaw bias and varied only heading yaw authority from
0.32 down to 0.04. Three candidates exceeded 80% purity but had only 36--41
falls; candidates with at least 50 falls had at most 75.47% purity. Its pooled
purity was 72.91%, so the authority-loss axis is also closed. No adaptation,
monitor, or audit was started. Pilot 10 restored nominal feedback, removed raw
yaw bias, and varied only a bounded visual heading-reference bias from 0.35 to
0.75 radians. It reached at most 45 falls and 76.92% purity, with pooled purity
of 71.33%, so that mechanism is rejected. Pilot 11 then held command, feedback,
action, encoder, rise, and tread terms nominal while narrowing only the physical
stair half-width from 1.00 to 0.45 m. Three contiguous candidates at 1.00,
0.95, and 0.90 m passed every scaled gate; the full sweep had 95.17% pooled
target purity. The current boundary is a prospectively frozen replacement
512-episode base-only calibration. Its `L2` family finely sweeps 1.00--0.89 m
with entirely fresh candidate and evaluation randomness.

本文记录 v21 的前瞻性实验设计。第一轮修正并预先冻结的 `L_dev` base-only sweep
完成了全部 12 个候选，但因没有候选满足校准门槛而停止。随后一轮非正式 base-only
范围 pilot 完成了全部 12 个 family：`C_dev`、`L1`、`L3`、`C1`、`C2`、`C4`、
`C5` 找到了合格候选，`L_dev`、`L2`、`L4`、`L5`、`C3` 未找到。随后 pilot 2
为这 5 个 family 都找到了合格点，但 `L2` 的唯一合格点恰好压在两条门槛上，因此
不视为稳健。pilot 3 隔离 `L2` 后找到了一个在每条 scaled gate 上都有严格余量的
合格点。随后 replacement 512-episode 校准冻结了 `L_dev`、`C_dev` 和 `L1`，但在
`L2` clean fail-fast：12 个候选仍然都太容易，没有一个达到 100 次跌倒。没有启动
任何 adaptation、monitor 或 audit。pilot 4 随后测试了确定性的双侧 hip-yaw
actuator 零偏移；它产生了明显的难度 cliff，但没有合格点：matched-success 点只有
53.57% lateral purity，最接近目标机制的点成功率已低于 70%，purity 也仍低于 80%。
pilot 5 回到 command-side yaw 后，在高端找到三个相邻近失配点：一个只差 1 次跌倒，
一个 purity 只差 0.69 个百分点，另一个差 3 次跌倒。pilot 6 将样本数加倍并只改变
持续 yaw bias；两个机制纯的候选分别比 scaled fall gate 少 2 次和 3 次跌倒。pilot 7
加入固定低幅 yaw-pulse carrier 后，达到 fall-count 门槛的候选失去了 lateral purity，
合并 purity 也低于无 pulse 的 pilot。pilot 8 随后测试了 0.65--0.95 的强无 pulse
yaw bias，但 12 个候选仍无一合格；最接近者的目标 purity 差 0.75 个百分点，池化
purity 仅 72.28%，因此 command-magnitude 轴到此停止。没有启动 adaptation、monitor
或 audit。pilot 9 固定温和的 0.34 yaw bias，只把 heading yaw authority 从 0.32
降到 0.04。三个候选的 purity 超过 80%，但只有 36--41 次跌倒；达到至少 50 次跌倒
的候选最高 purity 仅 75.47%，池化 purity 为 72.91%，因此 authority-loss 轴也停止。
pilot 10 恢复正常反馈、移除 raw yaw bias，只把有界的视觉 heading-reference bias
从 0.35 扫到 0.75 radians，但最多只有 45 次跌倒与 76.92% purity，池化 purity 为
71.33%，因此该机制也被否决。pilot 11 随后保持 command、feedback、action、encoder、
rise 和 tread 为 nominal，只把物理 stair half-width 从 1.00 缩小到 0.45 m。
1.00、0.95、0.90 m 三个连续候选通过全部 scaled gate，整条 sweep 的目标池化 purity
为 95.17%。当前边界是前瞻冻结的 replacement 512-episode base-only calibration；
其中 `L2` 使用全新的候选与评估随机数，在 1.00--0.89 m 内进行细扫。

## 实验单位 / Experimental unit

One formal unit is exactly:

```text
one frozen deployment context
+ the common frozen base policy pi0
+ one pre-seeded online adaptation
+ one fresh paired evaluation
```

Each context has one deployment seed and one adaptation. A poor algorithmic
outcome is never rerun. Only a documented infrastructure failure may be retried
with the identical context, seed, commit, and checkpoint.

正式统计单位是 deployment context，不再是同一场景中的 adaptation seed。

## Context matrix

The two development contexts `L_dev` and `C_dev` are excluded from every formal
claim. The ten formal contexts are:

| Mode | Context | Primary frozen deployment shift |
| --- | --- | --- |
| Lateral | L1 | command delay + low-pass |
| Lateral | L2 | narrow physical stair lateral clearance |
| Lateral | L3 | lateral bias + lateral pulse |
| Lateral | L4 | weak centerline correction |
| Lateral | L5 | moderate mixed lateral shift |
| Contact | C1 | low foot friction |
| Contact | C2 | left-right action asymmetry |
| Contact | C3 | action gain + encoder bias |
| Contact | C4 | friction + command/dynamics mismatch |
| Contact | C5 | moderate mixed contact shift |

Each family freezes an ordered 12-candidate severity sweep. Perturbation
direction and normalized bias pattern stay fixed within a family. Calibration
uses only `pi0` and freezes the first candidate satisfying all predeclared
success, fall-count, and failure-purity gates.

The failed first `L_dev` sweep showed that command smoothing and weaker
centering could make nominally stronger candidates easier. Pilot 1 then showed
that the shared geometry/actuator carrier could dilute lateral purity in
`L_dev`, `L2`, and `L4`; `L5` was mechanism-pure but uniformly too hard; and
`C3` crossed a sharp action-gain/encoder-bias difficulty cliff. Pilot 2 holds
command dynamics and low-amplitude actuator terms fixed in the first three
families so their named lateral mechanism drives severity, lightens the whole
`L5` carrier, and finely brackets the observed `C3` cliff. Pilot 2 confirmed
those four revisions but left `L2` on a two-gate boundary. Pilot 3 therefore
removes the remaining geometry/action/encoder carrier from `L2` and strengthens
only its yaw-dominant command disturbance. Its one 128-episode qualifier did
not replicate at 512 episodes: the formal range produced only 64–85 falls.
Pilot 4 therefore tested a plant-side alternative: the same hidden zero offset
on both hip-yaw actuator channels. The sweep created enough failures but did
not isolate heading drift, so that family is rejected rather than narrowed.
Pilot 5 returned to the higher-purity command-side yaw mechanism and found a
narrow high-end near-qualifying cluster, but 128 episodes did not establish a
strict qualifier. Pilot 6 doubled the sample size and removed pulse
confounding. It produced two mechanism-pure candidates only two or three falls
short of the scaled gate. Pilot 7 retained persistent yaw bias as the only
swept axis and added one fixed yaw-only pulse carrier. The carrier supplied
failures but diluted purity, so it is rejected. Pilot 8 disabled every pulse
and extended the clean persistent-bias axis to 0.65--0.95, but stronger commands
increased failure volume without a robust purity intersection. Pilot 9 changed
mechanism instead of extending magnitude: it fixed yaw bias at 0.34, preserved
nominal lateral feedback, and swept only the heading correction saturation
limit from 0.32 down to 0.04. Mechanism-pure candidates remained too easy,
while difficult candidates lost purity. Pilot 10 therefore restored all
feedback gains and limits, set raw yaw bias to zero, and swept only a bounded
visual heading-reference bias from 0.35 to 0.75 radians. The true heading error
used by diagnostics and failure classification remained unchanged, but the
sweep reached neither the fall-count nor purity gate. Pilot 11 then held all
command, feedback, action, encoder, rise, and tread terms nominal and narrowed
only the physical stair half-width from 1.00 to 0.45 m. The same frozen width
drove tread geometry, root/foot edge-clearance telemetry, and the
geometry-derived classifier threshold. Candidates at 1.00, 0.95, and 0.90 m
all qualified, followed by a mechanism-pure over-difficult transition at
0.85 m. This was base-policy range evidence, not formal selection. The
replacement calibration preserves this mechanism, narrows the prospective
`L2` sweep to 1.00--0.89 m, uses 512 episodes per candidate, and replaces every
candidate and evaluation seed before any new outcome is observed.

## Algorithm

At round `k`, the frozen behavior policy is `pi_ref = pi_k`. The actor objective
for v21 is:

```text
L_actor = PPO(normal + failure precursor)
          + beta * KL(pi_ref || pi_theta) on matched-success states
```

Matched-success Actor advantages and entropy terms are excluded when
`beta > 0`; their only direct Actor term is the local preservation KL. The
critic continues to use every transition. No broad D0/global retention bank
and no new network are introduced.

The same-context control uses `beta = 0`, which restores the v20-style Actor
objective: PPO over normal, failure, and matched-success transitions.

## Frozen development and candidate selection

- Development grid: `beta in {0, 1, 4, 16}` on `L_dev` and `C_dev` only.
- Selection score: mean context-level `RR - RG`, with prospective tie-breaks.
- One selected beta is frozen for all ten formal contexts.
- Each of eight fixed rounds screens three fractions with `3 x 64` paired
  episodes, then tests the selected fraction in three independent `64`-episode
  confirmation blocks.
- Acceptance requires positive mean success delta, at least two positive
  blocks, mean fall delta at most 3 percentage points, finite KL below 0.01,
  and D0 retention.
- Zero retained updates is a valid result and leaves `pi_final = pi0`.

## Formal evaluation and claims

For every formal context, base, control, and v21 receive identical fresh test
conditions:

- 1,024 paired target episodes per policy;
- 256 paired D0 episodes per policy;
- per-context paired 95% intervals;
- repair rate, regression rate, and `RR - RG`;
- direct control-minus-base, v21-minus-base, and v21-minus-control comparisons.

For each mode, the formal v21 gate requires:

- mean target success delta strictly above zero;
- at least four of five deployment contexts positive;
- mean target fall delta at most 3 percentage points;
- mean D0 success delta at least -5 percentage points.

The cross-context 95% interval bootstraps the five deployment-context point
estimates. A positive lower bound is reported as strong evidence, not as a
protocol-validity condition.

## Unbiased curves and mechanism evidence

Every adaptation saves `pi0,...,pi8`. Only after training ends are all nine
checkpoints evaluated on the frozen, previously unseen 256-condition
`E_curve`; candidate-selection diagnostics never enter this curve.

Formal target evaluation captures mechanism telemetry inline from the actual
base/control/v21 rollouts and embeds each trace's exact same-rollout outcome.
No post-audit replay is used. Published compact curves normalize each trace to
101 episode-phase bins before aggregation, while trace-level tables retain the
outcome binding.

## Evidence boundaries

The failed revision-0 sweep and each base-only range pilot are immutable Git
evidence, but neither is a formal context-selection boundary. After range
feasibility is established, the replacement experiment has three immutable Git
boundaries with candidate, evaluation, adaptation, monitor, and bootstrap
randomness not used or proposed by a pilot:

1. revision 0: source, ranges, randomness, base checkpoint, and analysis plan,
   committed before base-only calibration;
2. revision 1: all 12 calibrated contexts, committed before development;
3. revision 2: development selection and one beta, committed before any formal
   adaptation.

Raw checkpoints and simulator rows remain external and are hash-manifested.
GitHub receives code, committed protocols, calibrated contexts, compact tables,
figures, manifests, and bilingual result reports.

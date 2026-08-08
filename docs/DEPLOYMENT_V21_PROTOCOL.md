# v21: One-Deployment Local Matched-Success Preservation

## 当前证据状态 / Current evidence status

This document records the prospective v21 design. The first corrected,
prospectively frozen `L_dev` base-only sweep completed all 12 candidates and
stopped because none satisfied the calibration gates. A subsequent non-formal
base-only range pilot completed all 12 families: `C_dev`, `L1`, `L3`, `C1`,
`C2`, `C4`, and `C5` produced qualifiers, while `L_dev`, `L2`, `L4`, `L5`, and
`C3` did not. Both immutable results are range evidence, not algorithm results;
no development or formal adaptation, monitor, or audit was started. The
current boundary is a second base-only pilot restricted to those five failed
families, explicitly excluded from formal context selection and using fresh
randomness.

本文记录 v21 的前瞻性实验设计。第一轮修正并预先冻结的 `L_dev` base-only sweep
完成了全部 12 个候选，但因没有候选满足校准门槛而停止。随后一轮非正式 base-only
范围 pilot 完成了全部 12 个 family：`C_dev`、`L1`、`L3`、`C1`、`C2`、`C4`、
`C5` 找到了合格候选，`L_dev`、`L2`、`L4`、`L5`、`C3` 未找到。两轮不可变结果
都只是范围证据，不是算法结果；尚未启动 development/formal adaptation、monitor 或
audit。当前边界是仅针对上述 5 个失败 family 的第二轮 base-only pilot；它明确不属于
formal context selection，并使用全新随机数。

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
| Lateral | L2 | yaw bias + yaw pulse |
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
`L5` carrier, and finely brackets the observed `C3` cliff. Pilot observations
may revise ranges only. Replacement formal calibration must freeze another
entirely new candidate and evaluation namespace before it starts.

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

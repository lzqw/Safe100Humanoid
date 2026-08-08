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
easy and none reached 100 falls. No adaptation, monitor, or audit was started.
The current boundary is a fourth non-formal base-only pilot that replaces the
falsified yaw-command family with a deterministic bilateral hip-yaw actuator
zero-offset family and uses entirely fresh randomness.

本文记录 v21 的前瞻性实验设计。第一轮修正并预先冻结的 `L_dev` base-only sweep
完成了全部 12 个候选，但因没有候选满足校准门槛而停止。随后一轮非正式 base-only
范围 pilot 完成了全部 12 个 family：`C_dev`、`L1`、`L3`、`C1`、`C2`、`C4`、
`C5` 找到了合格候选，`L_dev`、`L2`、`L4`、`L5`、`C3` 未找到。随后 pilot 2
为这 5 个 family 都找到了合格点，但 `L2` 的唯一合格点恰好压在两条门槛上，因此
不视为稳健。pilot 3 隔离 `L2` 后找到了一个在每条 scaled gate 上都有严格余量的
合格点。随后 replacement 512-episode 校准冻结了 `L_dev`、`C_dev` 和 `L1`，但在
`L2` clean fail-fast：12 个候选仍然都太容易，没有一个达到 100 次跌倒。没有启动
任何 adaptation、monitor 或 audit。当前边界是第四轮非正式 base-only pilot：它用
确定性的双侧 hip-yaw actuator 零偏移替换已被证伪的 yaw-command family，并使用
全新随机数。

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
| Lateral | L2 | bilateral hip-yaw actuator zero offset |
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
Pilot 4 therefore replaces the family rather than increasing command magnitude
again. It applies the same hidden zero offset to both hip-yaw actuator channels
while keeping geometry, command dynamics, encoder, and all other actuator
channels nominal. This plant-side heading disturbance is distinct from `L1`
command latency, `L3` lateral commands, `L4` centering, and the `L5` mixed
shift. It remains base-policy range evidence, not formal selection.

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

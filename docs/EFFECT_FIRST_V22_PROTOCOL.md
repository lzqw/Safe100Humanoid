# v22 Effect-First Development Protocol

Status: prospective development design. This is not a formal multi-context
generalization audit and does not alter or relabel any v17--v21 result.

## Question

Can one fixed, clearly attributable OOD deployment context produce a visible
online-adaptation effect of at least 3 percentage points without degrading the
target fall rate by more than 1 point or D0 success by more than 5 points?

The experiment intentionally trades breadth for effect discovery. It does not
run five contexts per mode, a control/v22 pair, an off-diagonal matrix, a macro
gate, or a cross-context bootstrap.

## Contexts and order

Only two one-dimensional context families are declared:

1. `L_effect`: lateral command bias plus lateral disturbance pulses. Stair
   geometry, foot friction, action gain/bias/delay, encoder bias, command delay
   and low-pass, yaw bias/pulses, stair width, and centerline-controller strength
   stay nominal and fixed.
2. `C_effect`: low foot friction only. Geometry, action and command dynamics,
   encoder bias, gait phase, contact sensing, and left/right response remain
   nominal and fixed.

"Nominal" is bound to the actual `DQHMED` configuration rather than the older
v21 family defaults: 9 steps, 0.13 m rises, 0.35 m treads, a 35 s episode,
the original `[0.04, 0.16]` s command-delay distribution, a 0.08 s command
low-pass time constant, action gain 1 with zero action delay/bias, zero encoder
bias, and foot friction 0.60 except for the contact family's single friction
axis. The centerline controller remains at gains `0.80/1.40` and command limits
`0.16/0.45`, while the CBF toe margin remains `0.08 m`. The scalar
`command_delay_s = 0.10` stored in the generic context record is the nominal
midpoint; v22 explicitly preserves and hashes the full nominal delay
distribution when applying the environment. Contact also retains the inactive
nominal pulse configuration; pulses remain disabled, so friction is its only
physical effect axis. Lateral explicitly zeros yaw-pulse magnitude when it
enables its lateral-only disturbance pulse.

The v22 context schema is version 2. Its family specification declares and
hashes the only allowed physical effect axes: `lateral_command_bias` plus
`lateral_disturbance_pulse` for `L_effect`, and `foot_friction` for `C_effect`.
Config-level regression tests apply both severity endpoints to a fresh
`DQHMED` configuration and verify every other physical carrier remains nominal.

Every replaced prospective boundary is retained in Git and linked by commit,
file hash, and an explicit supersession reason. A replacement is permitted only
before any base-policy episode has run; the freezer rejects partial lineage
arguments and records that the waiting queue was terminated before GPU use.

Execution is conditional and sequential. `L_effect` is calibrated, adapted, and
tested first. Contact calibration and adaptation are prohibited unless the
lateral fresh final test passes the development gate. If lateral fails, the
revision stops and preserves that negative result. Additional fresh contexts
are out of scope unless both modes pass.

## Base-only calibration

Each family declares an ordered, previously unseen candidate list. Every
candidate receives 512 base-policy episodes in batches of 128. The first
candidate satisfying every condition is frozen:

- base success is within `[65%, 75%]`, inclusive;
- at least 100 failures occur;
- at least 85% of failures have the declared target type.

Only `pi0` is evaluated. Adapted-policy outcomes cannot affect context
construction or selection. If no candidate qualifies, the phase stops before
adaptation and records a calibration-negative result.

## Learning core

v22 reuses the validated v20/v21 beta-zero core:

- one Actor and one privileged critic;
- runtime CBF execution, while PPO stores the raw policy action;
- observable five-column zero expansion with exact `pi0` preservation;
- frozen legacy first-layer input columns;
- failure/success restart banks and exact paired restarts;
- two independent rollout batches per round;
- `40/12/12` normal/failure/matched-success environment slots;
- fixed eight-round budget;
- candidate fractions `{0.5, 1.0, 1.5}`.

Matched-success preservation is fixed to `beta = 0`. v22 adds no KL
preservation objective, extra critic, risk head, cost critic, macro gate, or
parallel control branch.

## Per-round candidate gate

Each fraction is screened on the same 64 paired target conditions. The finite
candidate with the largest success point-estimate improvement is selected,
with lower fall change and the smaller fraction as tie breaks. It is then
confirmed once on 128 completely fresh paired target conditions.

The target confirmation requires only:

- success delta strictly above zero;
- fall delta at most `+3 pp`.

The candidate must also retain D0 success within `-5 pp` of `pi0` on 128 paired
D0 episodes. Confidence intervals, multi-block voting, McNemar tests, and
repair/regression gates do not decide candidate acceptance. Non-finite model
parameters remain an invalid numerical state, not a statistical gate.

## Fixed validation monitor and best-so-far deployment

Before training, v22 freezes 256 target conditions `E_val`, disjoint from PPO,
restart-bank discovery, candidate screening/confirmation, and final testing.
`pi0` and every D0-safe accepted checkpoint are evaluated on exactly this same
monitor.

An eligible checkpoint must satisfy:

`Fall_val(pi_k) <= Fall_val(pi0) + 2 pp`.

Among eligible checkpoints, v22 deploys the one with highest validation success;
ties prefer lower fall rate and then the earlier round. `pi0` is always an
eligible round-zero candidate. Monitor measurements never alter PPO data,
candidate acceptance, the fixed eight-round budget, or later updates. The
deployed checkpoint is `best_so_far.pt`, not the last accepted checkpoint by
default.

## Fresh final test

After training and best-so-far selection, a previously inaccessible final set
is generated:

- 512 paired target episodes for `pi0` and `pi_best`;
- 256 paired D0 episodes for `pi0` and `pi_best`.

The four primary values are target success and fall rates for base and best.
Failure-to-success repairs and success-to-failure regressions are also reported.
A 2,000-resample paired bootstrap interval may be reported descriptively, but
it is not a development gate.

The development gate passes only when:

- target success improves by at least `+3 pp`;
- target fall increases by at most `+1 pp` (zero increase is also reported as
  the stricter safety interpretation);
- D0 success decreases by no more than `5 pp`.

No poor-outcome rerun is allowed. Only a documented infrastructure retry with
the identical context, checkpoint, seed, source commit, and protocol is valid.

## Evidence and figures

The compact publication contains the frozen protocol/context, calibration
summary, training/validation summary, paired final-test metrics, final aggregate
JSON, SHA-256 manifest, and exactly four figure categories in PNG and PDF:

1. validation success versus accepted checkpoint round;
2. fresh base-versus-best target success and falls;
3. repair versus regression;
4. same-rollout failure-specific telemetry (`|e_y|` and `|e_psi|` for lateral,
   maximum foot slip for contact when contact is eligible to run).

Large checkpoints and raw simulator rollouts remain in the external artifact
store and are bound by SHA-256 rather than committed to Git.

## 中文摘要

v22 只回答一个开发问题：在一个原因清晰、预先冻结的 OOD 场景中，一次在线
微调能否带来至少 `+3 pp` 的独立测试提升。先运行纯 lateral bias/pulse；只有
它通过后才运行纯 low-friction contact。每个场景只做一次 adaptation，不再做
control/v22 双分支或多 context 大审计。

场景仅用基础策略校准，目标成功率为 65--75%，至少 100 次失败，目标失败纯度
至少 85%。训练保持 beta=0 的 v20/v21 PPO/CBF 核心。新增固定 256 episode
validation monitor，从所有 D0-safe accepted checkpoints 中选择成功率最高且 fall
不超过基础策略 `+2 pp` 的 best-so-far checkpoint。最后使用完全 fresh 的 512
target pairs 和 256 D0 pairs 比较 base 与 best。正式开发门槛是 target success
至少 `+3 pp`、target fall 最多 `+1 pp`、D0 success 最多下降 `5 pp`。如果
lateral 未达到门槛，本 revision 立即保存负结果并停止，不启动 contact。

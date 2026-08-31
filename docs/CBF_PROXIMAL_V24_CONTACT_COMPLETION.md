# v24 Contact Completion — Prospective Protocol

## Scope / 范围

v24 is one independent follow-up to the frozen v23 lateral experiment. v23
remains a negative result and is never modified, rerun, recomputed, or
reinterpreted. v24 tests one pure low-foot-friction contact context. The two
experiments do not form a joint gate.

v24 是已冻结 v23 lateral 负结果之后的独立补充实验。它不修改、不重跑、不重新
计算、也不重新解释 v23。v24 只测试一个纯低足底摩擦 contact context；两者不构成
联合 gate。

## Two prospective boundaries / 两次前瞻冻结

Before the first v24 simulator episode, the implementation, ordered candidate
grid, base-only selection rule, all execution seeds, and pre-calibration
protocol are committed. Calibration evaluates candidates from light to severe
with the frozen base actor only.

The first qualifying context is then committed with its compact calibration
evidence. A second protocol commits its friction, context and parameter hashes,
new adaptation seed, final target/D0 seeds, implementation commit, and complete
formal procedure before any adapted-policy outcome is observed. Training and
final evaluation are each executed once; outcome-directed reselection or rerun
is forbidden.

第一次 v24 仿真之前，代码、由轻到重的候选网格、base-only 选择规则、全部执行
seed 和预校准协议均先提交。第一个合格 context 及精简校准证据随后提交；在观察
任何 adapted-policy 结果之前，再次冻结摩擦、context/hash、全新 adaptation seed、
target/D0 seed、代码 commit 与完整正式协议。禁止结果导向的重选或重跑。

## Pure contact context / 纯 contact 场景

The target belongs to the frozen `C_effect` / `pure_low_foot_friction` family.
Only `foot_friction` varies. Stair geometry, commands and command dynamics,
actuation, sensing, gait phase, left/right response, navigation feedback, and
disturbance pulses remain nominal.

Each candidate receives 512 deterministic base-policy episodes in four fresh
batches of 128. The first candidate satisfying every condition is selected:

- base success in the closed interval `[65%, 75%]`;
- at least 100 falls;
- at least 85% contact/slip purity over all non-success episodes;
- at least 85% contact/slip purity over falls.

The all-non-success denominator is deliberately stricter when timeouts occur.
Ordinary CBF intervention is telemetry, not a failure label.

## Algorithm reused without modification / 算法原样复用

v24 delegates its optimizer configuration and round implementation to the
frozen v23 code path. It uses one original 405-D actor, one 838-D privileged
critic, runtime CBF, raw policy-action PPO storage, and a moving round-start
reference `pi_k`. The environment executes the filtered action; PPO retains the
raw sampled action and behavior log probability.

| Item | Frozen value |
| --- | ---: |
| rounds / environments / steps | `8 / 64 / 1024` |
| actor / critic learning rate | `5e-6 / 1e-4` |
| PPO clip | `0.05` |
| maximum actor / critic epochs | `2 / 2` |
| minibatches | `4` |
| moving KL beta | `0.5` |
| target / hard KL | `0.003 / 0.01` |
| maximum gradient norm | `0.5` |
| base std scale / clamp | `0.35 / [0.05, 0.25]` |
| log std / entropy | frozen / `0.0` |
| gamma / GAE lambda | `0.99 / 0.95` |

All eight rounds run irrespective of success, return, fall, or intervention
telemetry. A rollback is allowed only for non-finite state, forward KL above
`0.01`, action/behavior routing corruption, or optimizer-state corruption. It
restores actor, critic, and both optimizers. The final actor is unconditionally
round 8.

No extra observations, specialist reward, failure/matched-success bank, state
restart, candidate line search, performance gate, best checkpoint, additional
critic, or risk head is introduced.

## One fresh paired audit / 唯一一次 fresh paired 审计

After round 8, deterministic policy means are compared with runtime CBF on:

- 512 paired target episodes;
- 256 paired D0 episodes.

Base and final actors receive identical initial conditions and reset randomness.
The report includes success, fall, return, reached riser, slip, contact mismatch,
CBF interventions per riser, correction norm, recovery takeover, repairs,
regressions, and per-round moving KL. Paired bootstrap intervals are report-only.

The post hoc development gate is:

```text
target success delta >= +3 percentage points
target fall delta    <= +1 percentage point
D0 success delta     >= -5 percentage points
```

It never affects training, rollback, stopping, or checkpoint selection.

## Outputs / 输出

The minimal v24 package contains `README.md`, `protocol.json`,
`training_summary.json`, `round_metrics.csv`, `final_test.json`,
`paired_episode_metrics.csv`, `verification.json`, and `SHA256SUMS`. Exactly
three figure categories are produced: `round_curve`, `base_vs_final`, and
`repairs_vs_regressions` (PNG and PDF renderings do not create new categories).

The combined package contains the byte-frozen v23 lateral row and the single
v24 contact row. It includes no additional context, adaptation seed,
off-diagonal audit, macro result, candidate ablation, or large figure suite.

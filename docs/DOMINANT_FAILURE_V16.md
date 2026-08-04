# Dominant-Failure Brief PPO (v16)

v16 is the predeclared Branch-B follow-up to Failure-Focused Brief PPO v15.
It changes only hard-case bank admission: the bank accepts target falls from
the frozen dominant failure class. The v15 late-state selector, priority,
window, thresholds, network, reward, PPO settings, screening gates, and
runtime CBF are unchanged.

## Frozen failure classification

The classifier used all 1,536 DQH-Medium online-start baseline episodes from
the completed v15 final audit. It did not evaluate a v16-adapted policy. The
265 baseline falls were assigned to one of three mutually exclusive classes:

| Frozen class | Falls | Fraction of falls |
| :--- | ---: | ---: |
| `lateral_heading_drift` | 146 | 55.094% |
| `non_lateral_high_cbf_demand` | 72 | 27.170% |
| `non_lateral_balance_or_phase` | 47 | 17.736% |

`lateral_heading_drift` was the unique largest class and was frozen before
v16 training. A fall enters that class if a root or foot breaches a side
edge, maximum centerline error reaches two thirds of the 1.2 m stair
half-width (0.8 m), or maximum heading error reaches 90 degrees. Remaining
falls are split at maximum CBF correction 0.5.

The classification JSON SHA-256 is
`a98895ba4c7f8bdfc700888f36f6ab92833d401b20442284c585eb7f73623552`.

```bash
bash experiments/scripts/run_failure_classification_v16.sh
```

## Filter-only bank refinement

Each adaptation seed rebuilds its target late-failure bank from scratch. The
failure-discovery policy parameters are restored after every discovery
rollout, and discovery is bounded at eight rollouts. Only
`lateral_heading_drift` falls may add a state. All admitted states still use
the exact v15 selector: 50--150 steps before an actual target fall, in the
latter half of the staircase, with the same priority weights and exclusion of
states followed by a normal riser crossing.

Everything else remains the v15 protocol:

- 64 environments and 1,024 steps in each of five rounds;
- 54 persistent normal starts and 10 persistent hard starts;
- hard-case actor weight 0.75;
- one scalar task reward, terminal fall penalty -2, and a second -2
  redistributed over at most 100 preceding transitions with decay 0.97;
- online dual CBF reward exactly zero;
- actor LR `5e-6`, critic LR `1e-4`, one PPO epoch, clip 0.05, target KL
  0.003, and hard KL gate 0.01;
- paired candidate fractions 0.5, 1.0, and 1.5 on 128 target episodes;
- D0 retention checked every two rounds with a five-point tolerance;
- runtime CBF enabled for discovery, training, screening, and audit.

```bash
for seed in 42 142 242; do
  SAFE100_SEED="$seed" bash experiments/scripts/run_dominant_failure_v16.sh
done
```

## Formal training integrity

| Adaptation seed | Five-round accepted sequence | Policy changes | Maximum attempted minibatch KL | Discovery rollouts | Final bank window |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 42 | 1.0, 1.0, 0.5, 0.5, 1.0 | 5/5 | 0.001833056 | 3 | 50--75 steps; riser 11 |
| 142 | rollback, 1.5, 0.5, 1.5, 0.5 | 4/5 | 0.002841832 | 2 | 50--65 steps; riser 11 |
| 242 | 1.0, rollback, 1.0, rollback, rollback | 2/5 | 0.002671600 | 1 | 50--73 steps; riser 11 |

Across 15 rounds, 11 candidates changed the accepted policy. Every final bank
contained 256 `lateral_heading_drift` entries and passed dominant-class purity
and successful-crossing exclusion checks. The hard transition fraction was
exactly `10/64 = 0.15625`; raw-policy storage and executed-safe-action routing
maximum errors were zero; redistributed penalty was exactly twice the fall
event count; all six periodic D0 checks passed; and every attempted update
remained below KL 0.01.

The per-round screens and small final evaluations in the training summaries
are diagnostics, not final evidence.

## Independent final audit

The final audit used fresh seed 1,700,000, disjoint from calibration,
adaptation, and the v15 audit. It paired the online-start and final policies
on identical initial episodes for each adaptation seed: 512 target, 256 D0,
and 256 report-only neighbor episodes per seed. The published CSV contains
3,072 paired rows. Intervals use 10,000 hierarchical paired-bootstrap samples
over adaptation seeds and episodes.

Success and fall deltas below are percentage points.

| Domain | Baseline success | Final success | Paired success delta (95% CI) | Baseline fall | Final fall | Paired fall delta (95% CI) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQH-Medium target | 81.966% | 82.682% | +0.716 pp [-2.279, +3.841] | 18.034% | 17.318% | -0.716 pp [-3.776, +2.279] |
| D0 retention | 90.885% | 91.016% | +0.130 pp [-2.604, +3.125] | 2.474% | 1.302% | -1.172 pp [-2.995, +0.391] |
| DQNH-Medium neighbor | 80.859% | 78.255% | -2.604 pp [-5.990, +1.042] | 19.141% | 21.745% | +2.604 pp [-0.911, +6.120] |

| Predeclared gate | Observed interval | Result |
| :--- | :--- | :---: |
| `LCB95[delta target success] > 0` | `[-2.279, +3.841]` pp | **Fail** |
| `LCB95[delta D0 success + 5 pp] >= 0` | `[+2.396, +8.125]` pp | Pass |
| `UCB95[delta target fall] <= 3 pp` | `[-3.776, +2.279]` pp | Pass |

Target CBF intervention per riser changed by `-0.0249`, with 95% interval
`[-0.1121, +0.0510]`. It and the neighbor domain are report-only.

```bash
bash experiments/scripts/run_final_audit_v16.sh
```

## Paired failure-mode report

The audit classified target outcomes after evaluation; these labels were
report-only and never used as a promotion gate.

| Baseline outcome \ Final outcome | Lateral/heading | Balance/phase | High CBF | Success |
| :--- | ---: | ---: | ---: | ---: |
| Lateral/heading | 19 | 17 | 11 | 114 |
| Balance/phase | 18 | 12 | 13 | 12 |
| High CBF | 16 | 19 | 16 | 10 |
| Success | 97 | 11 | 17 | 1,134 |

Baseline-to-final counts changed from 161 to 150 lateral/heading failures and
from 1,259 to 1,270 successes. The 114 lateral-to-success transitions show
that the focused bank can affect the intended mode, but 97 baseline successes
also became lateral/heading failures. This bidirectional movement and the
between-seed variation leave the aggregate improvement statistically
unresolved.

## Result and next decision

v16 does **not** establish a statistically supported target improvement, so
it is not promoted. The predeclared decision is **B2**: do not add another
mechanism yet; compare the paired failure-type transitions before choosing a
new design.

For context, v15 reported `+0.781` pp with CI `[-1.628, +3.190]`, while v16
reported `+0.716` pp with CI `[-2.279, +3.841]`. These use different audit
samples and are not a paired v15-versus-v16 comparison. The supported result
is narrower: dominant-failure filtering was implemented and audited with
protocol integrity, but it did not produce statistically sufficient evidence
of improvement under this fixed hidden simulation context. It does not imply
real-robot improvement or operation without the runtime CBF.

The compact evidence package is published under
[`results/online/dominant_failure_v16/`](../results/online/dominant_failure_v16/README.md).

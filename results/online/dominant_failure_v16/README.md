# Dominant-Failure Brief PPO v16 formal evidence

This directory publishes the frozen v15-baseline failure classification,
three formal filter-only adaptation runs, their audited checkpoints, and the
fresh paired final audit for Dominant-Failure Brief PPO v16.

**Result:** DQH-Medium target success changed from `81.966%` to `82.682%`, a
paired delta of `+0.716` percentage points with hierarchical paired 95% CI
`[-2.279, +3.841]`. Because the lower bound is not above zero, the predeclared
target-improvement gate **failed**. v16 is not promoted and does not establish
a statistically supported improvement.

The target-fall and D0-retention gates passed. The post-v16 decision is
**B2**: do not add mechanisms yet; inspect paired failure-type transitions
before another design. Full method and reproduction details are in
[`docs/DOMINANT_FAILURE_V16.md`](../../../docs/DOMINANT_FAILURE_V16.md).

## Frozen Branch-B classification

Classification used all 1,536 online-start baseline target episodes from the
completed v15 audit and no v16-adapted evaluation. Among 265 falls:

- `lateral_heading_drift`: 146 (`55.094%`);
- `non_lateral_high_cbf_demand`: 72 (`27.170%`);
- `non_lateral_balance_or_phase`: 47 (`17.736%`).

The unique largest class, `lateral_heading_drift`, was frozen for bank
admission. Its classification JSON SHA-256 is
`a98895ba4c7f8bdfc700888f36f6ab92833d401b20442284c585eb7f73623552`.

## Minimal v16 change and training integrity

v16 changes only which failures may enter the late-state bank. The selector
priority, window, thresholds, network, scalar reward, PPO hyperparameters,
acceptance gates, and runtime CBF are the exact v15 design.

| Adaptation seed | Accepted sequence | Changes | Maximum KL | Final bank |
| ---: | :--- | ---: | ---: | :--- |
| 42 | 1.0, 1.0, 0.5, 0.5, 1.0 | 5/5 | 0.001833056 | 256 lateral; 50--75 steps |
| 142 | rollback, 1.5, 0.5, 1.5, 0.5 | 4/5 | 0.002841832 | 256 lateral; 50--65 steps |
| 242 | 1.0, rollback, 1.0, rollback, rollback | 2/5 | 0.002671600 | 256 lateral; 50--73 steps |

All banks passed class-purity and successful-crossing exclusion checks. All
15 rounds had exact hard transition fraction `0.15625`, zero online dual CBF
reward, exact fall-penalty accounting, zero raw-action storage and safe-action
routing errors, and KL below 0.01. All six periodic D0 checks passed.

## Fresh paired audit

The audit used disjoint seed 1,700,000 and 10,000 hierarchical paired
bootstrap samples. It evaluated 512 target, 256 D0, and 256 report-only
neighbor episodes per adaptation seed, producing 3,072 paired rows.

| Domain | Baseline success | Final success | Paired delta (95% CI) |
| :--- | ---: | ---: | ---: |
| DQH-Medium target | 81.966% | 82.682% | +0.716 pp [-2.279, +3.841] |
| D0 retention | 90.885% | 91.016% | +0.130 pp [-2.604, +3.125] |
| DQNH-Medium neighbor | 80.859% | 78.255% | -2.604 pp [-5.990, +1.042] |

Target fall changed by `-0.716` pp with CI `[-3.776, +2.279]`, passing the
three-point safety cap. The D0 five-point-margin noninferiority interval was
`[+2.396, +8.125]` pp and passed. Target CBF intervention per riser changed by
`-0.0249`, CI `[-0.1121, +0.0510]`; it is report-only.

The target report counted 161 baseline versus 150 final lateral/heading
failures, and 1,259 baseline versus 1,270 final successes. It also observed
114 lateral-to-success and 97 success-to-lateral transitions. This is a useful
mechanistic signal, but not sufficient statistical evidence for promotion.

## Evidence files

- `key_results.json`: compact machine-readable result and integrity index.
- `classification/failure_classification.json`: frozen classification,
  provenance, thresholds, counts, and hashes.
- `classification/classified_baseline_falls.csv`: all 265 classified v15
  baseline falls.
- `training/train_seed*_summary.json`: complete five-round records.
- `checkpoints/train_seed*_accepted_final.pt`: the three audited policies.
- `final_audit/final_audit_summary.json`: protocol, confidence intervals,
  gates, failure transitions, and B2 decision.
- `final_audit/paired_episode_metrics.csv`: all 3,072 compact paired rows.
- `MANIFEST.sha256`: SHA-256 digest for every published evidence artifact.

The evidence supports only one fixed, training-unseen, actor-hidden composite
simulation context with runtime CBF enabled. It does not establish exhaustive
sim-to-real coverage, guaranteed real-robot improvement, or filter-free
operation.

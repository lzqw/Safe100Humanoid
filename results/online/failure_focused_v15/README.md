# Failure-Focused Brief PPO v15 formal evidence

This directory publishes the frozen deployment context, three formal
adaptation runs, and the independent final audit for Failure-Focused Brief PPO
v15. Runtime CBF is enabled in every calibration, training, screening, and
audit episode.

**Result:** target DQH-Medium success increased by `+0.781` percentage points,
but its hierarchical paired 95% confidence interval is
`[-1.628, +3.190]` points. The predeclared target-improvement gate therefore
**failed**. v15 does not establish a statistically supported target success
improvement.

The target-fall and D0-retention gates passed. The predeclared decision tree
therefore selects **Branch B**: classify target falls and rebuild the bank
around only the dominant failure type. The method and reproduction commands
are documented in
[`docs/FAILURE_FOCUSED_V15.md`](../../../docs/FAILURE_FOCUSED_V15.md).

## Frozen context calibration

The ordered calibration search considered seeds 1000--1019 and selected the
first context whose base-policy success was between 75% and 85%. Seed 1000
qualified immediately on 128 episodes:

- base success: `80.469%` (`103/128`);
- base fall: `19.531%` (`25/128`);
- intervention/riser: `1.1951`;
- mean reached riser: `10.7656`.

Selection used only base-policy success. No adapted policy was evaluated
during calibration. The frozen parameter hash is
`4eafa5b94792f8b709e2da98ed3638a285790497a1e876e937373d55ce8d75bf`;
the frozen context file hash is
`51cf8ad7b1cd353c285af8b62636000cfc36b0b9e8d379ac386b1a97e6ce06f4`.

## Training integrity

| Adaptation seed | Five-round accepted sequence | Policy changes | Maximum attempted minibatch KL | Final bank window |
| ---: | :--- | ---: | ---: | :--- |
| 42 | 1.0, 1.5, 1.0, 1.5, 0.5 | 5/5 | 0.002292554 | 50--140 steps; risers 10--11 |
| 142 | 0.5, 1.0, 1.0, rollback, 1.0 | 4/5 | 0.003209218 | 55--69 steps; riser 11 |
| 242 | 1.0, 1.0, 1.5, 0.5, 1.0 | 5/5 | 0.002507764 | 50--63 steps; riser 11 |

Across all 15 rounds:

- online dual CBF reward was exactly zero;
- 10/64 rollout slots were persistent hard starts, giving an exact hard
  transition fraction of `0.15625`, with actor weight `0.75`;
- every final bank contained 256 target-only late-failure states and passed
  the successful-crossing exclusion check;
- raw-policy storage and executed-safe-action routing maximum errors were `0`;
- redistributed fall penalty was exactly `2 * fall_event_count`, preserving
  the specified total penalty of `-4` per fall;
- all six periodic D0 checks passed and no D0 rollback was required;
- every attempted update remained below the hard KL gate of `0.01`.

The 128-episode per-round gates and small final evaluations inside the
training summaries are diagnostics only. They are not used as final evidence.

## Independent final audit

The audit paired the online-start and final policies on identical initial
episodes for each adaptation seed. It used 512 target, 256 D0, and 256
report-only neighbor episodes per seed, for 3,072 paired rows in total.
Intervals use 10,000 hierarchical paired-bootstrap samples over adaptation
seeds and episodes.

Success and fall deltas below are percentage points.

| Domain | Baseline success | Final success | Paired success delta (95% CI) | Baseline fall | Final fall | Paired fall delta (95% CI) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQH-Medium target | 82.747% | 83.529% | +0.781 pp [-1.628, +3.190] | 17.253% | 16.471% | -0.781 pp [-3.255, +1.563] |
| D0 retention | 91.797% | 92.057% | +0.260 pp [-1.823, +2.474] | 1.302% | 1.432% | +0.130 pp [-1.563, +1.823] |
| DQNH-Medium neighbor | 81.771% | 81.510% | -0.260 pp [-6.380, +5.859] | 18.229% | 18.359% | +0.130 pp [-5.990, +6.380] |

| Predeclared gate | Observed interval | Result |
| :--- | :--- | :---: |
| `LCB95[delta target success] > 0` | `[-1.628, +3.190]` pp | **Fail** |
| `LCB95[delta D0 success + 5 pp] >= 0` | `[+3.177, +7.474]` pp | Pass |
| `UCB95[delta target fall] <= 3 pp` | `[-3.255, +1.563]` pp | Pass |

Target CBF intervention/riser changed by `-0.0189`, with 95% interval
`[-0.0815, +0.0397]`; this is report-only and is not a promotion gate.
DQNH-Medium is also report-only.

## Evidence files

- `context/frozen_dqh_medium_context.json`: immutable hidden target and
  report-only neighbor contexts.
- `context/calibration_summary.json`: ordered-search attestation and hashes.
- `context/base_evaluation.{json,csv}`: the qualifying base-policy evaluation.
- `key_training_results.json`: compact machine-readable result and integrity
  index.
- `training/train_seed*_summary.json`: complete five-round training records.
- `checkpoints/train_seed*_accepted_final.pt`: the three policies audited.
- `final_audit/final_audit_summary.json`: protocol, per-seed metrics,
  confidence intervals, gates, and Branch-B decision.
- `final_audit/paired_episode_metrics.csv`: all 3,072 compact paired rows.
- `MANIFEST.sha256`: SHA-256 digest for every published evidence artifact.

The evidence supports only one fixed, training-unseen, actor-hidden composite
simulation context. It does not establish exhaustive sim-to-real coverage,
guaranteed real-robot improvement, or operation without the runtime CBF.

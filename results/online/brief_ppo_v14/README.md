# Brief PPO v14 formal evidence

This directory contains the three formal training runs and the independent
final audit for CBF-Guided Brief PPO Refinement v14. Runtime CBF is enabled in
every training and evaluation episode.

**Result:** the target DQH success point estimate increased by `+0.586` pp,
but the paired 95% interval crosses zero. The preregistered target-improvement
claim gate therefore **did not pass**, and no statistically supported DQH
improvement is claimed.

The implementation entered this branch in commit `fe1248a`. The exact method
and reproduction commands are documented in
[`docs/BRIEF_PPO_V14.md`](../../../docs/BRIEF_PPO_V14.md).

## Training integrity

| Training seed | Five-round accepted sequence | Maximum pre-update minibatch KL | Hard transitions | Raw-policy storage error | Safe-action routing error |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 42 | 0.5, 1.0, rollback, 1.0, 1.0 | 0.000631034 | 20.3125% | 0 | 0 |
| 142 | 0.5, 1.0, 0.5, 0.5, rollback | 0.000625066 | 20.3125% | 0 | 0 |
| 242 | 0.5, 0.5, rollback, 1.0, 0.5 | 0.000617760 | 20.3125% | 0 | 0 |

All attempted updates remained below the target-KL threshold of `0.005` and
the hard precheck limit of `0.01`. The fixed 13/64 hard slots maintained an
exact `0.203125` transition fraction in every round. No periodic D0 check
required a D0 rollback.

The per-round 128-episode target gates and small final evaluations in
`training/` are training diagnostics only. They are not used as final evidence.

## Independent final audit

The audit used three independent adaptation seeds with paired online-start and
final-policy episodes:

| Domain | Episodes per training seed | Total paired episodes |
| :--- | ---: | ---: |
| DQH | 512 | 1,536 |
| D0 | 256 | 768 |
| DQNH | 256 | 768 |

The reported intervals use 10,000 hierarchical paired-bootstrap samples,
resampling training seeds and then episodes. Success and fall deltas are in
percentage points.

| Domain | Baseline success | Final success | Paired success delta (95% CI) | Baseline fall | Final fall | Paired fall delta (95% CI) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQH | 89.388% | 89.974% | +0.586 pp [-1.174, +2.474] | 10.612% | 9.961% | -0.651 pp [-2.604, +1.107] |
| D0 | 89.583% | 93.229% | +3.646 pp [+0.521, +7.292] | 1.953% | 1.172% | -0.781 pp [-1.563, -0.130] |
| DQNH | 90.885% | 88.932% | -1.953 pp [-4.427, +0.521] | 9.115% | 11.068% | +1.953 pp [-0.521, +4.427] |

| Domain | Baseline interventions/riser | Final interventions/riser | Paired delta (95% CI) |
| :--- | ---: | ---: | ---: |
| DQH | 0.7005 | 0.7457 | +0.0452 [-0.0350, +0.1486] |
| D0 | 0.5963 | 0.5611 | -0.0352 [-0.1903, +0.1098] |
| DQNH | 0.8001 | 0.8304 | +0.0303 [-0.0800, +0.1425] |

The required criterion was
`LCB95[SR_DQH(final) - SR_DQH(online-start)] > 0`. Its observed lower bound is
`-1.174` pp, so the criterion failed. D0 success and fall improved under this
audit; DQH and DQNH changes remain statistically inconclusive at 95%.

## Evidence files

- `key_training_results.json`: compact machine-readable training integrity
  and diagnostic index.
- `training/train_seed*_summary.json`: complete five-round training summaries.
- `checkpoints/train_seed*_accepted_final.pt`: the three final actor/critic
  snapshots used by the audit.
- `final_audit/final_audit_summary.json`: protocol, checkpoint hashes,
  per-seed metrics, hierarchical confidence intervals, and claim gate.
- `final_audit/paired_episode_metrics.csv`: compact paired episode evidence
  for independent recomputation.
- `MANIFEST.sha256`: SHA-256 digest for every published evidence artifact.

Run the formal workflow with:

```bash
SAFE100_SEED=42 bash experiments/scripts/run_brief_ppo_v14.sh
bash experiments/scripts/run_final_audit_v14.sh
```

The evidence supports only the fixed unknown deployment context represented by
this protocol. It is not an exhaustive sim-to-real result, and it does not
support operation without the runtime CBF.

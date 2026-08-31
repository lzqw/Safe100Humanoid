# v25 Revision-7 — Terminal Calibration Result

The prospectively frozen base-only calibration evaluated all `25`
ordered actuator-under-response candidates and found no first qualifier. Per the
frozen protocol, formal adaptation and final evaluation were not started. No grid,
threshold, seed, policy outcome, or candidate order was changed, and no
outcome-directed rerun was performed.

预先冻结的基础策略配对校准已经按从轻到重的顺序评估全部 `25` 个候选，
没有候选同时进入对齐、可救援和成功率走廊。因此依据冻结协议停止：没有启动适应训练或
终评，也没有修改网格、阈值、seed 或根据结果重跑。

| Candidate | Gain | CBF-off success | CBF-on success | Alignment | Rescue |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.98 | 91.406% | 93.359% | 97.727% | 86.364% |
| 1 | 0.96 | 94.922% | 93.359% | 96.154% | 96.154% |
| 2 | 0.94 | 92.578% | 92.383% | 97.368% | 81.579% |
| 3 | 0.92 | 94.141% | 94.922% | 93.333% | 90.000% |
| 4 | 0.9 | 91.406% | 93.750% | 90.909% | 93.182% |
| 5 | 0.88 | 90.039% | 93.359% | 90.196% | 90.196% |
| 6 | 0.86 | 90.234% | 85.938% | 92.000% | 78.000% |
| 7 | 0.84 | 76.953% | 79.492% | 89.831% | 74.576% |
| 8 | 0.82 | 61.133% | 65.430% | 96.985% | 61.809% |
| 9 | 0.8 | 49.219% | 53.125% | 96.923% | 50.769% |
| 10 | 0.78 | 41.602% | 31.445% | 95.318% | 23.746% |
| 11 | 0.76 | 15.039% | 5.469% | 91.724% | 3.908% |
| 12 | 0.74 | 0.391% | 0.781% | 90.588% | 0.784% |
| 13 | 0.72 | 0.000% | 0.000% | 89.062% | 0.000% |
| 14 | 0.7 | 0.000% | 0.000% | 86.523% | 0.000% |
| 15 | 0.68 | 0.000% | 0.000% | 85.938% | 0.000% |
| 16 | 0.66 | 0.000% | 0.000% | 91.602% | 0.000% |
| 17 | 0.64 | 0.000% | 0.000% | 92.773% | 0.000% |
| 18 | 0.62 | 0.000% | 0.000% | 95.312% | 0.000% |
| 19 | 0.6 | 0.000% | 0.000% | 95.703% | 0.000% |
| 20 | 0.58 | 0.000% | 0.000% | 96.484% | 0.000% |
| 21 | 0.56 | 0.000% | 0.000% | 98.047% | 0.000% |
| 22 | 0.54 | 0.000% | 0.000% | 96.875% | 0.000% |
| 23 | 0.52 | 0.000% | 0.000% | 98.047% | 0.000% |
| 24 | 0.5 | 0.000% | 0.000% | 98.047% | 0.000% |

Evidence:

- [revision-7 precalibration protocol](precalibration_protocol_revision7.json)
- [terminal calibration summary](calibration/calibration_summary.json)
- [ordered attempts](calibration/attempts.json)
- [all evaluated calibration pairs](calibration/all_evaluated_paired_episodes.csv)
- [independent calibration evidence reconstruction](calibration/calibration_evidence_verification.json)
- [external artifact manifest](external_artifact_manifest.json)
- [execution orchestration provenance](orchestration/execution_orchestration.json)
- [zero-episode resource-guard migration receipt](orchestration/v25_revision7_guard2_migration_receipt.json)
- [Guardian terminal release receipt](orchestration/guardian_terminal_release_revision7.json) (`guardian_completed=false`)
- [compact-package hashes](SHA256SUMS)


## Runtime resource provenance (revision 8)

After the original exclusive-GPU guards terminated before any v25 episode, the
formal run was launched with the user-authorized resource-only revision 8. It
allows unrelated GPU workloads while requiring at least 3,500 MiB free before
each new GPU phase. The frozen gain grid, thresholds, identities, seeds,
algorithm, environment, and final-policy rule were unchanged.

- [revision-8 parallel orchestrator](orchestration/v25_complete_after_calibration_revision8_parallel.sh)
- [hash-bound parallel resource receipt](orchestration/v25_revision8_parallel_resource_receipt.json)

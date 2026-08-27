# v58--v61 Target-Height Curriculum and Filter Internalization

This series tests whether paper-style training domain randomization (DR) and a
stair-height curriculum can close the gap between CBF-filtered training and
filter-free deployment on the fixed 18.4 cm F2 stairs.

## Results

| Version | Training change | Key training result | Fixed 18.4 cm filter-off result |
|---|---|---:|---:|
| v58 | adaptive height curriculum + DR25 | 82--91% filtered rollout | 36/64 (56.25%) |
| v59 | adaptive curriculum + DR25 + filter 100%→0% | final pure-off 107/127 (84.25%) | round 3: 36/64 (56.25%); round 1: 39/64 (60.94%) |
| v60 | target-row floor + DR25 + filter 100%→0% | final pure-off **175/260 (67.31%)** | not run; training gate failed |
| v61 | v60 continuation, fixed target, failure-only A2 teacher | 53.28%→52.73%→49.28% | not run; training gate failed |

The high v58/v59 adaptive-curriculum rates were misleading because failed
environments could retreat to easier stair rows.  v60 fixes this structurally:
from round 3 onward, every reset is clamped to the highest row.  All four late
rounds record the exact histogram `[0, 0, 0, 0, 128]`.  With the filter fully
disabled, success stabilized at 67.31%.  This is stronger evidence than the
mixed-height 84.25%, but remains below the 75% acceptance threshold.

v61 also revealed an important robustness limitation.  The v60 checkpoint
started at only 53.28% under a fresh set of randomized friction, COM, encoder
bias, and observation noise.  Failure-only CBF residual teaching supplied
5,646--6,262 labeled transitions per round, but did not recover success.
Therefore no additional deployment evaluation was justified.

## Decision

No checkpoint in this series is accepted.  The target-row floor is retained as
the correct training implementation, while the next experiment should refresh
the physical DR sample between rollout rounds.  With only 128 environments,
the current startup-only randomization exposes each run to far fewer dynamics
samples than the paper's 4,096-environment setup.

## Provenance and files

- DR implementation commits: `f77adb93195f4290affba8aee66bbcbf296360a6`,
  `531798aa8540aefe00b577fa47087690c4df007c`
- Target-row floor commit: `577fb13a4de2fc1b116c8587a9386213bc2e559b`
- Common original checkpoint SHA-256:
  `cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8`
- v60 aligned round-5 checkpoint SHA-256:
  `f00e3a56276f629504234a20b40c124ee43a2f4d145cb143b3b2899acc024b27`
- v60 aligned actor SHA-256:
  `97a88c0ff2aa684766b499c606ae2c3eefd84e7f501c840058fa98efbb08fcc9`
- `decision_summary.json`: compact cross-version decision record.
- `v58_gate/`, `v59_gate_round03/`, and `v59_gate_round01/`: all 64
  fixed-target episode records and summaries.
- `v60_target_floor/round_05.pt`: exact checkpoint used to start v61.
- Each training directory contains complete `training_summary.json` and
  `round_metrics.{json,csv}` files.

Implementation: `experiments/scripts/refine_paper_dual_v35.py` and
`src/tasks/stairs_cbf/teacher_v30_math.py`.

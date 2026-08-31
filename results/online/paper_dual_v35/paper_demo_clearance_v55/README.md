# v55 Persistent Stair-Clearance CBF PPO

v55 addresses the next two training-signal mismatches found after v54.  The
paper states that the foot-clearance reference depends on the stair in front
of the robot, but the previous implementation only used that height during the
short CBF activation window and dropped back to the flat-ground reference as
soon as the toe cleared the edge.  v55 keeps the next-riser/top-platform
reference for the complete swing.  It also retains the humanoid Eq. (27)
reduced-order foot displacement while using the authors' public-demo scaling
of `10×` for the margin term and `100×` for action proximity.  The historical
extra moving-reference KL is disabled, leaving standard clipped PPO.

## Smoke and formal trajectory

The 65,536-transition smoke passed its signal gate: the pre-update filtered
rollout was 39/61 (63.93%), while the rollout after one actor update was 42/59
(71.19%).  Forward KL stayed at `4.70e-4` and clip fraction at 10.01%.

The independent formal run was configured for 8 rounds of 64 environments ×
1,024 steps.  It was stopped after 6 rounds (393,216 transitions) because the
aligned rollout peaked at checkpoint round 1 and all four later checkpoint
rollouts were lower:

| Rollout round | Evaluated checkpoint | Filtered stochastic success | KL | Clip fraction |
|---:|---:|---:|---:|---:|
| 1 | 0 (base) | 82/135 (60.74%) | 4.19e-4 | 8.34% |
| 2 | **1** | **91/132 (68.94%)** | 4.61e-4 | 9.91% |
| 3 | 2 | 83/131 (63.36%) | 4.28e-4 | 8.78% |
| 4 | 3 | 78/132 (59.09%) | 5.08e-4 | 11.33% |
| 5 | 4 | 90/136 (66.18%) | 5.76e-4 | 14.02% |
| 6 | 5 | 81/131 (61.83%) | 4.28e-4 | 8.83% |

The predeclared alignment rule therefore selected `round_01.pt`; round 2's
rollout is the first observation of that post-update actor.

## Untouched deployment gate

On previously unused seed `201352112`, the selected deterministic actor with
the runtime filter disabled achieved **42/64 (65.63%)**, below the 48/64 (75%)
threshold.  It was rejected, and filter-on plus additional seeds were not run.
This score is numerically three episodes above v54's untouched result, but the
seeds differ, so it is not claimed as a paired improvement.

The result shows that the corrected persistent clearance signal produces a
clear one-update improvement on the training distribution, but continued
fully filtered PPO does not transfer that improvement reliably to nominal
deployment.  The next iteration should change the rollout distribution rather
than strengthen the same filtered reward again.

## Provenance and files

- Source commit: `345f6a7c4d03b75685715c7d10f33231221783ac`.
- Base checkpoint SHA-256:
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`.
- Selected checkpoint SHA-256:
  `9c207de0b0b9868ee76c3f2ba8e6d22c281d022116de5e4d949d899744a13e85`.
- Selected actor SHA-256:
  `c64424fa6613428548802f253448cbed505d57b4c45eacbf8f7119bdc8c9ca49`.
- `early_stop_summary.json`: compact configuration, alignment, selection, and
  gate record.
- `round_metrics.{json,csv}`: complete six-round diagnostics.
- `smoke_training_summary.json`: complete two-round smoke evidence.
- `selected_round01.pt`: exact rejected deployment candidate.
- `untouched_seed201352112_filter_off_{summary.json,episodes.csv}`: untouched
  gate summary and all 64 episode records.

Implementation: `src/tasks/stairs_cbf/{cbf_math.py,mdp.py,paper_dual_v35.py}`
and `experiments/scripts/refine_paper_dual_v35.py`.

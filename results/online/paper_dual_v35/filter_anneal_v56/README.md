# v56 Linear Safety-Filter Annealing

v56 tests whether the gap between filtered training and nominal deployment can
be reduced by linearly annealing the fraction of environments that execute the
CBF filter from 100% to 0%.  The CBF dual reward remains counterfactual when an
environment is unshielded.  Rollout reporting now separates filter-on and
filter-off episodes, so the final checkpoint decision can use a completely
unshielded rollout rather than a shielded proxy.

## Smoke selection

Both smoke runs used four rounds of 64 environments × 512 steps.  The final
round executed no runtime filter and evaluated checkpoint round 3.

| Actor learning rate | Seed | Final unshielded success | Runtime |
|---:|---:|---:|---:|
| `5e-7` | 201352201 | 39/60 (65.00%) | 62.3 s |
| **`1e-6`** | **201352202** | **41/59 (69.49%)** | 63.3 s |

The `1e-6` smoke improved over v55's 42/64 (65.63%) untouched result enough to
justify one independent formal run.  The smoke is screening evidence only;
its different seed prevents a paired-improvement claim.

## Formal trajectory

The formal run used seed `201352301`, eight rounds of 64 environments × 1,024
steps (524,288 transitions), and completed in 242.7 seconds on an RTX 4080
SUPER.  A row's rollout evaluates the checkpoint from the preceding round.

| Round | Checkpoint | Filter fraction | Total success | Filter-off success |
|---:|---:|---:|---:|---:|
| 1 | 0 | 100.0% | 80/130 (61.54%) | n/a |
| 2 | 1 | 85.7% | 92/130 (70.77%) | 13/19 (68.42%) |
| 3 | 2 | 71.4% | 82/130 (63.08%) | 25/38 (65.79%) |
| 4 | 3 | 57.1% | 87/132 (65.91%) | 38/56 (67.86%) |
| 5 | 4 | 42.9% | 89/123 (72.36%) | 51/72 (70.83%) |
| 6 | 5 | 28.6% | 85/132 (64.39%) | 64/96 (66.67%) |
| 7 | 6 | 14.3% | 86/129 (66.67%) | 76/109 (69.72%) |
| 8 | **7** | **0.0%** | **75/133 (56.39%)** | **75/133 (56.39%)** |

The completely unshielded final rollout is below the 75% acceptance threshold
and also below the screening smoke.  The round-7 checkpoint is therefore
rejected.  A separate 64-episode untouched gate was deliberately not run: the
independent formal rollout already supplied 133 fully unshielded episodes and
failed the predeclared credibility threshold.

The useful negative result is that gradual filter removal is operational and
preserves mixed-rollout performance around 65–71%, but PPO updates remain
unstable once the rollout becomes entirely nominal.  Further work should alter
the objective or policy supervision, not spend more samples on this schedule.

## Provenance and files

- Source commit: `77dbd3a818db83fc4383470092217db6d0070804`.
- Warm-start v55 checkpoint SHA-256:
  `9c207de0b0b9868ee76c3f2ba8e6d22c281d022116de5e4d949d899744a13e85`.
- Rejected round-7 checkpoint SHA-256:
  `c5f311aeaa62815d458f5af67212bffc364d60fd692be9abbe7076bda616a512`.
- Rejected round-7 actor SHA-256:
  `31becd824a0e697192ecda78eb52253cc68db68ae6c66fdb77a7c655dbe38349`.
- `decision_summary.json`: compact smoke/formal decision record.
- `smoke_*_training_summary.json`: both complete screening runs.
- `formal_training_summary.json` and `formal_round_metrics.{json,csv}`:
  complete formal configuration and per-round diagnostics.
- `formal_round07.pt`: exact candidate evaluated by the final fully
  unshielded rollout.

Implementation: `src/tasks/stairs_cbf/teacher_v30_math.py` and
`experiments/scripts/{refine_cbf_teacher_v31.py,refine_paper_dual_v35.py}`.

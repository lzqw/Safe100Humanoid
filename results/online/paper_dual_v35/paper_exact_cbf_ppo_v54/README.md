# v54 Paper-Exact Stair CBF PPO

v54 corrects two implementation mismatches against CBF-RL
([arXiv:2510.14959v6](https://arxiv.org/abs/2510.14959)): the stair barrier is
the paper's horizontal next-riser hyperplane (`barrier_slope=0`) and the dual
reward measures the one-control-step swing-foot displacement in reduced-order
task space rather than the 12-D joint-action correction.  The deployable actor
interface remains the original blind 405-D proprioceptive history.

Before training, a same-seed 32-episode comparison checked the horizontal
barrier against the historical sloped barrier:

| Base checkpoint, filter on, seed `201351522` | Success | Intervention steps / riser | Mean velocity correction |
|---|---:|---:|---:|
| Historical slope `0.8` | 27/32 (84.38%) | 4.4569 | 1.0997 |
| Paper horizontal slope `0` | 26/32 (81.25%) | **1.6509** | **0.4119** |

The paper-exact geometry reduced interventions by about 63% while losing one
success episode in this short comparison.  It was retained for the formal run
because this experiment tests paper alignment, not a post-hoc geometry search.

## Formal result

The formal run used fully filtered on-policy PPO with the paper-style bounded
dual reward, no explicit teacher, 64 environments, 1,024 steps, 8 rounds,
action standard deviation `0.05`, and actor learning rate `2e-7`.

| Formal v54 result | Value |
|---|---:|
| Training transitions | 524,288 |
| RTX 4080 training time | 231.88 s (3 min 51.88 s) |
| Round-1 pre-update filtered rollout | 90/132 (68.18%) |
| Round-8 pre-update filtered rollout | 83/129 (64.34%) |
| Untouched seed `201351812`, final round-8, filter off | **39/64 (60.94%)** |
| Predeclared deployment threshold | 48/64 (75%) |
| Decision | **rejected** |

The round-N rollout precedes the round-N update, so round 8 evaluates the
round-7 actor rather than the published round-8 checkpoint.  The independently
seeded deterministic deployment gate evaluates the actual round-8 checkpoint.
It missed the threshold by nine success episodes, so filter-on and additional
validation seeds were not run.

This negative result narrows the remaining issue: correcting the CBF equation
and reward coordinate makes the safety projection substantially less invasive,
but the fully filtered PPO trajectory still does not improve nominal,
filter-free deployment behavior.

## Provenance and files

- Source commit: `115bb0c4fe2f4dff999e6a6d4593264f241f5605`.
- Base checkpoint SHA-256:
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`.
- Base actor SHA-256:
  `3964c0ef24707addbaa0dacfd4fd627882bd8b45a2c9799142710fa45bc29499`.
- Final checkpoint SHA-256:
  `61cd47bf2504e6f7a352c7a87769070e5cbe9c38ebde9ea4f0194669bcf46b9b`.
- Final actor SHA-256:
  `33ecce33a02221c2fe92590b185b3e241fbabf5454ab31aceb7a153f1350fbe0`.
- `training_summary.json` and `round_metrics.{json,csv}` contain the complete
  eight-round configuration and diagnostics.
- `candidate_round08.pt` is the exact rejected final checkpoint.
- `untouched_seed201351812_filter_off_{summary.json,episodes.csv}` contains the
  deployment-gate summary and all 64 episode records.
- `base_seed201351522_slope*_on_summary.json` records the geometry comparison.
- `smoke_lr*_training_summary.json` records the learning-rate calibration.

Implementation: `src/tasks/stairs_cbf/{actions.py,velocity_cbf_action.py,mdp.py,paper_dual_v35.py}`,
`experiments/scripts/refine_paper_dual_v35.py`, and
`experiments/scripts/evaluate_velocity_cbf_v34.py`.

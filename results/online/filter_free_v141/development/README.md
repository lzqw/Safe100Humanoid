# v141 Filter-Free Refinement — Development Status

Status at 2026-08-31 12:18 CST: autonomous successive-halving development is
running on the RTX 4080 SUPER. CARLA is not running; unrelated GuardianFlow
jobs were left untouched.

Implemented method: Intervention-Aware CBF Distillation PPO. Training always
executes the CBF-safe action, attenuates nominal-action PPO credit on corrected
transitions, distills detached safe raw actions with positive group-normalized
advantage weights, and standardizes target/F1-retention advantages separately.
The actor remains the original 405-D policy and the runtime CBF is disabled for
the primary deployment evaluation.

The first complete F2 development candidate (`g1_eta0_posadv_r0_l02`) used two
128-env × 1024-step rounds from frozen v139 and produced:

- target F2 CBF-off success: 74.21875%
- F1 retention CBF-off success: 74.21875%
- target F2 CBF-on success: 70.3125%
- counterfactual would-intervene fraction: 9.69465%
- mean counterfactual correction norm: 0.0186182
- actor moving forward KL: 0.000413863

This is promising but not yet a final claim. Against the previously published
v140 frozen evaluation (which used a different evaluation seed), F2 CBF-off is
about +5.27 percentage points, while the would-intervene and shield-gap gates
are not yet satisfied. Fresh-seed frozen baselines, all Generation 1–3
candidates, formal three-seed training, and 512-episode paired evaluation remain
in progress.

Machine-side resumable state and raw artifacts are currently stored at:

`/home/carla/LZQW/SAFE100/humanoid/artifacts/filter_free_v141/development/`

See `development_status.json` for the machine-readable snapshot. Final
development and formal tables/checkpoints will be published under the adjacent
`development/` and `formal/` directories only after their fixed evaluations
finish.

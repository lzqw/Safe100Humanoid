# v141 Filter-Free Refinement — Development Status

Status at 2026-08-31 12:26 CST: the initial development launch was stopped and
invalidated after a formula audit found that soft PPO weights were normalized
by their sum rather than by the transition population. Commit `c75b017`
corrects the objective to the required `mean(c_t * L_t)`, adds explicit
per-transition group IDs, and makes post-Generation-3 refinement continue
automatically. Corrected successive-halving development is restarting on the
RTX 4080 SUPER. CARLA is not running; unrelated GuardianFlow jobs remain
untouched.

Implemented method: Intervention-Aware CBF Distillation PPO. Training always
executes the CBF-safe action, attenuates nominal-action PPO credit on corrected
transitions, distills detached safe raw actions with positive group-normalized
advantage weights, and standardizes target/F1-retention advantages separately.
The actor remains the original 405-D policy and the runtime CBF is disabled for
the primary deployment evaluation.

The first F2 candidate from the superseded launch produced the following raw
measurements:

- target F2 CBF-off success: 74.21875%
- F1 retention CBF-off success: 74.21875%
- target F2 CBF-on success: 70.3125%
- counterfactual would-intervene fraction: 9.69465%
- mean counterfactual correction norm: 0.0186182
- actor moving forward KL: 0.000413863

These numbers are retained only as an audit trail and are excluded from every
ranking, gate, and claim because their PPO scaling did not match the specified
algorithm. Fresh corrected candidates, fresh-seed frozen baselines, formal
three-seed training, and 512-episode paired evaluation remain in progress.

Machine-side resumable state and raw artifacts are currently stored at:

`/home/carla/LZQW/SAFE100/humanoid/artifacts/filter_free_v141/development/`

See `development_status.json` for the machine-readable snapshot. Final
development and formal tables/checkpoints will be published under the adjacent
`development/` and `formal/` directories only after their fixed evaluations
finish.

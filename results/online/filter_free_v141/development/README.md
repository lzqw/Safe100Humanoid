# v141 Filter-Free Refinement — Development Status

## Latest snapshot — Generation 7 complete, Generation 8 running

Status at 2026-08-31 18:34 CST: the autonomous loop has completed Generations
1–7 and is running Generation 8 on the RTX 4080 SUPER. No CARLA/Unreal process
is running. One historical failed-job record is retained from the already
recovered CUDA RNG resume bug; there is no current training failure.

The Generation-7 F3 leader (`g7_epochs4_1`) reaches 76.5625% target CBF-off
success versus Frozen's 64.84375%, retains F1 at 75.78125%, and has a -1.5625
point shield gap. It passes target success, F1 retention, and shield-gap gates.
Its would-intervene fraction is 9.06759% versus Frozen's 10.16317%, so the
required 25% reduction remains the only failed F3 gate.

The score-selected F2 leader remains `g3_eta0.25_3`: 78.90625% target CBF-off,
71.875% F1 retention, -14.84375 points shield gap, and 9.62572%
would-intervene. It passes target and retention but fails the absolute/reduced
shield-gap gate and the would-intervene gate. This exposed a search-control
problem: adaptive generations were mutating the highest `J_dev` parent even
when another candidate satisfied more of the actual termination gates. Commit
`6e4a84b` preserves the required `J_dev` ranking but chooses future mutation
parents by gates passed, normalized gate deficit, then `J_dev`. It also resumes
from completed adaptive-generation summaries instead of replaying G3 onward.

See `generation_7_snapshot.json` for the compact machine-readable evidence.
Formal freezing is not allowed yet; both specialists must pass every
development gate first.

Status at 2026-08-31 13:17 CST: all sixteen corrected Generation-1 candidates
and the fresh paired Frozen baselines are complete. Generation 2 started
automatically on the RTX 4080 SUPER with zero failed Generation-1 jobs. The F2
leader reached 72.65625% target CBF-off success versus Frozen's 67.1875%
(+5.46875 percentage points), retained F1 at 73.4375%, and met the shield-gap
gate. The F3 leader reached 72.65625% target CBF-off success versus Frozen's
64.84375% (+7.8125 points), retained F1 at 76.5625%, and reduced shield gap to
1.5625 points. Both leaders therefore pass target success, F1 retention, and
shield-gap checks.

The remaining bottleneck is nominal-policy internalization. F2 would-intervene
changed from 9.97753% to 9.97640% (0.0113% relative reduction), while F3 changed
from 10.16317% to 9.72216% (4.3393% relative reduction); both remain below the
required 25% reduction. Generation 2 is consequently testing longer/stronger
allowed actor updates and retention ratios. CARLA is not running; unrelated
GuardianFlow jobs remain untouched.

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
algorithm. The corrected implementation and autonomous
freeze/formal/publication pipeline are committed in `839f695`. Generation-2
candidates, formal three-seed training, and 512-episode paired evaluation
remain in progress.

Machine-side resumable state and raw artifacts are currently stored at:

`/home/carla/LZQW/SAFE100/humanoid/artifacts/filter_free_v141/development/`

See `development_status.json` for the machine-readable snapshot. Final
development and formal tables/checkpoints will be published under the adjacent
`development/` and `formal/` directories only after their fixed evaluations
finish.

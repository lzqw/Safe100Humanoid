# Safe100Humanoid

Safe reinforcement learning for Unitree G1 stair climbing, built on
`unitree_rl_mjlab`, MJLab, MuJoCo-Warp, and RSL-RL PPO.

This repository combines:

- the training-time safety filtering and dual reward design from
  **CBF-RL: Safety Filtering Reinforcement Learning in Training with Control
  Barrier Functions**;
- Flat Patch target selection, position-based velocity commands, and terrain
  edge safety ideas inspired by **Hiking in the Wild**;
- the Unitree G1 model and locomotion stack from `unitree_rl_mjlab`;
- GPU-parallel simulation through MJLab and MuJoCo-Warp;
- PPO optimization through RSL-RL.

The implementation is simulation-only. It does not contain or run real-robot
control, networking, or sim-to-real deployment.

## Method

The selected stair barrier prevents the swing toe from entering the next stair
riser while the foot is below the tread:

```text
h(q) = x_riser - (x_swing_foot(q) + toe_margin)
psi(q, qdot) = -Jx_swing(q) qdot + alpha h(q) >= 0
```

The nominal 12-DoF lower-body position target is converted to joint velocity,
projected onto the single CBF half-space with a closed-form Euclidean QP, then
converted back to a safe position target. During training, the policy also
receives the bounded dual reward:

```text
min(psi_nominal, 0)
+ exp(-||q_target_policy - q_target_safe||^2 / sigma^2) - 1
```

The actor uses 405 observations: five frames of proprioception, a gait phase,
the 3-D velocity command, and previous actions. The asymmetric critic uses one
283-D frame that additionally contains a privileged terrain height scan,
base linear velocity, and foot state.

See [the training framework summary](docs/TRAINING_FRAMEWORK_SUMMARY.md) and
[the implementation report](docs/ALGORITHM_IMPLEMENTATION_REPORT.md) for the
full observation, action, reward, curriculum, and CBF definitions.

The repository also contains a simulation prototype for CBF-protected online
safe refinement under long-stair and joystick-command OOD shifts. It adds a
799-D full privileged critic, conservative single-clipped PPO, temporal credit
for actual CBF interventions, and transactional candidate rollback. See
[the online refinement report](docs/ONLINE_SAFE_REFINEMENT.md).
Measured OOD calibration and candidate rollback evidence are summarized in
[the online refinement results](docs/ONLINE_REFINEMENT_RESULTS.md).
One fall-aware, backtracked PPO candidate passed the robust safety gate by
retaining DQ success while reducing CBF intervention and correction. This is a
modest simulation result; filter-free performance did not improve. Curated GPU
artifacts, the accepted and rollback checkpoints, and successful and failed DQ
videos are under `results/online/`.
A resumed second round improved D0/DQN but regressed 512-episode DQ success and
was therefore rolled back, demonstrating that acceptance remains transactional
across rounds.

## One-seed result

Evaluation uses a fixed 13 cm six-step staircase, 128 deterministic episodes,
and seed 42.

| Method | Runtime filter | Success | Fall | Mean max progress | CBF violation integral |
|---|---:|---:|---:|---:|---:|
| CBF-trained | on | 95.31% | 4.69% | 3.309 m | 9.75 |
| CBF-trained | off | **95.31%** | **4.69%** | **3.315 m** | **7.56** |
| Equal-budget nominal | off | 0% | 34.38% | 0.748 m | 1006.18 |

This is a one-seed method-level result, not a multi-seed statistical claim or
a numerical reproduction of the paper. The evaluation policy retains 95.31%
success without the runtime filter, which is evidence of safety internalization
for this experiment.

The exact aggregate and per-episode outputs are under `results/evaluation/`.
A successful filter-free rollout is provided at
[`results/videos/g1-stairs-cbf-filter-off-seed42-step-0.mp4`](results/videos/g1-stairs-cbf-filter-off-seed42-step-0.mp4).

## Repository layout

```text
src/tasks/stairs_cbf/       Stair terrain, CBF, online PPO, rewards, and configs
experiments/scripts/        Smoke, evaluation, online refinement, and video scripts
experiments/tests/          Pure tensor CBF and online-refinement tests
docs/                       Method, training, evaluation, and assumption reports
results/evaluation/         Aggregate JSON/CSV and per-episode CSV
results/online/             Online gate audits, rollback checkpoint, and videos
results/models/             Final CBF and nominal PT/ONNX artifacts
results/tensorboard/        Final TensorBoard event files
results/videos/             Deterministic stair-climbing rollout
```

The remaining `src/`, assets, and runner code are the pinned
`unitree_rl_mjlab` base needed to run the task.

## Environment

The verified environment uses:

```text
Python 3.11.15
PyTorch 2.7.0+cu128
MJLab 1.2.0
MuJoCo 3.5.0
MuJoCo-Warp 3.5.0
Warp 1.12.0
RSL-RL 5.0.1
```

Install into an isolated Python 3.11 environment. The exact package snapshot is
in `requirements-lock.txt`; installation may require a PyTorch/CUDA index
appropriate for the host:

```bash
python -m pip install -r requirements-lock.txt
```

For a smaller setup based on the package metadata:

```bash
python -m pip install "torch==2.7.0" "rsl-rl-lib==5.0.1"
python -m pip install -e .
```

## Tests

Pure tensor tests:

```bash
pytest -q experiments/tests/test_cbf_math.py
pytest -q experiments/tests/test_cbf_math.py \
  experiments/tests/test_online_refinement.py
```

GPU environment smoke:

```bash
python experiments/scripts/smoke_stairs_cbf.py \
  --repo . \
  --task Unitree-G1-Stairs-CBF \
  --num-envs 4 \
  --steps 200 \
  --seed 42 \
  --expected-actions 12 \
  --output results/smoke.json
```

The verified migration test produced a real MuJoCo-Warp adversarial nominal
margin of about `-29.16`, repaired by the filter to approximately `+1.1e-6`.

## Training

The effective CBF run used 1024 environments, 24 steps per environment, seed
42, and 1500 PPO updates:

```bash
python scripts/train.py Unitree-G1-Stairs-CBF \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 1500 \
  --agent.save-interval 500 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --agent.run-name paper_dual_curriculum_seed42_1024 \
  --enable-nan-guard True
```

Portable wrappers are available in `experiments/train_cbf_1500.sh` and
`experiments/train_nominal_1500.sh`.

## Models and evaluation

Final checkpoints:

```text
results/models/cbf/model_1500.pt
results/models/cbf/policy.onnx
results/models/nominal/model_1499.pt
results/models/nominal/policy.onnx
```

`model_1499.pt` is the nominal runner's zero-based filename after 1500 PPO
updates.

Run a fixed-height evaluation with:

```bash
python experiments/scripts/evaluate_stairs.py \
  --repo . \
  --task Unitree-G1-Stairs-CBF \
  --checkpoint results/models/cbf/model_1500.pt \
  --label cbf_exact13 \
  --num-envs 128 \
  --num-episodes 128 \
  --seed 42 \
  --fixed-step-height 0.13 \
  --output-json results/evaluation/cbf_exact13.json \
  --output-csv results/evaluation/cbf_exact13.csv
```

## Scope and attribution

- The humanoid stair source code and author parameters from CBF-RL are not
  public; the documented toe margin, class-K gain, activation band, noise-free
  stair geometry, and reward coefficients are engineering assumptions.
- Hiking in the Wild informed the navigation and terrain-edge representation;
  this repository does not include InstinctLab code, depth policies, MoE,
  AMP/WASABI, or Foot Volume Points.
- The project inherits and preserves the Apache-2.0 license of
  `unitree_rl_mjlab`. See `THIRD_PARTY_NOTICES.md` for dependency and research
  attribution.

## References

- Yang et al., [CBF-RL: Safety Filtering Reinforcement Learning in Training
  with Control Barrier Functions](https://arxiv.org/abs/2510.14959).
- Zhu et al., [Hiking in the Wild: A Scalable Perceptive Parkour Framework for
  Humanoids](https://arxiv.org/abs/2601.07718).
- Zakka et al., [mjlab: A Lightweight Framework for GPU-Accelerated Robot
  Learning](https://arxiv.org/abs/2601.22074).
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab).
- [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl).

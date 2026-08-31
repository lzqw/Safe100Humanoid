# CBF-Protected Online Safe Refinement

## Scope

This module simulates brief on-policy refinement after a stair policy has been
deployed. It is simulation-only. It does not run on a Unitree robot, does not
claim that the toe-riser CBF prevents falls, and does not remove the runtime
safety layer.

The retained interface is:

- 405-dimensional actor observation (five frames);
- 12 lower-body joint-position actions;
- the existing Gaussian MLP policy and toe-riser CBF projection;
- the existing dual CBF reward.

The deployment shift changes the horizon, command process, fixed target stair,
episode length, and reset distribution.

## Controlled OOD matrix

| Domain | Risers | Command | Geometry | Purpose |
|---|---:|---|---|---|
| D0 | 6 | waypoint | 13 cm / 35 cm | ID retention |
| D1 | 18 | waypoint | 13 cm / 35 cm | horizon shift |
| D2 | 6 | joystick | 13 cm / 35 cm | command shift |
| D3 | 18 | joystick | 13 cm / 35 cm | horizon + command |
| D4 | 18 | joystick | fixed 14.5 cm / 33 cm profile | formal target |
| D5 | 18 | joystick | neighboring fixed profile | local generalization |
| DQ | 9 | joystick | 13 cm / 35 cm | calibrated quick prototype |
| DQN | 9 | joystick | 13.2 cm / 35.5 cm | quick-target neighbor |

D4 uses an explicit fixed vector of per-riser rise errors (up to 5 mm) and
tread errors (up to 10 mm). Geometry is generated once from this profile and
does not change between online episodes. DQ exists because the current base
checkpoint has zero success on D3/D4; it is reported separately and is never
renamed as the formal target.

## Geometry and command pipeline

`ForwardStairsTerrainCfg` emits both tread targets and exact riser metadata.
The CBF, privileged critic, riser reward, and success evaluator all read the
same metadata. Variable tread depths are therefore not reconstructed from a
global width.

`JoystickVelocityCommand` generates a long-held forward command in
0.30--0.50 m/s, occasional lateral/yaw correction pulses, 40--160 ms delivery
delay, low-pass filtering, and delayed release at the top. The actor receives
only the delivered `[vx, vy, wz]`. The full critic additionally sees raw and
delivered command, derivative, and delay state.

## Stationary reward and success

The online tasks replace absolute position reward with

```text
r_dx = clip((x_t - x_{t-1}) / (v_max * dt), -1, 1)
```

and add a fixed riser-crossing event reward and top-completion event bonus.
Success is derived from the actual number of risers and final platform
metadata. There is no six-step 2.65 m constant in the online evaluator.

Because both a fall and reaching the top are non-timeout terminations, the
generic termination reward cannot distinguish them. Online tasks therefore
use a dedicated `fall_termination` term with the base task's `-200` weight
(a fixed `-4` event after MJLab `dt=0.02` scaling). Successful completion is
never charged this penalty. Removing this distinction in the first prototype
silently removed the principal rare-failure signal; the accepted run includes
the corrected fall-only term.

## Shielded on-policy data

At each step the behavior actor samples `a_policy`. RSL-RL stores this original
sample and its log probability. The environment action term then computes and
executes `a_safe = F_CBF(s, a_policy)`. The PPO ratio is therefore

```text
pi_candidate(a_policy | actor_obs) / pi_behavior(a_policy | actor_obs)
```

and is never evaluated at `a_safe`. The CBF is treated as part of the modified
transition kernel; gradients do not pass through the projection.

The side buffer records actual projection events, correction magnitude,
nominal target, and safe target. Geometric activation and actual intervention
are logged separately.

## CBF temporal credit

For an actual intervention at time `t`, normalized correction intensity is
assigned to the preceding ten actions with decay 0.8. Credit never crosses an
episode boundary. It is subtracted before GAE:

```text
r_online = r_task + r_dual - lambda_pre * pre_intervention_cost
```

The only policy-gradient objective remains the single clipped PPO surrogate.

An optional safe-action auxiliary can be enabled on true interventions:

```text
L_safeBC = E[I_t ||mu_theta(o_t) - stopgrad(a_safe_raw)||^2]
```

It uses only actual projection states, never geometric activation alone. This
is a supervised auxiliary, not an additional policy-gradient objective. It is
applied as one stateless layer-wise SGD micro-step whose effective learning
rate is logged. This avoids a separate Adam step making small loss weights
produce nearly full-size first updates. It remains disabled by default because
the local correction need not be long-horizon optimal and the actor does not
receive every privileged geometric feature. The accepted round did not use
this auxiliary.

## Full privileged critic

The actor remains 405-dimensional. The online critic concatenates:

- actor history (405);
- previous 283-dimensional privileged critic group;
- 111 online deployment features.

The resulting critic input is 799 dimensions. Online features include true
root/joint/foot state, current and remaining risers, the next three rise/tread
pairs, joystick delivery state, previous-step CBF state, progress, episode
time, and accumulated interventions. No current sampled-action projection
quantity enters `V(s_t)`.

The base 283-dimensional critic is expanded by copying its first-layer columns
at offset 405; all new input columns are zero-initialized. The actor is frozen
during critic burn-in. The actor observation normalizer is frozen throughout
online refinement; the full critic normalizer may update.

## Conservative PPO

| Parameter | Value |
|---|---:|
| clip | 0.05 |
| actor LR | 1e-5 |
| critic LR | 1e-4 |
| epochs | 2 |
| minibatches | 4 |
| target KL | 0.003 |
| entropy coefficient | 0 |
| max grad norm | 0.5 |
| rollout | 256 steps/env default; 768 in accepted round |

Actor linear layers use LR multipliers `(0.10, 0.25, 0.50, 1.0)`. The base
standard deviation is scaled by 0.35 and clipped to `[0.05, 0.35]`; its
learning rate is zero in the first version. A new optimizer is built, so base
training momentum is not imported.

## Candidate transaction

Each accepted actor is snapshotted before a round. After one candidate update:

1. reject non-finite parameters, excessive KL, clip fraction, or saturation;
2. evaluate old and candidate on D0, the target, and its neighbor;
3. require no target success/fall regression, bounded intervention per riser,
   D0 retention, and bounded deterministic-mean KL from the base actor;
4. accept the candidate or atomically restore actor, critic, and optimizer;
5. after rejection halve actor LR and scale exploration std by 0.8.

The accepted experiment additionally uses a conservative backtracking line
search along the same PPO update direction. Only trainable actor MLP tensors
are interpolated; frozen normalization, bounded variance, critic, and PPO
objective are unchanged. Fractions 0.25, 0.50, 0.75, and 1.0 were screened;
the 0.50 point was the only one taken to the final large-batch gate. This is a
proximal step-size selection, not residual control or a second RL objective.

GPU MuJoCo-Warp contact solving is not bitwise deterministic. The GPU gate
therefore aggregates multiple fixed-seed batches and records replicate
statistics. Small four-episode checks are smoke tests, not acceptance evidence.

The cross-round KL holds the base exploration standard deviation fixed and
measures actor-mean drift. Exploration std is governed separately by clipping
and rejection-time reduction; otherwise intentionally reducing exploration
would incorrectly consume the entire policy-drift budget.

Accepted 799-D checkpoints can be supplied with
`--resume-online-checkpoint`. The original base checkpoint remains the D0 and
total-drift reference. On resume, actor, critic, and checkpoint integrity are
loaded strictly, then Adam is rebuilt: a backtracked actor's saved momentum
belongs to the full pre-line-search step and must not leak into the next
fractional round. The second simulated round exercised this path and was
rolled back after its 512-episode DQ gate failed.

## Commands

```bash
python -m pytest -q \
  experiments/tests/test_cbf_math.py \
  experiments/tests/test_online_refinement.py
```

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python \
  experiments/scripts/evaluate_online_stairs.py \
  --repo "$PWD" \
  --task Unitree-G1-Stairs-Online-D4 \
  --checkpoint results/models/cbf/model_1500.pt \
  --num-envs 32 --num-episodes 128 --seed 42 \
  --output-json results/online/base_d4.json \
  --output-csv results/online/base_d4.csv
```

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python \
  experiments/scripts/online_refine_stairs.py \
  --repo "$PWD" \
  --base-checkpoint results/models/cbf/model_1500.pt \
  --output-dir results/online/refine_dq \
  --train-domain DQ --neighbor-domain DQN \
  --num-envs 16 --rollout-steps 256 \
  --critic-burn-in-rounds 2 --online-rounds 3 \
  --actor-learning-rate 5e-6 --critic-learning-rate 1e-4 \
  --pre-intervention-weight 1.0 --std-scale-from-base 0.25 \
  --safe-bc-weight 0.05 \
  --eval-num-envs 16 --eval-num-episodes 16 \
  --gate-device cuda:0 --gate-repeats 3 --seed 42
```

Use `--train-domain D4 --neighbor-domain D5` for the formal target after a
base policy has sufficient nonzero D4 success to support brief refinement.

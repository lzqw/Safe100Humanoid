# CBF-Guided Brief PPO Refinement (v14)

v14 is the deliberately reduced online method. The simulation CBF-RL base
training stage is unchanged. After deployment to one fixed unknown context,
the online stage contains only:

- one 405-D actor;
- one 838-D privileged value critic;
- the runtime CBF action filter;
- one clipped PPO objective and GAE;
- a two-candidate transactional rollback.

The v12/v13 fall critic, intervention critic, risk head, adaptive Lagrange
multipliers, base-policy and retention-bank KL losses, correction
distillation, per-round neighboring-domain gate, and per-round confidence
promotion are not instantiated or optimized in this mode. The older drivers
remain available solely to reproduce those ablations.

## Shielded raw-action data flow

At every online step, the behavior policy samples `a_policy`. PPO stores this
raw action and its behavior log probability. The environment executes
`a_safe = CBF(state, a_policy)`. Runtime audits require the PPO buffer to match
`a_policy` exactly and the executed action to match the configured CBF route.
The runtime CBF is never removed in v14.

## One reward and one critic

The privileged critic fits returns from

```text
r_tilde = r_task + r_fall + lambda_k * r_dual_CBF
```

with the task-first schedule:

- rounds 1–2: `lambda_k = 0`;
- rounds 3–5: `lambda_k = 0.02`.

There are no cost advantages or multiplier updates. Hard-case transitions use
the same scalar reward and critic as normal transitions.

## Formal five-round protocol

```text
target domain                 DQH
runtime CBF                   on
rollout                       64 envs × 1024 steps
normal/hard transition slots  80% / 20% (51 / 13 envs)
hard-case actor weight        0.5
actor learning rate           2e-6 on every actor layer
PPO epochs                    1
PPO clip                      0.05
target-KL early stop          0.005
hard update KL gate           KL < 0.01
exploration std               fixed at 0.35× base, bounded
candidate fractions           {0.5, 1.0}
candidate evaluation          128 paired target episodes each
online rounds                 5
```

Thirteen fixed hard-case slots are refilled from the historical failure bank
after termination, keeping the full actor batch at `13/64 = 20.3125%`
hard-case transitions. This differs from the v12/v13 one-shot reset behavior,
which remains unchanged in its legacy path.

## Training gates

For each candidate, the target score is

```text
S_T = success_rate - fall_rate - 0.01 * CBF_interventions_per_riser
```

A candidate is eligible only when parameters are finite, update KL is below
0.01, paired initial-state signatures match, `S_T` strictly improves, and its
target fall rate is no more than three percentage points above the current
policy. No per-round confidence interval is computed.

Every second round, D0 is evaluated against the fixed online-start policy. Its
success rate may fall by at most five percentage points. A violation restores
the latest complete policy/critic/optimizer snapshot that passed D0. DQNH is
not consulted during training and is evaluated only after refinement.

## Independent final evidence

Training point gates are not paper evidence. The final audit evaluates the
online-start and final policies on identical initial episodes for each of
three independent adaptation seeds:

- DQH: 512 episodes per training seed;
- D0: 256 episodes per training seed;
- DQNH: 256 episodes per training seed.

The audit reports hierarchical paired-bootstrap 95% confidence intervals,
resampling training seeds and paired episodes. The target improvement claim is
made only if

```text
LCB95[SR_DQH(final) - SR_DQH(online_start)] > 0.
```

## Reproduction

Run one training seed with:

```bash
SAFE100_SEED=42 bash experiments/scripts/run_brief_ppo_v14.sh
```

After seeds 42, 142, and 242 finish, run:

```bash
bash experiments/scripts/run_final_audit_v14.sh
```

The training summary explicitly labels its small final evaluation as a
diagnostic. `final_audit_summary.json` is the independent evidence artifact.

## Formal result

The complete three-training-seed evidence is published in
[`results/online/brief_ppo_v14/`](../results/online/brief_ppo_v14/README.md).
The DQH success point estimate changed from 89.388% to 89.974%, but the paired
95% interval for the `+0.586` percentage-point delta was
`[-1.174, +2.474]` percentage points. The required positive lower-bound gate
did not pass, so v14 does not claim a statistically supported DQH success
improvement. D0 success improved by `+3.646` percentage points with a paired
95% interval of `[+0.521, +7.292]` percentage points.

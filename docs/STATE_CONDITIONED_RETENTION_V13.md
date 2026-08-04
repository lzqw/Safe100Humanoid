# State-conditioned retention anchors (v13)

## Claim boundary

This work studies safe online adaptation to an unknown, fixed, combined
deployment shift in simulation. The hidden deployment context is treated as

\[
\xi=(\text{terrain},\text{command},\text{plant},\text{sensor}),
\]

and is fixed during one adaptation run. D0 is the retention domain, DQH is the
target fixed context, and DQNH is a neighboring fixed context. This is not an
exhaustive sim-to-real claim. Runtime CBF remains part of the deployment stack
and is not removed by v13.

The evidence levels are:

- Level A: DQH last-mile adaptation from an already capable deployed actor;
- Level B: fixed combined OOD contexts, evaluated only after Level A succeeds;
- Level C: stress contexts used for evaluation, not as a promotion claim.

## Motivation

The strongest v12 no-correction candidate improved DQH point estimates but
lost too much D0 performance. A rollout-local global KL is a weak retention
signal because it is measured mainly on the target on-policy state
distribution. v13 keeps the target policy gradient unchanged and adds fixed,
state-conditioned anchors on states from D0 and DQNH.

## Objective

For target-domain on-policy observations and two fixed actor-observation banks,
v13 minimizes

\[
L = L_{\mathrm{PPO}}(D_{\mathrm{DQH}})
  + \beta_0\,\mathbb{E}_{s\sim B_0}
    [D_{\mathrm{KL}}(\pi_\theta(\cdot|s)\|\pi_{\mathrm{ref}}(\cdot|s))]
  + \beta_N\,\mathbb{E}_{s\sim B_N}
    [D_{\mathrm{KL}}(\pi_\theta(\cdot|s)\|\pi_{\mathrm{ref}}(\cdot|s))].
\]

`pi_ref` is the frozen actor at the start of adaptation, not a moving target.
Only DQH rollout data contributes a PPO policy gradient. The D0 and DQNH banks
contain the 405-dimensional deployed `actor` group only; critic and
`online_privileged` observations are rejected by schema validation.

The initial weights are `beta_0=0.02` and `beta_N=0.01`. Each has an independent
KL budget of `0.002`. A weight can increase, up to `0.20`, only when its exact
fixed-bank KL exceeds its budget. Weights, bank cursors, bank identities, and
the frozen reference actor are all included in transactional snapshots and
checkpoint state.

## Fixed banks

Both banks were collected with the deterministic deployed mean policy and
runtime CBF enabled. Samples are shuffled within a stair stage and stored in
round-robin stage order, so every contiguous training slice remains balanced
without exposing stage labels to the actor loss.

| Bank | Size | Stages | Per-stage counts | Observation SHA-256 |
|---|---:|---:|---|---|
| D0 | 24,000 | 6 | 4,000 each | `49685396180c50dfdbb17094a5badd70af437c253dc1e7fbc692e9e7754bb8a3` |
| DQNH | 24,000 | 9 | 2,667 × 6; 2,666 × 3 | `6acf2b7c6ea34dffae6b698e7cd371705dfb8ddd54410f1897c1a08ad8cb2f0b` |

Both came from deployed checkpoint SHA-256
`cc27f228809a4a5b9862119eded31e81588c95ad066168b77a77232d035620a5`.
The large `.pt` banks are reproducible artifacts and are intentionally not
committed. Their manifests and file checksums are tracked under
`results/online/retention_v13/banks/`.

## Matched protocol

The only intended actor-objective difference is:

- Arm A: v12 global base anchor weight `0.01`;
- Arm B: global anchor off, D0/DQNH fixed-bank anchors `0.02/0.01`.

Both arms use:

- 64 environments × 1,024 rollout steps;
- two critic burn-in rounds and three actor rounds;
- normal/hard/neighbor routing of 84%/8%/8%, with hard-case policy weight 0;
- actor learning rate `2e-6`;
- correction distillation off;
- candidate fractions `{0.25, 0.5, 1.0, 1.5}`;
- 64 paired DQH episodes for candidate-family screening;
- 48 paired episodes per domain and policy for the in-run gate;
- D0 and DQNH success/fall tolerance of 2 percentage points;
- a disjoint final audit of 512 paired episodes per domain and policy.

The stochastic distribution API samples an unused action while exposing its
parameters. Fixed-bank forwards therefore run inside saved/restored CPU and
CUDA RNG scopes, preventing the anchor diagnostics from changing later rollout
noise independently of their gradient.

## Promotion contract

A candidate must show task improvement with a positive paired 95% lower bound,
must not increase fall or CBF demand, and must retain D0 and DQNH within the
predeclared bounds. Candidate-selection seeds and final-audit seeds are
disjoint. Rejection restores actor, task/cost/risk critics, optimizer, adaptive
cost multipliers, anchor weights, anchor cursors, normalizers, and the frozen
reference state.

## Validation before formal evidence

- 45 pure tensor/config regression tests pass in the authoritative environment;
- Python compilation and shell syntax checks pass;
- real GPU bank collection validates dimensions, stage counts, finite values,
  checkpoint identity, and observation/file hashes;
- a real dual-anchor PPO smoke update produced finite D0/DQNH bank KL values of
  `2.321e-4` and `2.280e-4`;
- the restore smoke test exactly restored actor parameters, both adaptive
  weights, both bank cursors, and the frozen reference after deliberate
  mutation.

Formal A/B and final-audit results are published only after their JSON ledgers
finish; a smoke acceptance is not treated as performance evidence.

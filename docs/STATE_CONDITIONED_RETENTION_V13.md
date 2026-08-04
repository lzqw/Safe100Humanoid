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

## Formal matched-arm result

Both arms reused the byte-identical baseline matrix (SHA-256
`fa4e36fd4bdddf4470e0f239dee549005caae677d4df8371584d81f54853ff1d`).
All six 48-episode/domain candidates were rejected and transactionally rolled
back. Deltas below are candidate minus the paired old actor; success and fall
deltas are percentage points.

| Arm | Round | Fraction | DQH success | DQH fall | Return | CBF/riser | D0 success | DQNH success | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A global | 1 | 0.25 | +2.08 | -2.08 | -0.228 | +0.115 | -4.17 | 0.00 | no |
| A global | 2 | 0.25 | +2.08 | -2.08 | +0.265 | +0.050 | -10.42 | +8.33 | no |
| A global | 3 | 0.50 | +6.25 | -6.25 | +0.324 | +0.111 | -4.17 | -4.17 | no |
| B fixed bank | 1 | 1.50 | -4.17 | +4.17 | -0.239 | +0.037 | 0.00 | +6.25 | no |
| B fixed bank | 2 | 1.50 | +2.08 | -2.08 | +0.487 | -0.021 | -4.17 | -2.08 | no |
| B fixed bank | 3 | 0.25 | +4.17 | -4.17 | +0.769 | -0.092 | -12.50 | -4.17 | no |

The B fixed-bank KL values for rounds 1-3 were respectively
`3.178e-4/3.337e-4`, `3.185e-4/3.355e-4`, and
`9.296e-6/9.262e-6` for D0/DQNH. All stayed below the independent `0.002`
budgets, so the predeclared monotone adaptation rule correctly left the anchor
weights at `0.02/0.01`. Task-level retention still failed. This means the
tested bank coverage, weights, and budgets are not a sufficient retention
contract; it does not show that a smaller fixed-bank KL generally guarantees
task success.

## Disjoint 512-episode/domain audits

Round 3 was fixed as the audit candidate for each arm because it had that
arm's largest formal DQH safe-score increase. The audit used 64 environments,
8 repeats, seeds 23000-23007, runtime CBF on, and no seeds used by screening or
the in-run gate. Both candidates shared the same cached old-policy replicates.

| Policy | Domain | Success | Fall | Return | CBF/riser |
|---|---|---:|---:|---:|---:|
| Old | D0 | 91.21% | 2.54% | 8.278 | 0.628 |
| Arm A candidate | D0 | 89.65% | 1.56% | 8.592 | 0.674 |
| Arm B candidate | D0 | 92.19% | 1.56% | 8.625 | 0.625 |
| Old | DQH | 87.70% | 12.11% | 8.413 | 0.797 |
| Arm A candidate | DQH | 89.65% | 10.35% | 8.752 | 0.711 |
| Arm B candidate | DQH | 88.09% | 11.91% | 8.557 | 0.741 |
| Old | DQNH | 86.33% | 13.67% | 8.106 | 0.836 |
| Arm A candidate | DQNH | 86.13% | 13.87% | 8.027 | 0.833 |
| Arm B candidate | DQNH | 89.06% | 10.94% | 8.555 | 0.778 |

The paired DQH deltas and percentile 95% intervals were:

| Arm | Metric | Mean | 95% lower | 95% upper |
|---|---|---:|---:|---:|
| A | Success | +1.95 pp | -0.78 pp | +4.30 pp |
| A | Fall | -1.76 pp | -4.10 pp | +0.78 pp |
| A | Return | +0.340 | -0.056 | +0.748 |
| A | CBF/riser | -0.086 | -0.191 | +0.001 |
| A | Safe score | +0.066 | -0.020 | +0.142 |
| B | Success | +0.39 pp | -2.73 pp | +3.71 pp |
| B | Fall | -0.20 pp | -3.71 pp | +3.32 pp |
| B | Return | +0.144 | -0.328 | +0.612 |
| B | CBF/riser | -0.056 | -0.207 | +0.066 |
| B | Safe score | +0.013 | -0.103 | +0.130 |

Neither arm has a positive task-improvement lower bound. Arm B also reaches
only 92.19% D0 success, below the predeclared 93.83% absolute retention floor;
its paired D0 non-inferiority lower bound is negative as well. Both final
decisions are therefore `rollback`, for the same exact reasons:

```text
target metrics show no strict improvement
target task metrics show no strict improvement
D0 retention bound violated
```

The final accepted actor tensors for both arms are byte-identical to the
deployed start actor (13 tensors, maximum absolute error 0). Runtime CBF stays
enabled. Because Arm B did not pass the Level A retention gate, the planned
3 base seeds x 5 hidden deployment contexts expansion was not run.

## Evidence boundary and next decision

The result is a clean negative finding for the tested calibration: fixed,
stage-balanced actor-only observation banks and finite KL anchors are
implemented correctly, but `beta_0=0.02`, `beta_N=0.01`, and KL budget `0.002`
did not produce a promotable DQH update. It would be selection-biased to tune
the budget after viewing these audit seeds and call that the same experiment.
A future protocol should predeclare a tighter task-calibrated bank-KL budget or
broader failure-boundary bank coverage, then use fresh selection and audit
seeds. No claim about 15 hidden contexts or real-robot improvement is made from
this result.

The compact machine-readable ledger is
`results/online/retention_v13/key_results.json` (SHA-256
`4ebe894ecb051457268aa1d5e1f4b299c43f90c318861a0e8cfc720e9a6d9d6e`).
It includes every formal round, both final audits, checkpoint identities,
paired intervals, bank manifests, and source-artifact checksums.

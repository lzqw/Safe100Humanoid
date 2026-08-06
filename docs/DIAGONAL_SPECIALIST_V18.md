# Diagonal Specialist Audit v18

This document freezes a new prospective evaluation protocol for the already
sealed Failure-Mode-Conditioned Brief PPO v17 actors. It does not rewrite the
v17 formal result: v17 remains a failed joint claim under its original `>2pp`
scene thresholds and macro-LCB gate.

## 中文摘要

后续实验收缩为三个互相独立的问题：

1. lateral specialist 是否提高冻结 lateral 场景的成功率；
2. CBF specialist 是否提高冻结 high-CBF/toe-riser 场景的成功率；
3. balance specialist 是否提高冻结 balance/contact/phase 场景的成功率。

每个问题单独成立或失败，不再要求三个场景同时通过，不计算 macro gate，
也不运行 off-diagonal 或 CBF-independence 实验。95% CI 必须报告，但不是
acceptance gate；也不再使用统一的 `>2pp` 门槛。

这套判据是在看到 v17 结果后提出的，因此不能用同一份 v17 审计数据事后改判。
v18 复用训练结束前已经封存的九个 actor，但必须使用新的 paired evaluation
seeds。新数据产生后不允许重新挑选 actor、场景或阈值。

## Why the actors are reused

The requested change concerns final evaluation, not the training algorithm.
The nine v17 jobs already satisfy the intended training structure:

- all start independently from the same frozen policy;
- three modes × adaptation seeds 42/142/242;
- one actor and one privileged critic per job;
- mode-specific scalar reward;
- raw policy action/log probability stored for PPO while runtime CBF executes
  the safe action;
- target-only candidate gates and a broad D0 catastrophic check;
- mode-specific failure precursors plus matched-success counterexamples;
- integer 70/15/15 start allocation of 44/10/10 for 64 environments;
- five brief PPO rounds with candidate fractions 0.5/1.0/1.5.

Those actors were sealed before either the v17 or v18 final audits. Reusing
them isolates the requested evaluation change and avoids outcome-driven
retraining. Their complete hashes and decisions are fixed by
`results/online/specialist_v17/formal/training_manifest.json`.

## Frozen contexts

Only the frozen base policy was used to select these contexts. Each calibration
used 512 episodes and passed success rate 70–85%, at least 100 falls, target
failure purity at least 60%, and second failure fraction at most 30%.

| Context | Base success | Falls | Target failure | Second failure | Parameter SHA-256 prefix |
|---|---:|---:|---:|---:|---|
| lateral | 78.906% | 108 | 85.185% | 9.259% | `e79dad659a30` |
| CBF | 78.906% | 108 | 71.296% | 19.444% | `da07fd6cc606` |
| balance | 77.344% | 116 | 61.207% | 22.414% | `5bf28258852b` |

## Fresh paired audit

For each specialist and each adaptation seed, compare the common base actor and
the sealed final actor using identical initial states and simulator randomness:

- 512 paired episodes in that specialist's own frozen target context;
- 256 paired D0 episodes as a broad safety sanity check;
- runtime CBF enabled for both policies;
- audit seed `3,100,000` with non-overlapping deterministic seed ranges;
- 10,000 two-level paired-bootstrap samples with seed `4,000,000`.

The interval resamples adaptation seeds and then paired episode deltas within
each selected seed. Baseline and final outcomes are never resampled
independently.

The audit intentionally omits:

- all six off-diagonal cells;
- a three-scene macro average or macro confidence gate;
- a conjunctive requirement that all specialists pass;
- a uniform two-percentage-point minimum;
- filter-free or CBF-independence evaluation.

## Independent acceptance rule

For each mode `m`, define target success and fall deltas as final minus base.
That mode's claim passes exactly when all four conditions hold:

1. mean paired target success delta is strictly positive;
2. at least two of three adaptation-seed success deltas are strictly positive;
3. mean paired target fall delta is at most `+0.03`;
4. mean paired D0 success delta is at least `-0.05`.

The target success/fall 95% intervals are mandatory descriptive evidence, but
their lower bounds are not gates. D0 is a sanity safeguard, not the main
algorithmic endpoint.

There is no global `final_claim_passed` value. The output contains three
independent `claim_passed` values, and a failure in one mode cannot invalidate
evidence for another.

## Integrity boundary

The machine-readable protocol is
`results/online/specialist_v18/protocol.json`. A formal run must:

- execute from the exact Git commit that freezes this protocol;
- have a clean tracked worktree and index;
- match the common base checkpoint, context, compact training-manifest,
  checkpoint, summary, final-actor, and evaluation-source hashes;
- write 6,912 paired rows (`3 modes × 3 seeds × (512 target + 256 D0)`);
- preserve the v17 negative audit unchanged.

The formal launcher is:

```text
SAFE100_DIAGONAL_V18_PROTOCOL_COMMIT=<frozen-commit> \
  bash experiments/scripts/run_specialist_diagonal_audit_v18.sh
```

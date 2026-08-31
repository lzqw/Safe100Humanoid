# Diagonal Specialist Audit v18: formal evidence

**Final status: two independent specialist claims passed and one failed.**
Lateral and balance pass the frozen v18 point-estimate gates; CBF fails. There
is deliberately no joint three-scene claim, macro gate, or off-diagonal result.

This is a fresh audit of the nine actors sealed by the v17 training protocol.
It does not retroactively change the historical v17 verdict, which remains a
failed joint claim under its original `>2pp` and macro-LCB criteria.

## 中文结论

v18 将最终实验严格收缩为三个彼此独立的对角线问题，并用新的 audit seeds
重新运行 paired evaluation：

- **lateral specialist：通过。** 成功率平均提高 **+3.060pp**，3/3 adaptation
  seeds 为正，跌倒率下降 3.060pp，D0 成功率变化 +1.042pp；
- **CBF specialist：未通过。** 成功率平均变化 **-0.651pp**。虽然 2/3 seeds
  为正，跌倒和 D0 门槛也通过，但平均成功率没有提高；
- **balance specialist：通过。** 成功率平均提高 **+1.107pp**，3/3 seeds
  为正，跌倒率下降 1.107pp，D0 成功率变化 +0.391pp。

三个 target-success 95% CI 都跨过 0。按照预先冻结的 v18 规则，CI 必须报告但
不是 gate，因此 lateral 和 balance 的“通过”表示满足独立点估计规则，不表示
各自达到了 `LCB95 > 0` 的统计显著性。

## Independent results

All values are paired final-minus-base changes. Seed deltas are ordered
`42 / 142 / 242`.

| Specialist | Base SR | Final SR | Success delta (95% CI) | Seed deltas | Fall delta | D0 success | Claim |
|---|---:|---:|---:|---|---:|---:|---|
| lateral | 75.651% | 78.711% | **+3.060pp** [-0.195, +6.185] | +0.391 / +4.102 / +4.688pp | -3.060pp | +1.042pp | **PASS** |
| CBF | 78.255% | 77.604% | **-0.651pp** [-6.120, +3.255] | +1.758 / -6.250 / +2.539pp | +0.651pp | -0.260pp | **FAIL** |
| balance | 79.427% | 80.534% | **+1.107pp** [-0.521, +2.799] | +0.781 / +0.977 / +1.563pp | -1.107pp | +0.391pp | **PASS** |

The CBF result is not rescued by its two positive seeds: the frozen rule also
requires the aggregate paired success point estimate to be strictly positive.
Conversely, a CBF failure cannot invalidate the lateral or balance claims.

## Frozen acceptance rule

Each specialist is judged only in its own target scene. A claim passes when:

1. mean paired target success delta is strictly positive;
2. at least two of three adaptation-seed deltas are strictly positive;
3. mean target fall increase is at most `+3pp`;
4. mean D0 success delta is at least `-5pp`.

The protocol does not require a `>2pp` improvement or a confidence-interval
lower bound above zero. It does not compute a macro average, require all three
claims to pass, run off-diagonal cells, or run filter-free/CBF-independence
checks. See [`../protocol.json`](../protocol.json) and
[`docs/DIAGONAL_SPECIALIST_V18.md`](../../../../docs/DIAGONAL_SPECIALIST_V18.md).

## Evaluation protocol

- sealed v17 actors: three modes × adaptation seeds 42/142/242;
- one common frozen base actor for all nine comparisons;
- 512 paired target episodes per adaptation seed;
- 256 paired D0 episodes per adaptation seed;
- runtime CBF enabled for base and final actors;
- formal audit seed `3,100,000`;
- 10,000 two-level paired-bootstrap samples, seed `4,000,000`;
- exact protocol source commit `108e6013d8d1282b095dafcbb14fa16d73fabfe7`;
- 6,912 paired rows in 18 declared groups;
- 96 unique raw JSON blocks and 96 matching raw CSV blocks;
- no infrastructure retry or partial formal result.

## Evidence

- [`diagonal_audit_summary.json`](diagonal_audit_summary.json): complete audit
  summary, source/input hashes, paired signatures, raw aggregate evaluations,
  intervals, gates, and independent conclusions.
- [`paired_episode_metrics.csv`](paired_episode_metrics.csv): all 6,912 paired
  target and D0 rows.
- [`verification.json`](verification.json): independent reconstruction of all
  group rates, deltas, bootstrap intervals, gates, and pass/fail lists from the
  paired CSV.
- [`REQUIREMENTS_AUDIT.md`](REQUIREMENTS_AUDIT.md): requirement-by-requirement
  protocol and outcome audit.
- [`RUN_PROVENANCE.md`](RUN_PROVENANCE.md): protocol revision, smoke, formal-run,
  and raw-artifact provenance.
- [`diagonal_audit.log`](diagonal_audit.log): complete formal launcher output.
- [`SHA256SUMS`](SHA256SUMS): hashes of the published evidence files.

Key immutable hashes:

| Artifact | SHA-256 |
|---|---|
| Protocol | `48d9ad0ab3232927601b5a6c69c3f362bbc1184c7bdc59e7d199e6084eb66d2e` |
| Complete audit summary | `e4c43642179bc8b3a465fb7d00c46ba843176bd9b69c1a2c3ff7f4d9d59bcb78` |
| Paired episode CSV | `64ad56aab3acca6b73302d5a6e3a69fa4df23dbc134e49d52aba6400269a00db` |
| Formal audit log | `dc8e9040ed9ee08f31dbab4f4cd9d96e83f1c01b931edcbfdc6ed2761439e73c` |

## Verification

The exact frozen source passed:

```text
python -m pytest -q experiments/tests/test_diagonal_specialist_audit.py \
  experiments/tests/test_cbf_math.py \
  experiments/tests/test_online_refinement.py
78 passed in 10.13s
```

The independent verifier then reconstructed all 18 CSV groups, all per-seed
rates/deltas, all twelve hierarchical intervals, and all three gates exactly.

These results provide scene-specific simulation evidence with runtime CBF. They
do not establish real-robot improvement, filter-free behavior, or statistical
significance under an LCB-above-zero criterion.

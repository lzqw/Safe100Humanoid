# Failure-Mode-Conditioned Brief PPO v17: formal evidence

**Final status: the predeclared claim did not pass.** This directory preserves
the completed formal experiment and its negative result without changing the
training gates, evaluation thresholds, or seed set after seeing the data.

## 中文结论

v17 已按冻结协议完成 3 个 specialist × 3 个 adaptation seed 的训练，以及独立的
diagonal/D0/off-diagonal 最终审计。协议与数据完整性检查均通过，但性能结论未通过：

- lateral diagonal 平均成功率提升 **+1.758 个百分点**，3/3 seeds 为正，但没有达到严格的 `> 2pp`；
- CBF diagonal 平均成功率变化 **-0.521 个百分点**，只有 1/3 seeds 为正；
- balance diagonal 平均成功率提升 **+0.781 个百分点**，2/3 seeds 为正，但没有达到严格的 `> 2pp`；
- 三个 specialist 的 diagonal 跌倒率门槛和 D0 保持门槛全部通过；
- hierarchical macro 成功率均值为 **+0.673pp**，95% 区间为
  **[-1.215pp, +2.344pp]**，下界不大于 0。

因此 `final_claim_passed=false`。训练期的正向诊断结果不替代这份独立最终审计。

## Frozen inputs and calibration

All nine jobs started independently from the same frozen policy file:

- base checkpoint SHA-256: `cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8`
- initial actor SHA-256: `19432232a4acfebb2a838eb3c9449393a47cfd3154e93f68171dcf4d1aaf47b1`
- adaptation seeds: `42`, `142`, `242`

Each context was selected using only the frozen base policy and 512 episodes.

| Mode | Base SR | Falls | Target failure | Second failure | Parameter SHA-256 |
|---|---:|---:|---:|---:|---|
| lateral | 78.906% | 108 | 85.185% | 9.259% | `e79dad659a30...173bf85` |
| CBF | 78.906% | 108 | 71.296% | 19.444% | `da07fd6cc606...0aae3d1` |
| balance | 77.344% | 116 | 61.207% | 22.414% | `5bf28258852b...c811e81` |

All three satisfy SR 70–85%, at least 100 falls, target purity at least 60%,
and second-largest failure fraction at most 30%. The full calibration summaries
and frozen contexts are in [`../calibration`](../calibration) and
[`../contexts`](../contexts).

## Formal audit result

All deltas below are paired percentage-point changes, final minus the common
base policy. Diagonal results use 512 paired episodes per adaptation seed; D0
uses 256. The seed deltas are ordered `42 / 142 / 242`.

| Specialist | Diagonal success | Seed success deltas | Diagonal fall | D0 success | Scene gate |
|---|---:|---|---:|---:|---|
| lateral | **+1.758pp** | +2.148 / +1.367 / +1.758pp | -1.758pp | +0.260pp | **FAIL** (`>2pp` not met) |
| CBF | **-0.521pp** | -3.320 / -0.586 / +2.344pp | +0.521pp | +1.432pp | **FAIL** (`>2pp` and 2/3-positive not met) |
| balance | **+0.781pp** | +0.977 / +1.367 / 0.000pp | -0.846pp | +0.391pp | **FAIL** (`>2pp` not met) |

The report-only off-diagonal success deltas were:

| Trained specialist | lateral context | CBF context | balance context |
|---|---:|---:|---:|
| lateral | diagonal | +1.432pp | +1.563pp |
| CBF | +0.391pp | diagonal | +1.432pp |
| balance | +0.911pp | +4.427pp | diagonal |

These off-diagonal values were never training gates and do not rescue a failed
diagonal claim.

## Sealed training runs

`none` means that no candidate strictly improved the target score while meeting
the frozen gates; the previous accepted actor was retained.

| Mode | Seed | Accepted fraction by round | Final actor SHA-256 prefix |
|---|---:|---|---|
| lateral | 42 | none / 0.5 / 1.0 / none / none | `8f668fa9b6e4` |
| lateral | 142 | none / 1.0 / 1.5 / 1.0 / 0.5 | `79f7bca88461` |
| lateral | 242 | 1.0 / none / 1.5 / 0.5 / 1.0 | `8b98b2de1055` |
| CBF | 42 | none / none / 0.5 / none / 1.5 | `7e73454e449f` |
| CBF | 142 | 1.0 / none / 1.5 / 1.0 / 0.5 | `555d48ca1711` |
| CBF | 242 | 1.0 / 1.0 / 1.5 / 1.5 / none | `18f47c598733` |
| balance | 42 | 1.0 / none / 1.0 / 1.0 / none | `9d68f945774e` |
| balance | 142 | 1.0 / 1.0 / 1.5 / 0.5 / 0.5 | `4e71a82c6a07` |
| balance | 242 | 1.5 / 1.5 / 1.0 / 1.0 / 1.0 | `431cb5f08cab` |

The machine-readable [`training_manifest.json`](training_manifest.json)
contains the full hashes, all 135 candidate evaluations, all 45 round decisions,
D0 checks, KL values, final diagnostics, and hard-case-bank audits. Large model
checkpoints are intentionally not duplicated in Git; their SHA-256 values are
preserved in the manifest and independent audit.

## Evidence files

- [`audit/final_audit_compact.json`](audit/final_audit_compact.json): concise,
  machine-readable final verdict, matrix, gates, protocol, and provenance.
- [`audit/final_audit_summary.json`](audit/final_audit_summary.json): complete
  final audit summary, including aggregate raw evaluations.
- [`audit/paired_episode_metrics.csv`](audit/paired_episode_metrics.csv): all
  **11,520** paired episode rows used to compute the matrix and bootstrap.
- [`training_manifest.json`](training_manifest.json): compact record of the nine
  sealed training runs.
- [`REQUIREMENTS_AUDIT.md`](REQUIREMENTS_AUDIT.md): requirement-by-requirement
  protocol and outcome audit.
- [`INTERRUPTION_PROVENANCE.md`](INTERRUPTION_PROVENANCE.md): preserved record of
  the excluded CBF/seed242 infrastructure failure and whole-job retry.
- [`SHA256SUMS`](SHA256SUMS): hashes for the published evidence package.

Key evidence hashes:

| Artifact | SHA-256 |
|---|---|
| Complete audit summary | `8a3d4562c507de60e0a61b2608bf5197fb9b0cc590f0642c7a12fd6a59d1848c` |
| Paired episode CSV | `81a4b19eee85ddcfb7a1a0f4251c2461919ed81993cc86f5491a7908fe478054` |
| Compact audit | `62f3c84c5085e51c771ef67208e74a604e008f54056d34a0f090685ca816a458` |
| Training manifest | `6d8149d9a47ec786405feab2503fe5e2b21e64df3d3e943911c5f92e0044494d` |

The paired CSV has exactly 11,520 data rows and its recomputed hash matches the
hash embedded in the complete audit summary. The audit produced 132 unique raw
JSON/CSV blocks; off-diagonal baseline evaluations reuse the first two blocks of
the corresponding 512-episode diagonal baseline.

## Verification

The final v17 source passed:

```text
python -m pytest -q experiments/tests/test_cbf_math.py \
  experiments/tests/test_online_refinement.py
69 passed in 10.47s
```

This package supports the statement that the frozen v17 experiment was executed
and audited as declared. It does **not** support a claim that v17 achieved the
required specialist improvement.

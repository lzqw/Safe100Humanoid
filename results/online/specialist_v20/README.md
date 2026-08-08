# Fixed-Budget Observable Failure-Conditioned Brief PPO v20

## English

### Final status

The prospectively frozen v20 experiment is complete. Both specialists ran all
five adaptation seeds for exactly eight online rounds, received separate fresh
paired audits, and were reconstructed by the independent verifier. **Neither
specialist passed its independent claim gate.** There is no joint claim.

| Specialist | Independent result | Target success delta, pp (95% CI) | Positive seeds | Target fall delta, pp (95% CI) | D0 success delta, pp (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: |
| lateral | **FAIL** | +0.273 [−2.695, +3.242] | 2/5 | −0.313 [−3.359, +2.617] | −1.484 [−3.750, +0.783] |
| contact_stability | **FAIL** | +0.156 [−2.930, +3.359] | 2/5 | −0.156 [−3.438, +2.852] | −0.703 [−3.750, +2.268] |

Both modes passed the mean-success, target-fall, and D0-retention point gates,
but failed the prospectively declared requirement that at least four of five
adaptation-seed target deltas be positive. Neither 95% lower confidence bound
was positive, so neither mode has strong statistical evidence under the
protocol's report-only criterion.

The authoritative machine-readable result is
[`formal_results.json`](formal_results.json). Each audit contains 512 paired
target episodes and 256 paired D0 episodes per adaptation seed (3,840 rows per
mode), with 10,000 hierarchical paired-bootstrap samples.

### Full per-seed outcomes

Target and D0 values below are from the formal audit, not the training-time
diagnostic evaluations. `Retained` is diagnostic only and was never a run
validity gate.

| Lateral seed | Retained / 8 | Baseline SR, % | Final SR, % | Target ΔSR, pp | Target Δfall, pp | D0 ΔSR, pp | Repairs | Regressions | Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 73 | 4 | 72.852 | 75.195 | +2.344 | −2.344 | −1.172 | 90 | 78 | +12 |
| 173 | 3 | 69.141 | 72.656 | +3.516 | −3.516 | −2.344 | 100 | 82 | +18 |
| 273 | 4 | 75.586 | 73.047 | −2.539 | +2.539 | −2.344 | 80 | 93 | −13 |
| 373 | 2 | 75.781 | 74.414 | −1.367 | +1.172 | −2.344 | 72 | 79 | −7 |
| 473 | 2 | 75.000 | 74.414 | −0.586 | +0.586 | +0.781 | 84 | 87 | −3 |
| **Aggregate** | **15** | — | — | **+0.273** | **−0.313** | **−1.484** | **426** | **419** | **+7** |

| Contact seed | Retained / 8 | Baseline SR, % | Final SR, % | Target ΔSR, pp | Target Δfall, pp | D0 ΔSR, pp | Repairs | Regressions | Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 73 | 5 | 73.047 | 72.266 | −0.781 | +0.781 | +1.563 | 95 | 99 | −4 |
| 173 | 6 | 69.727 | 74.023 | +4.297 | −4.297 | −1.563 | 114 | 92 | +22 |
| 273 | 4 | 69.922 | 70.508 | +0.586 | −0.586 | +1.953 | 109 | 106 | +3 |
| 373 | 2 | 71.094 | 68.750 | −2.344 | +2.344 | −0.781 | 96 | 108 | −12 |
| 473 | 5 | 72.656 | 71.680 | −0.977 | +0.977 | −4.688 | 108 | 113 | −5 |
| **Aggregate** | **22** | — | — | **+0.156** | **−0.156** | **−0.703** | **522** | **518** | **+4** |

The small aggregate gains came from extensive reshuffling: lateral had 426
repairs and 419 regressions, while contact-stability had 522 repairs and 518
regressions. The seed-level instability, rather than the point estimate, is why
both independent claims fail.

### Training and integrity diagnostics

- All 10 runs completed exactly eight rounds; accepted-update counts were
  `4/3/4/2/2` for lateral and `5/6/4/2/5` for contact-stability.
- The combined tables contain 90 aligned round rows, 240 candidate rows, and
  160 replay rows. All 160 replay-bank transactions committed, every exact
  matched-pair count was 12, maximum joint marginal imbalance was 0, and no D0
  rollback occurred.
- Maximum pre-update PPO KL was `5.009e-4` for lateral and `5.077e-4` for
  contact-stability, below both the 0.003 target and 0.01 hard ceiling.
- Legacy 405-column actor drift was exactly 0 in every round. Final new-column
  RMS values were `1.53e-5…2.66e-5` for lateral and `1.47e-5…2.48e-5` for
  contact-stability.
- The expanded initial actor hash was
  `f8e27e9b3bb92dd33a460f38ecbd72f8b0fe03809683f697d54d3549626d69bb`;
  per-seed final actor and checkpoint hashes are in each
  [`training/`](training/) summary.

### Mechanism-telemetry disclosure

Representative identities were fixed as the lowest formal target
failure-to-success pair index for each seed. On this GPU simulator, replaying
the same actor and the same initial-state signature did not always reproduce
the terminal outcome. The original collector exposed this on its first
attempt; no formal audit was rerun and no trace was retried until it matched.

The frozen disclosure wrapper preserved/reused existing first attempts and
made exactly one attempt for each remaining policy role. Actor hashes and
initial-state signatures matched the formal audit for all 20 traces. Outcome
fields matched in 7/10 lateral traces and 5/10 contact traces. Therefore the
mechanism curves are **descriptive same-initial-state first-attempt replays**;
only the paired audit CSVs are authoritative for performance outcomes. See the
[`lateral`](audit/lateral/mechanism_selection.json) and
[`contact`](audit/contact_stability/mechanism_selection.json) selection files
for every identity, hash, outcome comparison, and reuse flag.

### Evidence map

- Independent audits and verifications:
  [`lateral`](audit/lateral/) and
  [`contact_stability`](audit/contact_stability/).
- Published training matrices: [`curves/`](curves/) and compact per-seed
  [`training/`](training/) tables.
- Paper figures: 22 PNG plus 22 PDF in [`figures/`](figures/). A second
  fixed-timestamp generation produced identical hashes for all 44 files.
- Full external provenance: [`training_manifest.json`](training_manifest.json)
  hashes 510 training/audit artifacts totaling 5.96 GB; large checkpoints and
  raw evaluator blocks are not duplicated in Git.
- Run-log provenance: [`run_log_manifest.json`](run_log_manifest.json) hashes
  all 26 v20 logs; compact logs are included under [`logs/`](logs/).
- Requirements and execution history:
  [`REQUIREMENTS_AUDIT.md`](REQUIREMENTS_AUDIT.md) and
  [`RUN_PROVENANCE.md`](RUN_PROVENANCE.md).
- Historical values are diagnostic only:
  [`historical_reference.csv`](historical_reference.csv). Contexts, methods,
  seed counts, and audit randomness differ, so they are not paired
  cross-version comparisons.

### Hash anchors and claim boundaries

- Frozen formal protocol commit: `1ded5b84f1c4b8605fd285ef3138c0363db20ee4`;
  protocol SHA-256: `74242b1131499f7163ef1985d25a94ca06b0758f6a3d5f2dea9f56ac200c28da`.
- Audit-only amendment commit: `1428f57ff58ad8bb76da8e6c2fd1f5a22d8bd21c`;
  amendment SHA-256: `5700b77302093ce708a9be2aa264e36267f7220d05e5037ca4eff109c85377dc`.
- Lateral summary / paired CSV SHA-256:
  `9dff606fad5f68b052c3e4b7372bfc4b179ff9616b942f3a4e54e1e9ed15a39d` /
  `170eb8ce10aeea0924880661ee3d0405472cb35f16225e0d06ab62381d4ef45e`.
- Contact summary / paired CSV SHA-256:
  `8bf00ce908c59147f834fc7e3fd7b560c315c69ea0767d6646f7a9afccc18bb1` /
  `e258f16a9049dac85439f2c0e494e318a6ef0979d2580f9d7ded8df67c1b7ae4`.

v20 defines no joint-specialist, macro-average, off-diagonal, filter-free, or
CBF-independence claim. Runtime CBF remained on. A failed point gate is a valid
scientific result and was not rerun. The pull request remains Draft.

## 中文

### 最终结论

v20 正式实验已经完成。两个 specialist 都独立完成了 5 个 seed × 固定 8 轮训练、
fresh paired audit 和独立 CSV 重建验证。**lateral 与 contact-stability 均未通过各自
独立结论门槛**，并且不存在 joint claim。

- lateral：目标成功率平均 `+0.273 pp`，95% CI
  `[−2.695, +3.242] pp`；只有 `2/5` seed 为正，因此 FAIL。目标跌倒率
  `−0.313 pp`，D0 成功率 `−1.484 pp`，这两个门槛通过。
- contact-stability：目标成功率平均 `+0.156 pp`，95% CI
  `[−2.930, +3.359] pp`；同样只有 `2/5` seed 为正，因此 FAIL。目标跌倒率
  `−0.156 pp`，D0 成功率 `−0.703 pp`，这两个门槛通过。
- 两个模式的 LCB95 都不大于 0，因此也没有达到协议中的 strong statistical
  evidence 标准。

结果不是因为训练无效：10 个 run 全部完成固定 8 轮，保留更新数允许为 0–8；其中
lateral 的 373/473 和 contact 的 373 都只有 2 次 retained update，仍按协议进入
正式审计。旧 405 输入列漂移始终严格为 0，replay exact pair 和 joint marginal
检查全部通过，KL 远低于 0.01 ceiling。

### 结果应如何解释

两个模式的 aggregate point estimate 都略为正，但跨 seed 不稳定。lateral 有
426 次 repair、419 次 regression（净 +7）；contact 有 522 次 repair、518 次
regression（净 +4）。因此这次结果更接近“重新分配哪些 episode 成功”，尚不足以
证明稳定改进。上面的完整逐 seed 表不会隐藏失败 seed。

机制曲线也有明确边界：GPU 仿真器在相同 actor hash 和相同 initial-state signature
下并不总能重放相同终局。lateral 只有 7/10、contact 只有 5/10 trace 的 outcome
与 formal audit 一致。所有 trace 都是固定 identity 的第一次尝试，没有为了匹配
结果而重试；所以机制曲线只作描述，正式性能结论只来自 paired audit CSV。

代码、逐 seed 表、完整 paired CSV、verification、PNG/PDF 图、日志/原始 artifact
哈希和 provenance 都在本目录。`SHA256SUMS` 可验证 Git 中的整个 v20 evidence
package；5.96 GB 的 checkpoint/raw 数据由 `training_manifest.json` 哈希寻址，
避免把大文件直接塞入仓库。v17/v18/v19 历史证据保持原样，PR 继续保持 Draft。

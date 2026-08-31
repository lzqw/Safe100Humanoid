# v22 Effect-First Development Result

## English

### Final status

The single prospectively frozen `L_effect` adaptation is complete. It ran the
fixed eight-round budget and the fresh paired final test, but **failed the
development gate**. Contact calibration and adaptation were therefore not run.
This is a development result, not a formal multi-context generalization claim;
v17--v21 evidence remains unchanged.

| Fresh evaluation | Base | Validation-selected best | Delta, pp | Report-only paired 95% CI, pp | Point gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Target success, 512 pairs | 70.117% (359/512) | 69.531% (356/512) | **-0.586** | [-5.859, +4.883] | **FAIL** (needs at least +3) |
| Target fall, 512 pairs | 29.883% (153/512) | 30.469% (156/512) | +0.586 | [-4.688, +5.664] | PASS (at most +1); strict-zero FAIL |
| D0 success, 256 pairs | 92.969% (238/256) | 93.750% (240/256) | +0.781 | [-3.135, +4.688] | PASS (at least -5) |
| D0 fall, 256 pairs | 1.953% (5/256) | 0.391% (1/256) | -1.563 | [-3.516, 0.000] | Report only |

The four primary values are the first two rows' base and best rates. On target,
93 base failures were repaired, but 96 base successes regressed, for a net
change of -3 episodes. The paired bootstrap interval was descriptive only and
did not decide the gate.

### What training did

Base-only calibration froze the first qualifying candidate, seed `51011`: 352
successes, 160 failures, 160 falls, and 92.5% target-failure purity in 512
episodes. Revision 1 then exposed an inapplicable sign-diversity quota and
stopped before any PPO update. Its stop evidence is preserved. Revision 2 used
fresh execution seeds and the mechanism-aligned lateral balance profile.

Revision 2 completed 8/8 rounds and accepted rounds 3, 4, and 7. The fixed
validation monitor selected round 3:

| Validation checkpoint | Success | Fall | Delta success vs pi0 | Delta fall vs pi0 | Selected |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 (`pi0`) | 64.453% | 35.547% | 0.000 pp | 0.000 pp | No |
| 3 | 68.750% | 31.250% | +4.297 pp | -4.297 pp | **Yes** |
| 4 | 64.844% | 35.156% | +0.391 pp | -0.391 pp | No |
| 7 | 66.016% | 33.984% | +1.563 pp | -1.563 pp | No |

The apparent validation improvement did not transfer to the inaccessible final
conditions. The protocol forbids a poor-outcome rerun, so this negative result
is final for `L_effect` and blocks `C_effect`.

### Integrity and evidence

- All 16 rollout-bank transactions committed. Every batch used 12 exact pairs
  with stage `4/4/4`, support foot `6/6`, growth `6/6`, and maximum marginal
  imbalance 0. Direction signs remained exact-match diagnostics, not quota
  axes.
- Legacy actor input-column drift was exactly 0; maximum pre-update minibatch KL
  was `4.678e-4`; there were no non-finite rounds, KL early stops, or D0
  rollbacks.
- [`training_summary.json`](final/L_effect/training_summary.json) is the compact
  eight-round record. The exact authoritative
  [`final_test.json`](final/L_effect/final_test.json), all 768 paired rows in
  [`paired_episode_metrics.csv`](final/L_effect/paired_episode_metrics.csv),
  and an independent [`verification.json`](final/L_effect/verification.json)
  are committed. Every reconstruction check passes.
- [`figures/`](figures/) contains exactly four categories in PNG and PDF:
  validation learning curve, base versus best final result, repair versus
  regression, and failure-specific telemetry.
- Large checkpoints, simulator raw files, telemetry CSV, calibration raw data,
  and logs remain external. The
  [`external_artifact_manifest.json`](external_artifact_manifest.json) binds
  132 files totaling 823,155,388 bytes by SHA-256. [`SHA256SUMS`](SHA256SUMS)
  binds the compact Git package.

Boundary anchors: protocol commit
`399908adcd61bcaa00fe669fa018b02768dbc3f9`, protocol SHA-256
`cff26e1e8768e79256af1fdf7ee8d04d267e164b1b2a5d09637ce0cfc7309dee`,
context SHA-256
`650a97519168382bf4f7fc45580fa179cb3c51a1f18195f4850c5667d6f0d6a7`,
base checkpoint SHA-256
`cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8`,
and selected best checkpoint SHA-256
`e3676de9e5507db596360de25f0968a05553d8a937d9007431462cf0f9e179e2`.

## 中文

### 最终结论

预先冻结的单次 `L_effect` adaptation 已完成固定 8 轮训练和 fresh paired
最终测试，但**未通过开发门槛**，因此协议要求停止，不能启动 contact 校准或训练。
这是 effect-first 开发结果，不是多场景泛化结论；v17--v21 证据未被修改。

- fresh 512-pair target：成功率从 `70.117% (359/512)` 降到
  `69.531% (356/512)`，即 `-0.586 pp`，未达到 `+3 pp` 门槛；report-only
  95% CI 为 `[-5.859, +4.883] pp`。
- target fall 从 `29.883%` 升到 `30.469%`，即 `+0.586 pp`，通过
  `+1 pp` 点门，但不通过更严格的零增长解释。
- fresh 256-pair D0：成功率从 `92.969%` 升到 `93.750%`，即
  `+0.781 pp`，通过 retention 门槛；fall 从 `1.953%` 降到 `0.391%`。
- target 中有 93 个 failure 被修复，同时有 96 个 success 退化，净值为
  `-3` 个 episode。CI 只作描述，不参与门控。

训练本身按协议有效。base-only calibration 的第一个合格候选是 seed `51011`：
512 次中 352 success、160 failure、160 fall，目标失败纯度 92.5%。revision 1
在任何 PPO update 前暴露旧 sign quota 问题并停止；revision 2 使用全新执行 seed
和修正后的固定方向配额，完成 8/8 轮，round 3/4/7 被接受。固定 validation
选择 round 3（success `+4.297 pp`、fall `-4.297 pp`），但该提升没有迁移到
不可见的 final 条件。协议禁止因为结果差而重跑，所以 lateral 的负结果是本次
revision 的最终结论，contact 保持未运行。

完整性检查均通过：16/16 bank transaction 提交，每个 batch 都有 12 个 exact
pair，stage/support/growth 边际分别为 `4/4/4`、`6/6`、`6/6`，最大不平衡为 0；
旧 405 个 actor 输入列漂移严格为 0，最大 pre-update KL 为 `4.678e-4`，没有
non-finite、KL early stop 或 D0 rollback。

Git 中包含紧凑训练摘要、权威 final aggregate、768 行 paired CSV、独立重建验证、
四类 PNG/PDF 图和 `SHA256SUMS`。823 MB 的 checkpoint/raw/telemetry/log 数据没有
塞进仓库，而是由 132 条 SHA-256 external manifest 记录绑定。Draft PR 保持
Draft，不把这次负结果包装成成功 claim。

# v23 CBF-Proximal Online Refinement Result

## English

### Final status

The single prospectively frozen lateral adaptation is complete. It used the
fixed eight-round budget and the fixed round-8 actor, followed by a fresh
paired base-versus-final evaluation. The method **did not pass the development
gate**, because target success decreased by 0.391 percentage points instead of
improving by at least 3 points. The contact context was therefore not run.

This is a single-context development result, not a multi-context or
generalization claim. The negative result is final for this frozen execution;
there was no outcome-dependent rerun, checkpoint selection, or performance
rollback. All v17--v22 evidence remains unchanged.

| Fresh paired evaluation | Base | Fixed round 8 | Delta, pp | Report-only paired 95% CI, pp | Point gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Target success, 512 pairs | 69.336% (355/512) | 68.945% (353/512) | **-0.391** | [-5.864, +5.078] | **FAIL** (needs at least +3) |
| Target fall, 512 pairs | 30.664% (157/512) | 31.055% (159/512) | +0.391 | [-4.888, +5.664] | PASS (at most +1) |
| D0 success, 256 pairs | 92.578% (237/256) | 94.531% (242/256) | +1.953 | [-1.953, +5.859] | PASS (at least -5) |
| D0 fall, 256 pairs | 0.781% (2/256) | 2.734% (7/256) | +1.953 | [0.000, +4.297] | Report only |

On target, 93 base failures were repaired while 95 base successes regressed,
for a net change of -2 episodes. On D0, there were 16 repairs and 11
regressions, for a net change of +5. Confidence intervals are descriptive and
were not gates. Target mean return changed from `6.4576` to `6.5257`, while
runtime-CBF interventions per riser changed from `0.9447` to `0.9914`.

### What was tested

- One original-interface actor (405 observations) and one privileged critic
  (838 observations) were warm-started from the retained v13 checkpoint.
- The environment executed runtime-CBF-filtered actions. PPO retained the raw
  sampled action and its behavior-Gaussian log probability; routing audits were
  exact.
- Each of eight rounds collected one on-policy batch with 64 environments and
  1,024 steps. The actor used clipped PPO plus analytic forward
  `KL(pi_theta || pi_k)` to a freshly frozen round-start reference. The critic
  had one value loss and a separate optimizer.
- The actor learning rate was `5e-6`, clip ratio `0.05`, KL coefficient `0.5`,
  target KL `0.003`, and hard KL ceiling `0.01`. Actor and critic each used at
  most two epochs; log standard deviation was frozen.
- Specialist rewards, success/failure banks, state restart, grouped
  advantages, dual rollouts, candidate screens, validation selection, and
  training performance gates were absent. Ordinary CBF intervention was
  telemetry, not failure.

The rollout rates below are training telemetry only. They did not select,
stop, accept, reject, or roll back a policy.

| Round | Status | Moving forward KL | Rollout success | Rollout fall |
| ---: | --- | ---: | ---: | ---: |
| 1 | updated | 0.001690 | 46.154% | 53.846% |
| 2 | updated | 0.001044 | 41.441% | 58.559% |
| 3 | updated | 0.000583 | 44.762% | 55.238% |
| 4 | updated | 0.000666 | 45.872% | 54.128% |
| 5 | updated | 0.000604 | 43.689% | 56.311% |
| 6 | updated | 0.000580 | 37.037% | 62.963% |
| 7 | updated | 0.000580 | 36.697% | 63.303% |
| 8 | updated | 0.000578 | 40.000% | 60.000% |

### Integrity and evidence

- All 8/8 rounds updated. There were zero hard or performance rollbacks, zero
  action-routing error, zero policy-storage error, and no KL early stop. The
  maximum moving forward KL was `0.001689753`, well below the `0.01` hard
  ceiling. Every round-start actor hash equals the preceding round-end hash.
- The final actor is unconditionally round 8: actor SHA-256
  `a08a2ac986735dfe7bc1fedf54b8c33782a9d45e2162c1eb8382e692136aa5e7`;
  external checkpoint SHA-256
  `d71cff3df1f576670e4de291b4e6e684f928a05503c245baadf3589b6a6a2004`.
- The final audit contains 512 target pairs and 256 D0 pairs. Base and final
  initial-state signatures match for every batch; both use deterministic policy
  means, runtime CBF, and the original actor interface. All 768 episode pairs
  are committed in
  [`paired_episode_metrics.csv`](final/paired_episode_metrics.csv).
- [`training_summary.json`](training/training_summary.json),
  [`round_metrics.json`](training/round_metrics.json), the authoritative
  [`final_test.json`](final/final_test.json), and an independent reconstruction
  [`verification.json`](final/verification.json) are committed. Two figure
  categories are available in PNG and PDF under [`figures/`](figures/).
- The external inventory binds 61 files totaling 254,687,686 bytes, including
  the base checkpoint, all round checkpoints, raw isolated evaluations, and
  logs. Its SHA-256 is
  `b77a12325d11cbffd98360c8a147de3fe002d32451322d11e1d7f099d2a590fc`.
  [`SHA256SUMS`](SHA256SUMS) binds the compact Git package.

Boundary anchors: implementation commit
`883fc3f59691df98633711e77a5d05a451e64f23`; protocol commit
`7d39a0fb8df98535bada54aec16c514f894a8f9b`; protocol SHA-256
`745e888e47d9d33fe87fffa4bbaba618a7e91f37b55ebbf8cb08f578fc1d8f38`;
context SHA-256
`650a97519168382bf4f7fc45580fa179cb3c51a1f18195f4850c5667d6f0d6a7`;
base checkpoint SHA-256
`cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8`.

## 中文

### 最终结论

预先冻结的单次 lateral CBF-proximal adaptation 已完成固定 8 轮训练，并用
固定的 round-8 actor 完成 fresh paired base-vs-final 测试。结果**未通过开发
门槛**：target success 从 `69.336% (355/512)` 降至
`68.945% (353/512)`，变化 `-0.391 pp`，没有达到 `+3 pp` 要求；其
report-only paired 95% CI 为 `[-5.864, +5.078] pp`。因此按预注册规则不运行
contact context。

其余结果为：target fall 从 `30.664%` 升至 `31.055%`（`+0.391 pp`）；
D0 success 从 `92.578% (237/256)` 升至 `94.531% (242/256)`
（`+1.953 pp`）；D0 fall 从 `0.781%` 升至 `2.734%`。target 中有 93 次
failure 修复和 95 次 success 回归，净值 `-2/512`；D0 有 16 次修复、11 次
回归，净值 `+5/256`。置信区间只作报告，不参与门控。

训练本身按冻结协议有效：8/8 轮均正常更新，最终策略无条件取 round 8；没有
hard rollback、performance rollback 或 KL early stop。最大 moving forward KL
为 `0.001689753`，低于 `0.01` 硬上限；动作路由误差和策略存储误差均严格为
0，round-start/round-end actor 哈希链连续。405 维 actor、838 维 privileged
critic 和 runtime CBF 均保持原始部署接口。

正式评估共包含 512 个 target pair 和 256 个 D0 pair。每批 base/final 的 seed
与初始状态签名完全相同，全部使用 deterministic policy mean、runtime CBF 和
原始 actor observation interface。协议禁止因结果不佳而重跑、挑 checkpoint
或做 performance rollback，所以这次负结果是该冻结执行的最终结论，v17--v22
历史证据没有被修改。

Git 中已包含冻结协议、逐轮 JSON/CSV、训练摘要、权威 final JSON、768 行 paired
CSV、独立重建验证和两类 PNG/PDF 图。约 255 MB 的 checkpoint、raw evaluation
和日志不直接塞入仓库，而由 61 条 SHA-256 external inventory 完整绑定；紧凑
证据包再由 `SHA256SUMS` 绑定。

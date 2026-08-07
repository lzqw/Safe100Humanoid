# Fixed-Budget Observable Failure-Conditioned Brief PPO v20

## English

v20 preserves the v19 Revision-4 learning core and changes only experimental
control, evidence isolation, telemetry, and reporting. Every adaptation seed
runs exactly eight rounds. A run with zero, one, or two retained updates is
valid and proceeds to its specialist's audit; retained-update count is a
diagnostic, never a performance surrogate or validity gate.

Lateral and contact-stability have separate five-seed queues, audit
directories, verifiers, and conclusions. Completion or failure of one queue
cannot suppress the other's formal audit. No joint specialist claim, macro
gate, off-diagonal evaluation, filter-free evaluation, or CBF-independence gate
is defined.

The fresh randomness preflight passed for adaptation seeds
`73/173/273/373/473`, audit seed `5,500,000`, bootstrap seed `6,500,000`, and
the declared calibration randomness. The candidate context lists and
first-qualifying base-policy-only rule are frozen in
[`protocol_precalibration.json`](protocol_precalibration.json). Formal
adaptation must not begin until both selected contexts and their hashes are
sealed in `protocol.json` by a later commit.

Before any fresh calibration was run, the initial contact candidate list was
amended from `8217…8224` to valid generator IDs `8212…8219`; v19 context IDs
must end in `00…19`. The protocol records the prior commit and explicitly
states that no calibration or adaptation outcome had been observed.

Current evidence status: **implementation/pre-calibration only**. No v20
adaptation, formal audit, or performance result exists yet.

## 中文

v20 完整保留 v19 Revision 4 的学习核心，只修改实验控制、独立审计、遥测和
论文级报告。每个 adaptation seed 固定运行八轮；即使最终只有 0、1 或 2 次
retained update，该 run 仍然有效并进入本 specialist 的正式审计。accepted
update 数量只作诊断，不再作为有效性或性能门槛。

Lateral 与 contact-stability 使用完全独立的五 seed 队列、审计目录、验证器和
科学结论；其中一条队列的失败不能阻止另一条审计。v20 不定义 joint claim、
macro gate、off-diagonal、filter-free 或 CBF-independence gate。

新的 adaptation/audit/bootstrap/calibration 随机性碰撞检查已经通过。当前只到
实现与 calibration 前冻结阶段，尚无任何 v20 adaptation、formal audit 或性能
结果。在两个 fresh base-policy-only context 及其哈希写入后续 `protocol.json`
并提交以前，不得启动正式训练。

## Planned evidence package

```text
results/online/specialist_v20/
├── protocol_precalibration.json
├── fresh_randomness_preflight.json
├── protocol.json
├── calibration/
├── training/
├── audit/
│   ├── lateral/
│   └── contact_stability/
├── curves/
├── figures/
│   ├── main/
│   └── appendix/
├── historical_reference.csv
├── training_manifest.json
├── REQUIREMENTS_AUDIT.md
├── RUN_PROVENANCE.md
└── SHA256SUMS
```

All plots are generated deterministically as PNG and PDF from published
CSV/JSON inputs. Representative repaired trajectories use the lowest formal
paired episode ID per seed; they are never chosen visually.

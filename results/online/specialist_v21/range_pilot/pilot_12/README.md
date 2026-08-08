# v21 base-only C4 fine-range pilot / v21 C4 细网格基础策略试验

Pilot 12 completed normally at 2026-08-08 20:51:32 UTC. It evaluated all 12
prospectively frozen `C4` candidates with 256 base-policy episodes each (3,072
total). No adaptation, development selection, monitor, or formal audit was
started.

Pilot 12 于 2026-08-08 20:51:32 UTC 正常完成。它对 12 个前瞻冻结的 `C4` 候选
各运行 256 个基础策略 episode，共 3,072 个。没有启动 adaptation、development
选择、monitor 或正式审计。

Prospective protocol commit: `59fe0d4555c01be6bb44b5663b5d228c463ed2be`

Prospective protocol SHA-256: `e4140c590f47b5a638e87659835e6f458c0c333265eaa84c947ac6b19cb7c4ce`

## Result / 结果

Ten of twelve candidates passed every scaled pilot gate. Nine qualifiers were
consecutive (`38110` through `38118`), and all 12 candidates passed the
contact-stability purity gates. Across the full sweep, 721 of 735 falls were
target contact-stability failures, for pooled purity of 98.10%.

12 个候选中有 10 个通过全部 scaled pilot 门槛，其中 9 个连续通过（`38110` 至
`38118`）；全部 12 个候选都通过 contact-stability 纯度门槛。整条 sweep 共发生
735 次跌倒，其中 721 次为目标 contact-stability 失效，池化纯度为 98.10%。

The two failures identify useful boundaries. `38109` achieved 80.86% success
but recorded 49 falls, only one below the 50-fall gate. At the hard endpoint,
`38119` retained 98.84% target purity but dropped to 66.41% success, below the
70% lower bound. This rules out both an under-excited light edge and an
over-difficult heavy edge.

两个未通过点给出了有效边界：`38109` 成功率为 80.86%，但仅发生 49 次跌倒，距离
50 次门槛差 1 次；重端 `38119` 仍保持 98.84% 目标纯度，但成功率降至 66.41%，
低于 70% 下限。因此轻端激励不足和重端过难都被明确排除。

For the fresh 512-episode replacement calibration, the recommended bracket is
the stricter contiguous interior represented by `38112` through `38118`:
friction 0.456745→0.447964, forward scale 1.092982→1.099855, low-pass
0.076764→0.082109 s, and action gain 0.915127→0.909018. New candidate and
evaluation randomness remains mandatory. This pilot is range evidence only;
it is not formal context selection or an adaptation result.

下一次全新 512-episode replacement calibration 推荐使用 `38112` 至 `38118` 所代表
的连续严格内区间：friction 0.456745→0.447964、forward scale
1.092982→1.099855、low-pass 0.076764→0.082109 秒、action gain
0.915127→0.909018。仍必须使用全新的候选与评估随机性。本试验仅提供范围证据，
不属于正式 context 选择或 adaptation 结果。

The exact 12 rows are in
[`C4/calibration_progress.json`](C4/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 73 external raw files
(50,042,599 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256
manifest 绑定该文件与算力机上的 73 个原始文件（50,042,599 字节）。

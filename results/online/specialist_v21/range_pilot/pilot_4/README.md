# v21 base-only L2 actuator-offset pilot / v21 L2 执行器偏置基础策略试验

Pilot 4 completed normally at 2026-08-08 17:16:19 UTC. It evaluated all 12
frozen `L2` candidates with 128 base-policy episodes each (1,536 total). No
adaptation, development selection, monitor, or formal audit was started.

pilot 4 于 2026-08-08 17:16:19 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
128 个基础策略 episode，共 1,536 个。没有启动 adaptation、development 选择、
monitor 或正式审计。

Prospective protocol commit: `d7c4dabdeaea7cbd196935c4a654478c7ebfd900`

Prospective protocol SHA-256: `ba7b9f584e0595084c3371991fb0d51c8d26b005a84aeb1faca3d79899ce7af3`

## Result / 结果

No candidate passed every scaled pilot gate. Candidate `19412` was matched in
success and fall count (78.13% success, 28 falls), but only 53.57% of failures
were lateral-heading drift. Candidate `19413` moved toward the intended
mechanism (76.19% lateral purity, 23.81% second mechanism), but was already too
hard at 67.19% success and still missed the 80% purity gate.

没有候选通过全部 scaled pilot 门槛。候选 `19412` 的成功率和跌倒数匹配目标
（成功率 78.13%、28 次跌倒），但 lateral-heading drift 仅占 53.57%。候选
`19413` 更接近预期机制（lateral purity 76.19%、第二机制 23.81%），但成功率已降至
67.19%，过难，并且仍未达到 80% purity 门槛。

The sweep exposes a sharp difficulty cliff between these two candidates.
Higher offsets drove success as low as 7.03% without improving failure purity.
The deterministic same-sign bilateral hip-yaw zero offset is therefore
rejected as the formal `L2` family. This is a negative range result, not an
algorithm result.

该 sweep 在这两个候选之间暴露出明显的难度 cliff。继续增大偏置会把成功率降至最低
7.03%，却没有提高失败 purity。因此，确定性的同号双侧 hip-yaw 零偏置被否决，不能
作为正式 `L2` family。这个结论只是负面的范围证据，不是算法结果。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 49 external raw files
(48,274,665 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 49 个原始文件（48,274,665 字节）。

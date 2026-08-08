# v21 base-only L2 extended-yaw pilot / v21 L2 扩展 yaw 基础策略试验

Pilot 5 completed normally at 2026-08-08 17:30:10 UTC. It evaluated all 12
frozen `L2` candidates with 128 base-policy episodes each (1,536 total). No
adaptation, development selection, monitor, or formal audit was started.

pilot 5 于 2026-08-08 17:30:10 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
128 个基础策略 episode，共 1,536 个。没有启动 adaptation、development 选择、
monitor 或正式审计。

Prospective protocol commit: `afdd519c246c4933bffeba83d7977d521a6e95ae`

Prospective protocol SHA-256: `3d7169f273e71c47e5a69f752d2fbcb1607b048341829674908607993dced23b`

## Result / 结果

No candidate passed every scaled pilot gate, but the high end formed a narrow
near-qualifying cluster. Candidate `21417` had 81.25% success and 83.33%
lateral purity but 24 falls, one short of the gate. Candidate `21418` had
77.34% success and 29 falls but 79.31% purity, 0.69 percentage points short.
Candidate `21419` had 82.81% success and 81.82% purity but 22 falls.

没有候选通过全部 scaled pilot 门槛，但高端形成了一个狭窄的近合格区域。候选
`21417` 的成功率为 81.25%、lateral purity 为 83.33%，但只有 24 次跌倒，差 1 次；
`21418` 的成功率为 77.34%、29 次跌倒，但 purity 为 79.31%，差 0.69 个百分点；
`21419` 的成功率为 82.81%、purity 为 81.82%，但只有 22 次跌倒。

This is not sufficient to declare range feasibility. It does show that the
command-side yaw mechanism is substantially better targeted than the rejected
actuator-offset family. The next non-formal test increases the sample size to
256 episodes per candidate and varies only persistent yaw bias above the
centerline controller's approximately 0.45 yaw-correction authority while
holding pulse magnitude fixed.

这些结果不足以宣布范围可行，但说明 command-side yaw 机制比已否决的 actuator-offset
family 更接近目标。下一轮非正式试验将每候选样本数提高到 256，并只改变超过 centerline
控制器约 0.45 yaw 修正能力的持续 yaw bias，同时固定 pulse 幅度。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 49 external raw files
(48,260,574 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 49 个原始文件（48,260,574 字节）。

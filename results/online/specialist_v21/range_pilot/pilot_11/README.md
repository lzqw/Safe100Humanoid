# v21 base-only L2 narrow-stair pilot / v21 L2 窄阶梯基础策略试验

Pilot 11 completed normally at 2026-08-08 19:19:05 UTC. It evaluated all 12
frozen `L2` candidates with 256 base-policy episodes each (3,072 total). No
adaptation, development selection, monitor, or formal audit was started.

Pilot 11 于 2026-08-08 19:19:05 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
256 个基础策略 episode，共 3,072 个。没有启动 adaptation、development 选择、
monitor 或正式审计。

Prospective protocol commit: `6342888ee07f2d78c861ab6e66ebb3432546d4d4`

Prospective protocol SHA-256: `c64d8ee43fa5929e834adc9da15a8c326f3b8bc09381d1a3db1fa7c95df1b406`

## Result / 结果

Three contiguous candidates passed every scaled pilot gate. The representative
candidate `33408` used a 1.00 m stair half-width (2.00 m physical width), reached
76.17% success, recorded 61 falls, and classified 86.89% of those falls as the
target lateral/heading mechanism. Candidates `33409` and `33410`, at 0.95 m and
0.90 m half-width, also qualified with target purity of 93.75% and 92.42%.

三个连续候选通过全部 scaled pilot 门槛。代表候选 `33408` 使用 1.00 m 阶梯半宽
（2.00 m 物理全宽），成功率为 76.17%，发生 61 次跌倒，其中 86.89% 被分类为目标
横向/航向机制。半宽 0.95 m 与 0.90 m 的 `33409`、`33410` 也通过，目标纯度分别为
93.75% 与 92.42%。

The next point, `33411` at 0.85 m, retained 93.59% target purity but fell just
below the success gate at 69.53% (178 successes, two short of the 70% lower
bound). Across the full sweep, 1,280 of 1,345 falls were target failures, for a
pooled purity of 95.17%. The physical-clearance axis therefore gives a clear,
mechanism-pure difficulty transition.

下一个 0.85 m 候选 `33411` 仍保持 93.59% 的目标纯度，但成功率为 69.53%
（178 次成功，距离 70% 下限少 2 次），刚好进入过难区。整条 sweep 共 1,345 次
跌倒，其中 1,280 次属于目标失效，池化纯度为 95.17%。因此，物理净空轴形成了清晰、
机制纯净的难度过渡。

This result establishes exploratory base-policy range feasibility; it is not a
formal context selection or an adaptation result. The next boundary is a newly
frozen 512-episode replacement calibration with fresh candidate and evaluation
randomness and a fine `L2` half-width sweep in the observed feasible neighborhood.

该结果只建立探索性的基础策略范围可行性，并不等于正式 context 选择或 adaptation
结果。下一边界是重新前瞻冻结的 512-episode replacement calibration：使用全新的候选
与评估随机性，并在已观察到的可行邻域内细扫 `L2` 半宽。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 73 external raw files
(49,549,262 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 73 个原始文件（49,549,262 字节）。

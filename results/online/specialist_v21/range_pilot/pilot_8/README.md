# v21 base-only L2 strong persistent-yaw pilot / v21 L2 强持续 yaw 基础策略试验

Pilot 8 completed normally at 2026-08-08 18:19:24 UTC. It evaluated all 12
frozen `L2` candidates with 256 base-policy episodes each (3,072 total). No
adaptation, development selection, monitor, or formal audit was started.

pilot 8 于 2026-08-08 18:19:24 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
256 个基础策略 episode，共 3,072 个。没有启动 adaptation、development 选择、
monitor 或正式审计。

Prospective protocol commit: `3526effef2aa4595a9ea93cc567736409491359d`

Prospective protocol SHA-256: `df5e8ef38353912257727908175cb29c71da7a3ddecb9142834c6a9b248825a2`

## Result / 结果

No candidate passed every scaled pilot gate. Candidate `27417` was the closest:
79.30% success, 53 falls, 79.25% lateral/heading failure purity, and 20.75% for
the second failure type. It passed the difficulty, fall-count, and secondary
failure gates, but missed the 80% target-purity gate by 0.75 percentage points.

没有候选通过全部 scaled pilot 门槛。最接近的是 `27417`：成功率 79.30%，53 次
跌倒，lateral/heading failure 纯度 79.25%，第二失败类型占 20.75%。它通过了难度、
跌倒数量和第二失败类型门槛，但距离 80% 目标纯度门槛仍差 0.75 个百分点。

Across all candidates, 425 of 588 falls were lateral/heading failures, for a
pooled purity of 72.28%. Increasing pulse-free persistent yaw bias from 0.65 to
0.95 therefore raised failure volume without producing a robust purity
intersection. Together with pilots 5--7, this closes the command-magnitude axis
within the tested implementation; another larger-yaw extension is not justified.

全部候选合计 588 次跌倒，其中 425 次属于 lateral/heading failure，池化纯度为
72.28%。因此，把无 pulse 的持续 yaw bias 增大到 0.65--0.95 虽增加了失败量，却
没有形成稳健的纯度交集。结合 pilots 5--7，本实现下的 command-magnitude 轴到此
停止，不再继续加大 yaw 幅度。

The next base-only design should change the mechanism: reduce heading-feedback
authority under a fixed, modest yaw excitation while preserving lateral
centering. Its protocol and randomness must be frozen prospectively before any
execution.

下一轮 base-only 设计应改变机制：在固定、适中的 yaw 激励下削弱 heading feedback
authority，同时保持 lateral centering。执行前仍须前瞻冻结协议和全新随机性。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 73 external raw files
(49,506,219 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 73 个原始文件（49,506,219 字节）。

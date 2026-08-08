# v21 base-only L2 heading-authority pilot / v21 L2 heading authority 基础策略试验

Pilot 9 completed normally at 2026-08-08 18:41:09 UTC. It evaluated all 12
frozen `L2` candidates with 256 base-policy episodes each (3,072 total). No
adaptation, development selection, monitor, or formal audit was started.

pilot 9 于 2026-08-08 18:41:09 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
256 个基础策略 episode，共 3,072 个。没有启动 adaptation、development 选择、
monitor 或正式审计。

Prospective protocol commit: `e1f9c2769ad3b09bcef42cfe8ff9a0a6640226a7`

Prospective protocol SHA-256: `60f38baec35ec824501052a1e1ebf0d08dd3560bbe7aa49d1a087a475d4f406a`

## Result / 结果

No candidate passed every scaled pilot gate. Three candidates demonstrated a
clean heading-drift mechanism but remained too easy: `29408`, `29409`, and
`29414` had 80.49%, 80.56%, and 84.21% target purity but only 41, 36, and 38
falls. Conversely, candidates reaching at least 50 falls had only 63.46% to
75.47% target purity.

没有候选通过全部 scaled pilot 门槛。`29408`、`29409`、`29414` 显示出干净的
heading-drift 机制，目标纯度分别为 80.49%、80.56%、84.21%，但只有 41、36、38 次
跌倒，仍然过于容易。反之，达到至少 50 次跌倒的候选，其目标纯度只有 63.46%--75.47%。

Across all candidates, 401 of 550 falls were lateral/heading failures, for a
pooled purity of 72.91%. Lowering heading correction authority under a fixed
0.34 yaw bias therefore did not close the difficulty/purity intersection. Very
low authority is rejected rather than extended further.

全部候选合计 550 次跌倒，其中 401 次属于 lateral/heading failure，池化纯度为
72.91%。因此，在固定 0.34 yaw bias 下削弱 heading correction authority 仍未形成
难度与纯度的交集，不再继续降低 authority。

The next base-only mechanism should use a bounded biased visual heading
reference with nominal feedback authority. That can create gradual geometric
drift without a large yaw-rate transient or a weak controller. Its protocol,
implementation, and randomness must be frozen prospectively before execution.

下一轮 base-only 机制应在正常反馈 authority 下使用有界的视觉 heading reference
偏置，使几何漂移逐步积累，同时避免强 yaw-rate 瞬态或弱控制器。实现、协议和随机性
仍须在执行前前瞻冻结。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 73 external raw files
(49,509,424 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 73 个原始文件（49,509,424 字节）。

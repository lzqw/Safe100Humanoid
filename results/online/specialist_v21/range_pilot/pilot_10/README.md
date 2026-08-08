# v21 base-only L2 visual-heading pilot / v21 L2 视觉 heading 基础策略试验

Pilot 10 completed normally at 2026-08-08 19:00:01 UTC. It evaluated all 12
frozen `L2` candidates with 256 base-policy episodes each (3,072 total). No
adaptation, development selection, monitor, or formal audit was started.

pilot 10 于 2026-08-08 19:00:01 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
256 个基础策略 episode，共 3,072 个。没有启动 adaptation、development 选择、
monitor 或正式审计。

Prospective protocol commit: `750ae8866d967d0895b68cfafa20b28be0e6e878`

Prospective protocol SHA-256: `6335208d0cc85cf32381d8e671df71dd6246046436758da47a5ca4cb1eaf33e3`

## Result / 结果

No candidate passed every scaled pilot gate. The sweep never reached either
the 50-fall gate or the 80% target-purity gate. Candidate `31412` had the most
falls (45) but only 66.67% target purity; candidate `31410` had the highest
purity (76.92%) but only 39 falls.

没有候选通过全部 scaled pilot 门槛。整条 sweep 既未达到 50 次跌倒门槛，也未达到
80% 目标纯度门槛。`31412` 的跌倒数最多（45 次），但目标纯度只有 66.67%；
`31410` 的纯度最高（76.92%），但只有 39 次跌倒。

Across all candidates, 306 of 429 falls were lateral/heading failures, for a
pooled purity of 71.33%. A 0.35--0.75 rad biased visual heading reference with
nominal feedback did not form a useful difficulty axis and is rejected.

全部候选合计 429 次跌倒，其中 306 次属于 lateral/heading failure，池化纯度为
71.33%。在正常反馈下使用 0.35--0.75 rad 的视觉 heading reference 偏置没有形成
有效难度轴，因此否决该机制。

The next base-only mechanism should directly vary physical lateral clearance
while leaving commands, feedback, actions, encoders, and rise/tread profiles
nominal. Actual root/foot edge clearance and the matching classifier threshold
must use the same frozen stair half-width.

下一轮 base-only 机制应直接改变物理侧向净空，同时保持 command、feedback、action、
encoder 与 rise/tread profile 为 nominal。真实 root/foot edge clearance 与分类阈值
必须共同使用同一个冻结 stair half-width。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 73 external raw files
(49,512,807 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 73 个原始文件（49,512,807 字节）。

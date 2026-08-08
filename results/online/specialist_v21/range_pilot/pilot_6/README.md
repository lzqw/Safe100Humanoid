# v21 base-only L2 persistent-yaw pilot / v21 L2 持续 yaw 基础策略试验

Pilot 6 completed normally at 2026-08-08 17:45:56 UTC. It evaluated all 12
frozen `L2` candidates with 256 base-policy episodes each (3,072 total). No
adaptation, development selection, monitor, or formal audit was started.

pilot 6 于 2026-08-08 17:45:56 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
256 个基础策略 episode，共 3,072 个。没有启动 adaptation、development 选择、
monitor 或正式审计。

Prospective protocol commit: `e9f7fb5fbb9fc7e47435b6a6075fb57f473e6c5b`

Prospective protocol SHA-256: `f57f8db0143d9ff3a2dfb208e1132c68c606b5c8576f71311cd2c9ab8d88f823`

## Result / 结果

No candidate passed every scaled pilot gate. Two candidates were close while
meeting both success and purity gates: `23410` had 81.64% success, 47 falls,
and 82.98% lateral purity; `23419` had 81.25% success, 48 falls, and 81.25%
purity. They missed the 50-fall gate by three and two falls respectively.

没有候选通过全部 scaled pilot 门槛。两个候选在满足成功率与 purity 门槛的同时非常
接近：`23410` 的成功率为 81.64%、47 次跌倒、lateral purity 为 82.98%；`23419`
的成功率为 81.25%、48 次跌倒、purity 为 81.25%。它们分别比 50 次跌倒门槛少 3
次和 2 次。

Persistent yaw bias above the controller's frozen correction authority is a
clean and mechanism-pure axis, but by itself it remains slightly too easy. The
next non-formal confirmation keeps a narrow persistent-bias sweep and adds one
fixed low-amplitude yaw-pulse carrier. Pulse magnitude and timing do not vary
with severity, so persistent bias remains the only swept axis.

超过控制器冻结修正能力的持续 yaw bias 是干净且机制纯的轴，但单独使用仍略微偏易。
下一轮非正式 confirmation 保留狭窄的持续 bias sweep，并加入一个固定的低幅 yaw
pulse carrier。pulse 幅度和时序不随 severity 改变，因此持续 bias 仍是唯一 sweep 轴。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 73 external raw files
(49,507,988 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 73 个原始文件（49,507,988 字节）。

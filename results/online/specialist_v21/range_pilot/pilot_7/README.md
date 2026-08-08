# v21 base-only L2 fixed-carrier pilot / v21 L2 固定 carrier 基础策略试验

Pilot 7 completed normally at 2026-08-08 18:02:21 UTC. It evaluated all 12
frozen `L2` candidates with 256 base-policy episodes each (3,072 total). No
adaptation, development selection, monitor, or formal audit was started.

pilot 7 于 2026-08-08 18:02:21 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
256 个基础策略 episode，共 3,072 个。没有启动 adaptation、development 选择、
monitor 或正式审计。

Prospective protocol commit: `c32ef0b3d000afbca195e899b9c3dd1f7dbd5afb`

Prospective protocol SHA-256: `324e40fbb5918104593beda62fd795a2eceb5c773a1ba18e426066461fb05e3d`

## Result / 结果

No candidate passed every scaled pilot gate. The fixed pulse carrier supplied
enough failures for `25410`, `25411`, and `25419` (51, 53, and 54 falls), but
their lateral purity was only 68.63%, 77.36%, and 72.22%. The two
mechanism-pure near misses, `25416` and `25418`, still had only 47 and 46 falls.

没有候选通过全部 scaled pilot 门槛。固定 pulse carrier 使 `25410`、`25411` 和
`25419` 达到足够失败量（51、53、54 次跌倒），但 lateral purity 只有 68.63%、
77.36% 和 72.22%。两个机制纯的近失配点 `25416`、`25418` 仍只有 47 和 46 次跌倒。

Across all candidates, pooled lateral purity was 74.05%, below the 75.56% from
the pulse-free pilot 6. The carrier is rejected. The next base-only pilot
returns to pulse-free persistent yaw and extends the bias range so heading
drift can dominate before ordinary contact failures.

全部候选合并后的 lateral purity 为 74.05%，低于无 pulse 的 pilot 6 的 75.56%。
因此该 carrier 被否决。下一轮 base-only pilot 回到无 pulse 的持续 yaw，并扩展 bias
范围，使 heading drift 能在普通 contact failure 之前占主导。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 73 external raw files
(49,505,745 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 73 个原始文件（49,505,745 字节）。

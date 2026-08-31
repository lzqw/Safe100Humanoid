# v21 base-only L2 robustness pilot / v21 L2 基础策略稳健性试验

Pilot 3 completed normally at 2026-08-08 16:21:24 UTC. It evaluated all 12
frozen `L2` candidates with 128 base-policy episodes each (1,536 total). No
adaptation, development selection, or formal audit was started.

pilot 3 于 2026-08-08 16:21:24 UTC 正常完成。它对 12 个冻结的 `L2` 候选各运行
128 个基础策略 episode，共 1,536 个。没有启动适配、development 选择或正式审计。

Prospective protocol commit: `1c406ae3a82e1d11ad613fdbc1c28a2092671b63`

Prospective protocol SHA-256: `46e1f6429cc49f8d4c3dd40a2a3b9addd8172492bda15a4d41315c522151c00b`

## Result / 结果

Candidate `15417` qualified with strict margin on every scaled pilot gate:
78.13% success, 28 falls, 82.14% lateral purity, and 17.86% for the second
failure mechanism. Candidate `15419` remained nearby at 82.03% success, 23
falls, and 86.96% lateral purity, but is correctly not counted as a qualifier.

候选 `15417` 在每条 scaled pilot 门槛上都有严格余量：成功率 78.13%、28 次跌倒、
lateral purity 82.14%、第二失败机制 17.86%。候选 `15419` 也处于邻近区域：成功率
82.03%、23 次跌倒、lateral purity 86.96%，但它没有达到跌倒数门槛，因此不计为
合格点。

The pilot also falsified the assumption that removing all non-yaw carriers
would monotonically increase useful difficulty: early candidates became too
easy. The stronger yaw-dominant end nevertheless establishes range feasibility.
The next appropriate test is the precommitted 512-episode base-only formal
calibration with entirely fresh randomness, not another small-sample pilot.

本轮也否定了“移除所有非 yaw carrier 就会单调增加有效难度”的假设：前段候选反而
过于容易。不过较强的 yaw-dominant 后段已经证明该范围可行。下一步应使用全新随机数，
进行预先提交、每候选 512 episode 的正式 base-only 校准，而不是继续小样本调参。

The exact 12 rows are in [`L2/calibration_progress.json`](L2/calibration_progress.json).
[`summary.json`](summary.json) binds that file and 49 external raw files
(48,260,384 bytes) with deterministic SHA-256 manifests.

精确的 12 行结果见上述 progress 文件；`summary.json` 使用确定性的 SHA-256 manifest
绑定该文件与算力机上的 49 个原始文件（48,260,384 字节）。

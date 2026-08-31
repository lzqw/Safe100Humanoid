# v21 base-only range pilot 2 / v21 基础策略范围试验 2

This non-formal pilot completed normally at 2026-08-08 16:09:30 UTC. It
evaluated all 12 frozen candidates for the five families that failed pilot 1,
using 128 base-policy episodes per candidate (7,680 episodes total). No
adaptation, development selection, or formal audit was started.

本轮非正式 pilot 于 2026-08-08 16:09:30 UTC 正常完成。它只评估 pilot 1 中未通过
的 5 个 family，每个 family 跑完 12 个冻结候选，每个候选 128 个基础策略 episode，
共 7,680 个 episode。没有启动适配、development 选择或正式审计。

Prospective protocol commit: `170a7940c1fbebfebceaa428f939e88e38d6c575`

Prospective protocol SHA-256: `92e6003165d61c2128fb0267160f3645baecb10eb43ca673966c7cb01b5167dc`

## Result / 结果

All five families produced at least one qualifying candidate. Four ranges are
robust enough to retain. `L2` produced only one qualifier, and that candidate
sat exactly on both its 25-fall and 80% target-purity thresholds; it therefore
needs one fresh base-only robustness pilot before formal calibration.

5 个 family 都至少产生了一个合格候选。其中 4 个范围可以保留。`L2` 只有一个
合格点，而且它恰好落在 25 次跌倒和 80% 目标 purity 两条边界上；因此在正式校准前，
还需要一轮使用全新随机数的 base-only 稳健性确认。

| Context | Qualifying seeds | Representative result | Disposition |
| --- | --- | --- | --- |
| L_dev | 13108, 13110 | 13110: 79.69% SR, 26 falls, 92.31% purity | Keep |
| L2 | 13408 | 80.47% SR, 25 falls, 80.00% purity | Confirm again |
| L4 | 13610–13611, 13613–13619 | 13610: 79.69% SR, 26 falls, 88.46% purity | Keep |
| L5 | 13712–13715 | 13712: 78.13% SR, 28 falls, 89.29% purity | Keep |
| C3 | 14008, 14010–14013 | 14010: 77.34% SR, 29 falls, 93.10% purity | Keep |

The exact rows are the five `calibration_progress.json` files beside this
summary. [`summary.json`](summary.json) binds them and the external raw evidence
with deterministic SHA-256 manifests. The raw store contains 245 files
(241,580,609 bytes) and remains on the compute host.

精确结果保存在本目录的 5 份 `calibration_progress.json`；
[`summary.json`](summary.json) 以确定性的 SHA-256 manifest 绑定紧凑证据和外部原始
证据。原始产物共 245 个文件、241,580,609 字节，保留在算力机上。

## Scientific boundary / 科学边界

Pilot 3 is restricted to `L2` and may use only prior base-policy evidence. It
must use fresh candidate/evaluation randomness and a protocol committed before
execution. No adaptation outcome may influence the range.

pilot 3 仅允许处理 `L2`，并且只能使用此前的基础策略证据。它必须使用全新的候选与
评估随机数，并在执行前提交冻结协议；任何适配结果都不得影响范围设计。

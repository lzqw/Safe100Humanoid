# v21 replacement-v2 calibration results / replacement-v2 校准结果

Replacement-v2 completed normally at 2026-08-08 22:01:48 UTC and froze all
12 deployment contexts. It evaluated 30 prospectively declared candidates with
512 base-policy episodes each (15,360 episodes total). No adaptation,
development selection, monitor evaluation, or formal audit was started.

replacement-v2 于 2026-08-08 22:01:48 UTC 正常完成并冻结全部 12 个 deployment
context。它评估了 30 个前瞻声明的候选，每个候选运行 512 个基础策略 episode，共
15,360 个。没有启动 adaptation、development selection、monitor 或正式 audit。

Prospective protocol commit: `5dac6962a2a9b13053032477008c81cc9a82f269`

Prospective protocol SHA-256: `af073b100c7446e2ffe34033ef24bff3905e770dd2cc77b3c89df9d3a83df23d`

## Selected contexts / 冻结结果

| Context | Attempts | Seed | Success | Falls | Target | Second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L_dev | 2 | 39109 | 79.10% | 107 | 84.11% | 15.89% |
| C_dev | 1 | 39208 | 80.47% | 100 | 98.00% | 2.00% |
| L1 | 1 | 39308 | 72.66% | 139 | 91.37% | 8.63% |
| L2 | 1 | 39408 | 78.52% | 110 | 82.73% | 17.27% |
| L3 | 1 | 39508 | 77.93% | 113 | 86.73% | 13.27% |
| L4 | 1 | 39608 | 76.56% | 120 | 85.83% | 14.17% |
| L5 | 5 | 39712 | 75.59% | 125 | 88.00% | 12.00% |
| C1 | 5 | 39812 | 79.88% | 103 | 99.03% | 0.97% |
| C2 | 1 | 39908 | 77.15% | 117 | 96.58% | 3.42% |
| C3 | 1 | 40008 | 78.52% | 110 | 99.09% | 0.91% |
| C4 | 2 | 40109 | 80.08% | 102 | 98.04% | 1.96% |
| C5 | 9 | 40216 | 76.95% | 118 | 96.61% | 3.39% |

`L_dev` and `C_dev` remain development-only and are excluded from every formal
claim. The other ten contexts are now fixed inputs for later formal runs.

`L_dev` 与 `C_dev` 仍只用于 development，并从所有正式结论中排除；其余 10 个
context 已成为后续正式运行的固定输入。

## C4 repair / C4 修复

The prior coarse C4 grid skipped the formal window: one point had only 91
falls, while the next had 358/512 successes (69.92%), exactly one success below
the lower gate. In replacement-v2, `40108` remained slightly easy at 92 falls,
but the next fine-grid point `40109` reached 102 falls at 80.08% success with
98.04% target contact-stability purity. The former blocking context is therefore
formally frozen under fresh randomness.

上一轮粗 C4 网格跳过了正式窗口：一个点只有 91 次跌倒，下一个点为 358/512 次
成功（69.92%），距离成功率下限恰好差 1 次成功。replacement-v2 中，`40108` 仍
略易（92 次跌倒），但下一个细网格点 `40109` 在 80.08% 成功率下达到 102 次跌倒，
目标 contact-stability 纯度为 98.04%。因此，原阻塞 context 已在全新随机性下正式
冻结。

## Evidence boundary / 证据边界

The committed evidence contains 24 per-context calibration files and 12 frozen
context files. The external raw root contains 324 files (131,758,452 bytes),
including 120 JSON and 120 CSV evaluation files. All 30 actor copies have the
same frozen base-policy SHA-256. Thirteen logs were scanned with no error
signature, the queue wrote `queue_completed`, and no v21 process remained after
completion.

已提交证据包含 24 份逐 context 校准文件与 12 份冻结 context 文件。外部原始目录
包含 324 个文件（131,758,452 字节），其中有 120 份 JSON 与 120 份 CSV 评估文件。
30 份 actor 副本的冻结基础策略 SHA-256 完全一致。13 份日志未发现错误特征，队列已
写入 `queue_completed`，完成后没有残留 v21 进程。

Machine-readable totals, selected rows, and deterministic manifests are in
[`calibration/replacement_v2/summary.json`](calibration/replacement_v2/summary.json).
The next evidence boundary is a committed development protocol followed by the
predeclared beta grid on `L_dev` and `C_dev` only.

机器可读的总数、选择行与确定性 manifest 见上述 summary。下一证据边界是先提交
development protocol，再仅在 `L_dev` 与 `C_dev` 上运行预声明 beta 网格。

# v21 replacement calibration status / v21 replacement 校准状态

The prospectively frozen replacement calibration stopped normally at `C4` on
2026-08-08 20:32:28 UTC. It evaluated 33 candidates and 16,896 base-policy
episodes. No adaptation, development beta selection, monitor run, or formal
audit was started, and no calibration process remains running.

前瞻冻结的 replacement calibration 于 2026-08-08 20:32:28 UTC 在 `C4` 正常
fail-fast。它共评估 33 个候选、16,896 个基础策略 episode。没有启动 adaptation、
development beta 选择、monitor 或正式 audit，当前也没有校准进程运行。

## Completed contexts / 已完成 context

| Context | Selected seed | Success | Falls | Target purity | Second failure |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L_dev` | 35109 | 80.47% | 100 | 81.00% | 19.00% |
| `C_dev` | 35211 | 77.34% | 116 | 97.41% | 2.59% |
| `L1` | 35308 | 77.73% | 114 | 85.96% | 14.04% |
| `L2` | 35408 | 75.59% | 125 | 84.80% | 15.20% |
| `L3` | 35508 | 79.10% | 107 | 82.24% | 17.76% |
| `L4` | 35608 | 78.91% | 108 | 83.33% | 16.67% |
| `L5` | 35710 | 76.17% | 122 | 85.25% | 14.75% |
| `C1` | 35813 | 77.15% | 117 | 99.15% | 0.85% |
| `C2` | 35908 | 76.56% | 120 | 92.50% | 7.50% |
| `C3` | 36008 | 77.54% | 115 | 100.00% | 0.00% |

The new physical-clearance `L2` mechanism therefore replicated successfully at
512 episodes with entirely fresh randomness. This fixes the previous `L2`
calibration failure.

新的物理净空 `L2` 机制在完全不同的随机数下，以 512 episodes 成功复现，因此上一轮
`L2` 校准失败已被修复。

## C4 stop / C4 停止原因

No `C4` candidate passed every gate. Candidate `36112` was mechanism-pure and
inside the success band but had only 91 falls. The next point, `36113`, had 154
falls and 100% target purity but only 358 successes: exactly one success short
of the 70% lower gate. Every stronger candidate was over-difficult. `C5` was not
started after the fail-fast stop.

`C4` 没有候选通过全部门槛。`36112` 的机制纯度和成功率合格，但只有 91 次跌倒；
下一个 `36113` 有 154 次跌倒且目标纯度为 100%，却只有 358 次成功，距离 70% 下限
恰好少 1 次成功。更强的候选全部过难。fail-fast 后没有启动 `C5`。

The next step is a non-formal, base-policy-only Pilot 12 that finely scans the
same `C4` mechanism between original severity 0.38 and 0.45 with new candidate
and evaluation randomness. It will not reuse any adapted-policy outcome.

下一步是非正式、仅基础策略的 Pilot 12：保留同一个 `C4` 机制，在原始 severity
0.38--0.45 之间细扫，并使用全新的候选与评估随机数；不会使用任何 adapted-policy
结果。

The exact compact evidence is under [`calibration/replacement`](calibration/replacement)
and [`contexts_replacement`](contexts_replacement). The machine-readable failure
boundary is
[`revision0_C4_failure_amendment.json`](calibration/replacement/revision0_C4_failure_amendment.json).
The amendment binds 31 committed files, 351 external raw artifact files
(145,134,335 bytes), and 12 logs with deterministic SHA-256 manifests.

精确紧凑证据见上述两个目录；机器可读的失败边界见 amendment。它以确定性的
SHA-256 manifest 绑定 31 个已提交文件、351 个外部原始工件（145,134,335 bytes）
与 12 份日志。

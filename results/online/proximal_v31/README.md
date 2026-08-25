# v31 CBF-Teacher Formal Matrix

## 中文摘要

v30 因 F2 behavior-log-prob float32 reduction 容差过紧以及 F3 固定 10-slot terrain patch allocation 而未完成；其结果保持原样且未补写。v31 是独立冻结的新实验：将 log-prob 容差预先改为 `1e-3`（raw action、Gaussian 参数与 safe-action 路由检查仍保持严格），并按 `num_risers + 1` 动态分配 stair-target patches，使 F3 的 11 个 riser 正确包含 12 个 target patches。

三个方法固定为 A0（PPO + moving KL，无 teacher）、A1（完整 safe-action、50-step local-success gate、Gaussian NLL、weight 0.1）和 A2（residual `eta=0.25`、全部 intervention、weighted Smooth-L1、weight 1.0）。每个 context/method 只适配一次，固定使用 round 8；KL、训练回报和最终评价均未用于停止或选择 checkpoint。

预检透明性：F1-A0、F2-A1、F3-A2 三个 GPU case 均只执行一次且全部通过。顶层编排器随后在纯 CPU 聚合阶段误从已安装的旧 mirror 导入 `5e-4` 容差并以状态 1 退出；恢复过程只用冻结仓库的 `PYTHONPATH` 重新聚合既有 case summaries 并重复纯容差断言，没有重建环境、重跑 rollout/update、修改源码或修改协议。详见 [`preflight_failure.json`](preflight_failure.json) 与 [`preflight_aggregation_recovery.json`](preflight_aggregation_recovery.json)。

| context | A0 on / off / D0 | A1 on / off / D0 | A2 on / off / D0 |
|---|---:|---:|---:|
| F1 | 71.09% / 63.87% / 89.84% | 72.85% / 75.59% / 93.75% | 76.37% / 63.28% / 92.19% |
| F2 | 66.21% / 58.59% / 90.62% | 61.13% / 66.60% / 94.92% | 70.12% / 58.20% / 91.02% |
| F3 | 59.96% / 50.20% / 94.53% | 63.09% / 70.12% / 95.31% | 69.34% / 51.17% / 93.36% |

| method | mean CBF-on success | mean CBF-off success | mean D0 success | interventions/riser | off kick events/riser |
|---|---:|---:|---:|---:|---:|
| A0 | 65.76% | 57.55% | 91.67% | 6.315 | 0.752 |
| A1 | 65.69% | 70.77% | 94.66% | 3.510 | 0.711 |
| A2 | 71.94% | 57.55% | 92.19% | 6.173 | 0.759 |

三场景平均 shielded success 最高的方法是 **A2**（71.94%）。A1/A2 相对 A0 的平均 CBF-on success 差异分别为 -0.0007 和 +0.0618；CBF-off 差异分别为 +0.1322 和 +0.0000。相对 A0 的 interventions/riser 差异分别为 -2.8055 和 -0.1419。D0 相对 base 的变化为 A0 +0.0182、A1 +0.0482、A2 +0.0234。

实机建议：Use A2 as the adapted candidate only if its shielded gain, D0 retention, and risk profile are acceptable beside the base+CBF baseline. 真机仍应以 base+CBF 为安全基线，并同时审查 nominal would-intervene、correction norm、toe-riser risk 和 D0 retention，而不是只看单一 success 指标。

## English summary

v31 is a separately frozen rerun; v30 remains an immutable incomplete result. It prospectively raises only the float32 behavior-log-probability reduction tolerance to `1e-3`, while retaining strict raw-action, Gaussian-parameter, and safe-action routing audits. It also allocates stair patches dynamically, including all 12 target patches for F3's 11-riser profile.

All nine A0/A1/A2 × F1/F2/F3 adaptations use their predeclared seeds and unconditional round-8 policies. The highest mean CBF-on success belongs to **A2** at 71.94%. Paired 95% bootstrap intervals, repairs/regressions, nominal internalization, CBF dependence, toe-riser risk, and D0 retention are reported in each context JSON. No result was used as a gate.

Preflight transparency: the F1-A0, F2-A1, and F3-A2 GPU cases each ran exactly once and all passed. The top-level orchestrator then exited with status 1 during CPU-only aggregation because it imported the installed old mirror's `5e-4` tolerance. Recovery only re-aggregated the existing case summaries and repeated the pure tolerance assertion with `PYTHONPATH` fixed to the frozen repository; no environment, rollout/update, source, or protocol was rerun or changed. The original failure and recovery receipts are published beside this report.

For a real-stair follow-up, keep base+CBF as the safety baseline and consider the highest-success adapted method only together with its D0 retention and risk/dependence measurements.

Protocol source commit: `b1cd615c827cbb0ee47d2429dd9cf86d5f1a0a85`. Protocol SHA-256 and every published aggregate are bound by `SHA256SUMS`; external checkpoints and raw telemetry are bound by `external_artifacts_manifest.json`.

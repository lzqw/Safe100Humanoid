# CBF-Proximal Completion: v23 Lateral + v24 Contact

## English

v23 and v24 are independent, single-context development tests. They use the
same CBF-shielded, raw-action, moving-KL PPO algorithm, but **do not form a
joint gate**. v24 does not alter, rerun, or reinterpret the frozen v23 result.

| Scene | Base SR | Final SR | Delta SR | Delta fall | Repairs / regressions | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v23 pure lateral | 69.336% | 68.945% | -0.391 pp | +0.391 pp | 93 / 95 | FAIL |
| v24 pure low foot friction | 72.266% | 71.484% | -0.781 pp | +0.781 pp | 95 / 99 | FAIL |

Conclusion `C_bounded_updates_without_task_gain`: Moving-KL proximal PPO preserves a bounded online update path but did not improve task success in either representative deployment shift.

The lateral row is copied from the byte-frozen v23 final JSON (SHA-256
`7cbbbfc596e5ad39177c946998055fa460c646730a00385701b595f77cff0148`), not reconstructed from simulator episodes. The contact
row comes from the single frozen v24 execution. Confidence intervals remain
report-only in both studies.

## 中文

v23 与 v24 是两个相互独立的单场景开发实验。两者使用同一套 CBF shield、
raw-action、moving-KL PPO 算法，但**不存在联合 gate**。v24 不修改、不重跑、
也不重新解释已冻结的 v23 lateral 负结果。

综合结论 `C_bounded_updates_without_task_gain`：Moving-KL proximal PPO 能保持有界在线更新，但在两个代表性部署偏移中均未提高任务成功率。

lateral 行直接复制自 byte-frozen v23 final JSON，不重新运行仿真；contact 行来自
唯一一次冻结的 v24 正式执行。两项实验的置信区间都只作报告，不参与 gate。

# v30 Overnight Paper-Grade CBF-Teacher Experiment

> **Status: incomplete formal matrix.** This package publishes every aggregate result and failure from the frozen run. It is not a completed three-context paper-value result.

## 中文总结

v30 使用预先冻结的协议与源码执行，protocol SHA-256 为 `25a33f90cc2c3e6cf59805fef302d79666687552bc3424ef9969f60bc4f3e965`，formal freeze commit 为 `567a9fe2f43618904b98b6f8783ca60b886abd62`。正式运行开始后没有修改源码或阈值，也没有结果导向重跑。

- 唯一一次 smoke 通过全部指定检查。
- Development 六个 arm 均完成 8/8 rounds。按冻结规则选中 **A2**：residual teacher，`eta=0.25`，所有 intervention，weighted SmoothL1，teacher weight `1.0`。
- A2 在 development target 上的 CBF-on success 为 78.91%，相对 base 为 +2.73 pp；CBF-off 为 66.02%，相对 base 为 +2.73 pp；D0 为 89.06%，相对 base 为 -7.03 pp。
- Formal 完成 28/48 个计划 training rounds；F1 完整训练并完成唯一一次审计，F2 不完整，F3 在 rollout 前失败。
- 因 F2/F3 缺少完整 teacher/control 结果，预先定义的三场景论文价值标准**不能计算**。不能根据单独 F1 宣称方法有效或无效。

### Formal 执行状态

| context | A0 control | A2 teacher | audit |
|---|---:|---:|---|
| F1 | 8/8 complete | 8/8 complete | complete |
| F2 | 8/8 complete | stopped after 4/8 | not run |
| F3 | failed before rollout | failed before rollout | not run |

F2/A2 在第 5 轮 update 开始时触发冻结的 behavior-log-prob 路由检查：观测误差 `0.0005512237548828125`，冻结绝对容差 `0.0005`。该 arm 随即停止，未放宽阈值、未重跑。

F3/A0 与 F3/A2 各执行一次构建尝试，均在环境构建阶段报错：`could not broadcast input array from shape (12,3) into shape (10,3)`。诊断为冻结的 F3 11-step terrain 生成 12 个 stair-target patches（含顶部平台），而继承的 patch allocation 只有 10 个槽位。未修改冻结源码，也未重跑。

### F1 唯一完整 formal 结果

F1 audit 共 15 个 evaluation batches：target 6 个条件各 512 episodes，D0 3 个条件各 256 episodes，共 3,840 episodes。

| condition | base | A0 control | A2 teacher | teacher − control |
|---|---:|---:|---:|---:|
| target, CBF on | 77.15% | 75.39% | 72.66% | -2.73 pp |
| target, CBF off | 63.09% | 63.09% | 65.62% | +2.54 pp |
| D0, CBF on | 92.19% | 89.06% | 92.58% | +3.52 pp |

在 F1 中，teacher 的 target CBF-on success 相对 base 为 -4.49 pp，paired 95% CI 为 `[-9.77, +0.78] pp`；相对 control 为 -2.73 pp，CI 为 `[-8.01, +2.73] pp`。CBF-off success 相对 control 为 +2.54 pp，但 nominal internalization 指标有好有坏，并非一致改善。完整 success、fall、return、riser、CBF dependence、toe-riser risk、margin、repairs/regressions 与所有 paired CI 见 `formal/F1/context_results.json`。

F1 checkpoint monitor 未运行，因为冻结协议规定它只能在全部 formal training 完成后运行。`monitor/NOT_RUN.json` 记录了这一点。

## English summary

This is an honest partial evidence package from the frozen v30 run. The one permitted smoke passed, all six development arms completed, and the frozen selection rule chose **A2** (residual teacher, `eta=0.25`, all interventions, weighted SmoothL1, weight `1.0`).

Only F1 has a complete control/teacher training pair and final audit. On F1, A2 reached 72.66% target CBF-on success versus 75.39% for A0 and 77.15% for base. Its CBF-off success was 65.62% versus 63.09% for A0, and its D0 CBF-on success was 92.58% versus 89.06% for A0. These single-context effects do not support a three-context conclusion.

F2/A2 stopped after four completed rounds when the frozen behavior-log-probability audit measured `0.0005512237548828125` against an absolute tolerance of `0.0005`. Both F3 arms failed during environment construction because 12 terrain patches were assigned to 10 allocated slots. No frozen source, threshold, or result was changed, and no failed formal arm was rerun.

The predeclared paper-value assessment and F1 checkpoint monitor are unavailable because the formal matrix did not complete. `formal/combined_results.json` therefore has `complete: false` and `paper_value_assessment: null`.

## Published evidence

- `execution_status.json`: machine-readable top-level completion status.
- `smoke/smoke_summary.json`: aggregate evidence from the single smoke.
- `development/`: six-arm aggregate matrix and frozen A2 selection.
- `formal/F1/`: the complete F1 aggregate audit; episode-level telemetry remains outside Git.
- `formal/F2/` and `formal/F3/`: exact failure/status evidence.
- `formal/combined_results.{csv,json}`: incomplete formal matrix, explicitly marked incomplete.
- `training/`: 48 development and 28 available formal round-level aggregate rows.
- `figures/`: the development matrix, F1-only result, and execution-status figures in PNG and PDF.
- `SHA256SUMS`: checksums for every published file in this directory.

Checkpoints, raw simulator traces, and per-step/per-episode telemetry remain outside Git; this directory contains only aggregate CSV/JSON evidence and figures.

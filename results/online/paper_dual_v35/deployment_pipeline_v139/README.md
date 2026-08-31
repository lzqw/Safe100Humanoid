# v139 initializer + CBF-protected two-round Safe-FT

This experiment repositions v139 as a frozen simulation deployment initializer and tests
the next method stage: short PPO adaptation with **100% runtime-CBF execution**. PPO stores
the raw nominal policy action, the executed action is always CBF-filtered, the 8 cm
next-riser clearance and Eq. (27) dual reward are retained, and no teacher, DR, new CBF,
candidate selection, ratio search, or extra evaluation seed is introduced.

## Protocol

- Every evaluation cell contains 512 deterministic episodes at seed `201356200`.
- Within each context, all actors and CBF on/off conditions have the same initial-state
  signature, so episode rows are paired by environment ID.
- v139 Safe-FT uses 128 environments × 1,024 steps × 2 rounds at seed `201356300`.
- All Safe-FT executed actions are CBF-filtered; fixed round-2 final actors are reported.
- F1, F2, and F3 training times were 50.30, 49.83, and 50.19 seconds.
- One targeted protocol test passed; no full suite or parameter/checkpoint search was run.

## Main simulation table

| Context | Base v138 off | v139 off | v139 on | Safe-FT on | Safe-FT off |
|---|---:|---:|---:|---:|---:|
| F1 | 382/512 (74.61%) | 395/512 (77.15%) | 409/512 (79.88%) | 397/512 (77.54%) | 375/512 (73.24%) |
| F2 | 342/512 (66.80%) | 348/512 (67.97%) | 377/512 (73.63%) | 380/512 (74.22%) | 370/512 (72.27%) |
| F3 | 358/512 (69.92%) | 338/512 (66.02%) | 353/512 (68.95%) | 379/512 (74.02%) | 352/512 (68.75%) |
| Pooled | 1082/1536 (70.44%) | 1081/1536 (70.38%) | 1139/1536 (74.15%) | **1156/1536 (75.26%)** | 1097/1536 (71.42%) |

Safe-FT changes protected success by -2.34 pp on F1, +0.59 pp on F2, and +5.08 pp on
F3. Pooled protected success rises by 17/1,536 episodes (+1.11 pp), but the paired exact
McNemar result is `p=0.478`; this is directional rather than statistically decisive.
Nominal success rises by 16/1,536 (+1.04 pp, `p=0.524`) and is also context-dependent.

## Mixed-execution control revises the v139 interpretation

A matched control starts from the same v138 checkpoint and uses the same F2 training seed,
8 cm clearance, PPO, learning rate, and 524,288-transition budget as v139. The only change
is 100% filter-on instead of 25% on / 75% off. Both policies are compared after one update,
using the same 512 F2 filter-off initial states:

| Actor | F2 filter-off |
|---|---:|
| v139 mixed-execution round-1 | 348/512 (67.97%) |
| Matched full-filter round-1 | **358/512 (69.92%)** |

The control is +1.95 pp (`p=0.546`). Across F1/F2/F3, v138 base off is 1082/1536 while
v139 off is 1081/1536 (`p=1.000`). Therefore the previous 64-episode v139 screen/gate remains
a valid record of that protocol, but the new paired large-sample evidence does **not** support
a robust causal claim that mixed execution improved nominal deployment.

## What is supported

Runtime CBF improves v139 by 58/1,536 paired successes (+3.78 pp, `p=0.0209`) and Safe-FT
by 59/1,536 (+3.84 pp, `p=0.0177`). This is the clearest positive result.

After Safe-FT, riser-weighted protected metrics move in the intended direction:

| Metric | v139 + CBF | Safe-FT + CBF | Change |
|---|---:|---:|---:|
| Intervention events / riser | 2.1171 | 2.0795 | -1.78% |
| Intervention steps / riser | 4.4314 | 4.3771 | -1.23% |
| Mean correction norm | 1.1335 | 1.1147 | -1.65% |
| Unsafe overlap steps / riser | 0.7811 | 0.7636 | -2.24% |

These reductions are small and come from one training seed. The defensible conclusion is:

1. CBF runtime protection has a reproducible pooled benefit.
2. Two-round protected adaptation has a positive pooled direction, strongest on F3, but is
   not yet robust or context-consistent.
3. v139 is useful as a frozen initialization artifact, but mixed execution itself is not
   established as the cause of improvement.
4. The paper's real-robot safe-online-adaptation claim is not complete until the protected
   hardware protocol in `REAL_ROBOT_PROTOCOL.md` is executed.

## Files and provenance

- Code commit: `7edc4da60a089985681cb6b78243b53cfb730003`.
- `main_results.csv`: main table in machine-readable form.
- `paired_comparisons.csv`: paired discordance counts and exact McNemar results.
- `decision_summary.json`: formal interpretation and pooled shield metrics.
- `eval_seed201356200/`: all per-episode CSV and summary JSON files.
- `training/`: control and Safe-FT round metrics/training summaries.
- `checkpoints/`: the three fixed Safe-FT round-2 model binaries.
- `checkpoint_index.json`: local/remote paths and SHA-256 identities.
- `deployment_artifacts/v139_actor_405x12.onnx`: deterministic v139 deployment
  candidate with embedded observation normalization; this is an actor artifact, not a
  complete or approved robot controller.
- `deployment_artifacts/actor_io_contract.json`: exact term-major five-history input and
  12-action simulation interface that a hardware bridge must reproduce.
- `real_robot_readiness.json`: current hardware readiness audit and hard-stop gaps.
- `real_robot_run_manifest.template.json`: operator/safety-reviewer freeze template required
  before any motor command is sent.
- `src/tasks/stairs_cbf/real_robot_reference.py`: offline executable reference for the
  term-major history, action mapping, and slope-0.8 x-z CBF; it has no robot transport and
  cannot send motor commands.

Implementation: `src/tasks/stairs_cbf/paper_deployment_pipeline.py` and
`experiments/scripts/refine_paper_dual_v35.py`. The actor-only exporter is
`experiments/scripts/export_stairs_actor_onnx.py`; its offline bridge contract is covered by
`experiments/tests/test_real_robot_reference.py`.

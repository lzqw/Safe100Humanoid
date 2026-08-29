# Filter-free deployment ablation (v140)

Formal simulation ablation from the frozen v139 actor. Adaptation uses four rounds × 128 environments × 1024 steps, three preregistered seeds, and fixed round-4 publication. Primary deployment evaluation is deterministic CBF-off.

| Arm | Filter | CBF reward | F1 off | F2 off | F3 off | Mean off | Off Δ | Shield gap | Training falls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | — | — | 0.730 | 0.689 | 0.693 | 0.704 | +0.000 | +0.035 | — |
| Nominal FT | Off | No | 0.736 | 0.680 | 0.674 | 0.697 | -0.008 | +0.045 | 960.7 |
| Reward-only FT | Off | Yes | 0.731 | 0.671 | 0.686 | 0.696 | -0.008 | +0.054 | 966.0 |
| Filter-only FT | On | No | 0.738 | 0.707 | 0.665 | 0.703 | -0.001 | +0.045 | 850.3 |
| Dual Safe-FT | On | Yes | 0.754 | 0.675 | 0.670 | 0.700 | -0.005 | +0.052 | 864.3 |

Main claim supported: **False**.

The preregistered simulation claim gate failed. These artifacts must not be presented as evidence that runtime CBF can be removed on the physical robot. Any physical follow-up is restricted to CBF-on adaptation and shadow-only filter-free measurements.

Global fall- and executed-violation-free simulation training supported: **False**. A false result means the filtered methods improved the local barrier constraint but did not prove globally fall-free online adaptation.

## Hardware proxy

| Arm | Nominal-sim off | Hardware-proxy off | Absolute drop | Fall rate |
|---|---:|---:|---:|---:|
| Frozen | 0.704 | 0.187 | +0.518 | 0.813 |
| Dual Safe-FT | 0.700 | 0.189 | +0.511 | 0.811 |
| Filter-only FT | 0.703 | 0.189 | +0.515 | 0.811 |

The proxy combines actor sensor noise/encoder bias, 1- and 2-step action delay, 0.95 actuator gain, +1 cm stair-height estimate bias, ±1.5 cm tread perturbation, friction variation, and command delay. All proxy evaluations execute CBF-off.

## Offline ONNX/bridge validation

| Context | Bridge parity | PyTorch bridge p95 (ms) | 20 ms deadline |
|---|---:|---:|---:|
| F1 | True | 0.238 | True |
| F2 | True | 0.223 | True |
| F3 | True | 0.235 | True |

Each representative fixed round-4 Dual actor is exported to ONNX and checked on deterministic five-frame bridge inputs. Latency covers observation assembly, actor inference, and 12-to-29 target mapping on one CPU thread. ONNX ReferenceEvaluator latency is retained in the JSON reports as portability evidence, not as a production-runtime claim.

## Contents

- `final_results.json` and `main_table.csv`: primary result and claim checks
- `learning_curves.csv`: rounds 0/1/2/4
- `training_safety.csv`: per-run falls, violations, recoveries, and min h
- `training_safety_curves.csv`: per-round safety metrics versus transitions
- `task_learning_curves.*`: CBF-off success and shield-gap figures
- `training_safety_curves.*`: executed violations and falls figures
- `paired_statistics.*`: report-only paired-bootstrap intervals and repair/regression counts
- `factorial_contrasts.csv`: report-only Filter, Reward, and interaction contrasts
- `evaluation/checkpoint_summaries/`: all 222 paired-condition summaries
- `hardware_proxy/`: 42 Frozen/Filter-only/Dual proxy summaries and aggregate table
- `training/`: all 36 training summaries and round metrics
- `checkpoints/`: fixed round-4 Dual Safe-FT models for F1/F2/F3, seed 201357000
- `checkpoint_index.json`: hashes for all 36 fixed round-4 models
- `deployment/`: three ONNX actors plus bridge parity/latency reports
- `deployment_index.json`: deployment artifact hashes and compact checks
- `COMPLETION_AUDIT.json`: one structural audit of all formal deliverables

Episode CSVs and redundant checkpoints remain in the archived 4080 run; the repository contains the compact evidence needed to reproduce tables.

# v57 Paper Training Domain Randomization

v57 restores the native G1 training-time domain randomization that was absent
from the earlier v35--v56 continuation runs.  The fixed 18.4 cm deployment
environment used by the acceptance gate remains unchanged.  The implementation
supports observation corruption, encoder bias, foot friction, base-COM shifts,
optional pushes, and a scalar randomization strength.

## Screening result

All runs warm-started from the same F2 checkpoint
`3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`
and used the runtime CBF filter during training.  Each rollout row evaluates the
checkpoint produced by the preceding update.

| Run | Seed | Training rollout success by round | Runtime |
|---|---:|---|---:|
| Static DR, strength 1.0, A0 | 201352401 | 27/73, 24/76, 25/77, 19/77 | 62.9 s |
| Static DR, strength 0.25, A0 | 201352401 | 44/66, **50/64**, 44/65, 39/70 | 60.5 s |
| Static DR, strength 0.25, A2 residual teacher | 201352402 | 35/65, 40/65 | 26.0 s |

Full-strength randomization immediately damaged the existing final-height gait.
At 25% strength, A0 briefly reached 78.13% under the training filter, so its
aligned `round_01.pt` checkpoint was selected for the single independent gate.
The A2 branch stayed below A0 and was stopped without extra evaluation.

## Independent gate and decision

The selected DR25 A0 checkpoint was evaluated with deterministic policy means,
the runtime filter disabled, a fresh reset, and seed `201352412`:

- success: **37/64 (57.81%)**
- fall: 27/64 (42.19%)
- acceptance threshold: 48/64 (75%)
- decision: **rejected**

The result does not support applying domain randomization only after the policy
has already specialized to fixed 18.4 cm stairs.  The next justified experiment
is to introduce DR while the stair-height curriculum is still progressing,
rather than spend more samples on this post-hoc branch.

## Provenance and files

- DR restoration source commit: `f77adb93195f4290affba8aee66bbcbf296360a6`
- DR strength-scaling source commit: `531798aa8540aefe00b577fa47087690c4df007c`
- Selected checkpoint SHA-256:
  `0d7735ae3d8918278f03a0842718d2e11c6d6fd9e3ef9146357610af0633e973`
- Selected actor SHA-256:
  `f50e6e53157fec157082ca387a53a748161d2166d198387a31ddf769209aa2a0`
- `decision_summary.json`: compact decision and provenance record.
- `*/training_summary.json`: exact configurations and hashes.
- `*/round_metrics.{json,csv}`: complete per-round diagnostics.
- `dr25_a0_gate_seed201352412/`: summary and all 64 episode records.
- `dr25_a0/round_01.pt`: exact checkpoint used by the independent gate.

Implementation: `src/tasks/stairs_cbf/paper_dual_v35.py` and
`experiments/scripts/refine_paper_dual_v35.py`.

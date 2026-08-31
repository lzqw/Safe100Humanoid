# v52 KL-Targeted Geometry-Aware Paired CBF PPO

v52 tests the specific bottleneck exposed by v51: its formal update consumed
only `2.98e-6` of a `0.001` reference-KL cap.  The new line search expands the
same paired on/off consensus-gradient direction only while every rollout batch
retains positive PPO surrogate gain, and stops at a requested KL target.

The short smoke encountered a 5/6-positive raw direction and correctly refused
to expand it.  The formal run then used three paired 16 x 512 rollout batches
whose raw direction was positive on all six conditions.

| Formal v52 result | Value |
|---|---:|
| Training transitions | 49,152 |
| RTX 4080 training time | 99.77 s |
| Selected adapter scale | 14.0923x |
| Selected reference KL | 0.000249991 |
| Positive post-update batches | **6/6** |
| Minimum batch surrogate gain | +0.00026076 |
| Untouched seed `201351012`, filter off | **42/64 (65.63%)** |
| Decision | rejected |

The larger update improved every collected on/off surrogate but reduced
deployment performance on the untouched gate.  Filter-on and further seeds
were not run.  This falsifies the hypothesis that simply filling more of the
available KL budget would solve v51's one-episode shortfall; training-batch
surrogate gain is not a sufficient model-selection signal for this shift.

Provenance:

- Source commit: `356c6880de3bbb65b9a4702e396ce6689a68c1f6`.
- Training seeds: `201350722, 201350732, 201350742`.
- Candidate checkpoint SHA-256: `ec04c5807e62d493e6164fd8901fe13e831af7f23a4121d091966db57e61049b`.
- Candidate actor SHA-256: `45c6cf80dda26a75a33a59aec6677ab84b5b0ba05a0ef3b79777d89c7a42d64a`.
- Legacy actor parameter change: exactly zero.

Files:

- `formal_training_summary.json`: full rollout, line-search, KL, and integrity record.
- `formal_candidate.pt`: exact rejected formal candidate.
- `untouched_seed201351012_filter_off_{summary.json,episodes.csv}`: deployment evidence.
- `smoke_training_summary.json`: safe refusal of a 5/6-positive smoke direction.
- `*_execution_started.json`: immutable launch records.

Implementation: `experiments/scripts/refine_observable_cbf_ppo_v52.py` and
the shared v51 core.


# v50 Six-Seed Deployable CBF-Geometry Adapter

v50 enlarges the v49b rescued-only data collection from three short training
seeds to six new paired filter-on/off seeds with 16 environments per seed.  It
keeps the same deployable 410-D actor interface and trains only the five new
first-layer input columns; every legacy actor parameter remains unchanged.

| Stage | Result | Decision |
|---|---:|---|
| Paired data collection | 37,549 transitions; 17 rescued episodes; 555 teacher transitions | offline gate passed |
| Teacher correction cosine | 0.89922 | direction learned |
| Active-state forward KL | 0.00493496 | within 0.005 limit |
| Untouched F2 seed `201350972`, filter off | **45/64 (70.31%)** | deployment gate failed |

The untouched evaluation did not reach the registered 48/64 (75%) threshold,
so filter-on evaluation was intentionally not run.  The larger and more varied
rescued-only dataset therefore did not fix cross-seed generalization, and this
candidate is rejected rather than presented as a robust result.

Key provenance:

- RTX 4080 training time: 165.19 s.
- Training seeds: `201350622, 201350632, 201350642, 201350652, 201350662, 201350672`.
- Candidate checkpoint SHA-256: `796f8f0a846bcffff473db5498138a035e72aed115d674a75791a21ae30d2233`.
- Candidate actor SHA-256: `5ef1cee60d5091a09aae1d08cabd1aa0be78ee452647edb0a910dfea6db13251`.
- Inactive-geometry behavior remains exactly the base policy.

Files:

- `training_summary.json`: full rollout, optimization, KL, model-hash, and provenance record.
- `untouched_seed201350972_filter_off_summary.json`: aggregate untouched-seed gate.
- `untouched_seed201350972_filter_off_episodes.csv`: episode-level evidence.
- `candidate.pt`: exact rejected 410-D candidate checkpoint, retained for reproducibility.
- `execution_started.json`: immutable launch record.


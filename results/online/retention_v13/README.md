# v13 state-conditioned retention evidence

This directory contains compact, machine-readable evidence for the matched
v12-global-anchor versus v13-fixed-bank-anchor experiment. Runtime CBF remains
enabled; no checkpoint here supports a claim that the shield can be removed.

Tracked bank manifests:

- `banks/D0_actor_observations.json`
- `banks/DQNH_actor_observations.json`

The manifests prove the fixed bank sizes, stage balance, actor-only schema,
source-checkpoint identity, and both observation/file SHA-256 values. The two
approximately 39 MB tensor banks are reproducible from
`experiments/scripts/collect_retention_banks_v13.sh` and are deliberately not
stored in Git.

Formal arm summaries and a compact result ledger are added only after the
matched runs and disjoint 512-episode/domain promotion audit finish. See
`docs/STATE_CONDITIONED_RETENTION_V13.md` for the objective and promotion
contract.

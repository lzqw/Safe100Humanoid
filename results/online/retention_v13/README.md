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

## Result

The matched experiment and disjoint final audits are complete. The overall
decision is `rollback`:

- all three Arm A global-anchor candidates were rejected;
- all three Arm B fixed-bank candidates were rejected;
- the final 512-episode/domain Arm A and Arm B audits were both rejected;
- the accepted actor tensors are exactly unchanged in both arms;
- runtime CBF remains enabled;
- the 3 base seeds x 5 hidden-context expansion was not run because Arm B did
  not pass the predeclared Level A gate.

The final Arm B audit changed DQH success from 87.70% to 88.09%, fall from
12.11% to 11.91%, return by +0.144, and CBF/riser by -0.056. The paired DQH
return 95% interval was `[-0.328, 0.612]`, and D0 success was 92.19%, below the
93.83% retention floor. The exact rejection reasons were:

```text
target metrics show no strict improvement
target task metrics show no strict improvement
D0 retention bound violated
```

`key_results.json` is the compact authoritative ledger. It contains all six
formal rounds, both 512-episode/domain audits, paired intervals, bank metadata,
checkpoint hashes, source-summary hashes, and the explicit no-scale decision.
Its SHA-256 is
`4ebe894ecb051457268aa1d5e1f4b299c43f90c318861a0e8cfc720e9a6d9d6e`.

Raw audit summaries remain in the experiment artifact store and are identified
by these hashes:

- Arm A audit: `e54b3d7aa8215fdd6d01ea595d24b0be4e2b7c3e290f57971ac18f329e559d25`;
- Arm B audit: `f87fb217dd09d28ca94b07cb7baf51a244c7684715a36561694052c780c142ff`;
- Arm A online summary: `82d8a6fb242095390aee5bf00d34b472dda635ee70eb31d4391d867212773584`;
- Arm B online summary: `f8d9aec5a1401fdd5c6efba6a4c80ab3107437cb1dbc63e1015c35212c8120d2`.

See `docs/STATE_CONDITIONED_RETENTION_V13.md` for the protocol, full tables,
interpretation, and claim boundary.

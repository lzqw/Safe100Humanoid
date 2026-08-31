# v111 paired terminal-outcome contrast trajectory adapter

v111 replaces rescued-only supervision with a paired terminal-outcome contrast.
For identical initial states, an `off failed / on succeeded` episode gives the
complete counterfactual trace sign `+1`, while an `off succeeded / on failed`
episode gives sign `-1`. Thus the adapter learns both where the CBF helps and
where it destroys otherwise successful task behavior. Every discordant episode
has unit total weight. The teacher uses same-state filter-off CBF projections,
20 pre-intervention steps with decay `0.9`, and 50 post-intervention steps.
Only the 5,120 persistent-geometry input weights are trainable.

## Training evidence

Four seeds with 32 paired environments each completed on the shared RTX 4080
SUPER in 136.81 seconds.

| Metric | Result |
|---|---:|
| paired initial states | 128 |
| filter-off successes | 83/128 = 64.84% |
| filter-on successes | 91/128 = 71.09% |
| rescued / positive episodes | 32 |
| harmed / negative episodes | 24 |
| weighted teacher transitions | 3,675 |
| teacher correction cosine after update | 0.065752 |
| teacher weighted distance | 0.0612153 -> 0.0612072 |
| reference forward KL | 1.82224e-6 |
| adapter interpolation scale | 1.0 |
| legacy first-layer max change | 0 |

The positive and negative outcome-conditioned directions almost cancel: the
full gradient norm is only `0.002636`, compared with `0.01950` for rescued-only
v110, and it uses just 3.6% of the allowed KL budget.

## Aligned filter-off screen

One deterministic 64-episode filter-off screen used development seed
`201353480`, matching v109/v110 candidate comparison. v111 obtained
**38/64 = 59.375%**, with mean reached riser `7.625`. This recovers two episodes
relative to v110's 36/64 on the same screen, but remains far below the 75%
trigger. No independent gate or additional rollout was run; v111 is rejected
and v79 remains selected at 139/193 = 72.02%.

The paired outcome contrast shows why a global CBF-action adapter cannot solve
the task: beneficial and harmful CBF effects are both frequent and their policy
directions nearly cancel. The next method needs to learn a state-dependent
counterfactual value/gating function rather than average those directions into
one global adapter update.

## Reproducibility

- source commit: `b7f586532e146621079bb39cd530f105456637eb`
- targeted v111 tests: 1 passed
- input v79 checkpoint SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- zero-adapter checkpoint SHA-256: `0ec10db00e6955a5da4c6c6be7098e9fafd1185acb1d56b48c304eace81c5005`
- candidate actor SHA-256: `c07bb5524c2c70a6e8764f380a36e24639334b97d328b335911dddefacbf3a73`
- candidate checkpoint SHA-256: `e52dbc2bb96461af661e484c98bdc892a420ff2e3670e76c50530f065b4bc109`
- remote candidate: `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/paired_outcome_contrast_v111_b7f5865_4x32_s201353400/candidate.pt`

The rejected binary is not committed. Exact paths and hashes are in
[`checkpoint_index.json`](checkpoint_index.json), with complete machine-readable
training and screen evidence in this directory.

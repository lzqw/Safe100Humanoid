# v95 persistent-geometry paired CBF PPO

v95 puts the v94 bilateral, pre-toe-off next-riser geometry into the paper's
filtered-training loop instead of regressing local safe actions. For each of
three seeds, one filter-on and one filter-off rollout shared the same initial
state. PPO stored the nominal policy action and used task+CBF reward. The two
gradients for each seed were averaged, then the three paired gradients were
aggregated by coordinate median. Only the 5,120 new geometry input weights were
updated.

## Training evidence

The run used six batches of `16 × 512` steps, or 49,152 transitions. All six
post-update PPO surrogates improved:

- minimum batch surrogate gain: `+0.000125963`
- mean filter-on surrogate gain: `+0.000140788`
- mean filter-off surrogate gain: `+0.000267107`
- reference forward KL: `4.99966e-5`
- legacy 405-D first-layer max change: `0`
- interpolation scale selected by the KL cap: `0.69970703125×`

The offline gate passed. However, paired-seed gradient cosine averaged only
`0.02015` and reached `-0.06865` at minimum, showing that outcome-credit
directions remained weakly consistent across seeds. Training took 139.65
seconds on the local GTX 1660 SUPER.

## Single untouched filter-off gate

Exactly one unseen deterministic filter-off gate was run with seed `201352920`.
It obtained **44/64 = 68.75%**, below the fixed 75% target. v95 is therefore
**rejected**. No filter-on comparison, base rerun, extra seed, or larger formal
gate was run after observing the result. The selected version remains v79.

## Reproducibility

- source commit: `10adeebdafabc04d349603d78e2348a460002b8b`
- targeted tests: 9 passed
- input v79 checkpoint SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- candidate actor SHA-256: `a03712eeeaab97c5723402a3977f12ca1b1f02cf8e4bd799a3bfaec8871aa816`
- candidate checkpoint SHA-256: `e6b3c68400e03994b54b7470e5466893ef3c7a325810a879f52150a96e586b6a`
- local candidate: `/home/lzqw/PycharmProject/safe100/HUMANOID/artifacts/paper_dual_v35/persistent_geometry_ppo_v95_10adeeb_3x16x512/candidate.pt`

The rejected binary is not committed. Exact identity and path are in
[`checkpoint_index.json`](checkpoint_index.json); training and gate metrics are
in [`decision_summary.json`](decision_summary.json).

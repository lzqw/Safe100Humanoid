# v96 persistent-geometry filter-free elite refinement

v96 combines the earlier v42 filter-free successful-trajectory signal with the
v94 persistent bilateral next-riser observation. The 405-D v79 actor was
expanded to 415-D with ten exact-zero input columns. Four stochastic
filter-off first-episode rollouts supplied sampled-action targets; only
transitions belonging to successful episodes were retained. One full-batch
SGD direction changed only the 5,120 new geometry input weights. The direction
was then scaled to the fixed reference-KL cap while the legacy 405-D route
remained exactly unchanged.

## Training evidence

- stochastic training rollouts: `173/256` successful episodes
- all rollout transitions: `103,914`
- successful-episode transitions: `77,875`
- exploration standard deviation: `0.05` for every action coordinate
- elite target distance: `0.1696496852 -> 0.1696479108`
- aggregate exploration-direction cosine: `0.0049973`
- reference forward KL: `4.99910e-5` (cap `5e-5`)
- trust-direction scale: `22.8984375x`
- legacy 405-D first-layer max change: `0`
- rollout collection time: `125.54 s`
- reused-data trust scaling time: `17.25 s`

The offline gate passed, but the very small direction cosine already indicated
that successful-trajectory exploration residuals were weakly consistent.

## Single untouched filter-off gate

Exactly one unseen deterministic filter-off gate was run with seed
`201352990`. It obtained **35/64 = 54.69%**, below the fixed 75% target. v96 is
therefore **rejected** and does not replace v79. No filter-on comparison, base
rerun, second seed, or larger formal gate was run after observing this result.

## Reproducibility

- final source commit: `b339bcb7f3db78b26d46645d12bc8403f5943dbf`
- initial collection source commit: `f6380dc624b92c8d317eb463adddf15e295c621f`
- targeted tests: 9 passed
- input v79 checkpoint SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- candidate actor SHA-256: `da16a2fb7f9e373e59c04a742a6a3d645bb2c000f823741616d6acc6b9b5e5a3`
- candidate checkpoint SHA-256: `0c9d8076a303f02315b177e6d3219a08ab23aa3fc5cf8e8e6d7a3f88255c88bb`
- local candidate: `/home/lzqw/PycharmProject/safe100/HUMANOID/artifacts/paper_dual_v35/persistent_geometry_elite_v96_b339bcb_4x64/candidate.pt`

The rejected binary and 172 MB rollout dataset are not committed. Their exact
identities and local paths are recorded in
[`checkpoint_index.json`](checkpoint_index.json); compact training and gate
metrics are in [`decision_summary.json`](decision_summary.json).

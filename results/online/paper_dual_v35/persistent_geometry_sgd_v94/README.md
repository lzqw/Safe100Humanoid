# v94 persistent next-riser geometry adapter

v94 addresses a timing gap in v92/v93. Their geometry became nonzero only after
a foot was already airborne and the runtime CBF activation window had opened.
The v94 policy instead receives ten deployable values: horizontal clearance,
vertical clearance, sloped barrier, contact, and validity for each foot. The
next riser becomes visible during approach, including while both feet remain
planted, so the policy can plan lift before toe-off.

The historical 405-D policy was expanded to 415 inputs with zero new columns.
Only the 5,120 new first-layer weights were trainable. Four new seeds with 64
paired filter-off/filter-on environments each produced 61 matched rescues and
2,306 effective teacher transitions.

## Training evidence

Across the 256 paired initial states, the base filter-off execution succeeded in
169 episodes and filter-on succeeded in 191. This +22 net outcome is stronger
than v93 and supports the early-geometry diagnosis. One full-batch SGD update
then produced:

| Metric | Before | After |
|---|---:|---:|
| teacher correction cosine | 0 | **0.517290** |
| teacher weighted distance | 0.133976 | **0.133686** |
| active/relevant-state forward KL | 0 | **4.99643e-5** |
| legacy first-layer max change | 0 | **0** |

The unprojected update exceeded the KL cap, so interpolation selected
`0.252685546875×`. Training took 248.92 seconds on the local GTX 1660 SUPER.

## Single untouched filter-off gate

Exactly one unseen deterministic filter-off gate was run with seed `201352870`.
It obtained **45/64 = 70.3125%**, below the fixed 75% target. v94 is therefore
**rejected**. No filter-on comparison, base rerun, extra seed, or formal
256-episode gate was run after seeing this result. The selected version remains
v79.

## Reproducibility

- source commit: `adf83d4bfae6406d1d3abc439e06b9cef0552aa5`
- targeted tests: 8 passed
- input v79 checkpoint SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- zero-adapter checkpoint SHA-256: `3aeb0bd4e4a528144be8e370fbd339033676b0de9f45caf4796ed1b9f361e89d`
- candidate actor SHA-256: `0f4e20e2532592cdd0ba17bda6ddffcbfadcdd25b986b35c27bb16f91a156dd6`
- candidate checkpoint SHA-256: `249dd5bda299008e6adf7e62bc437c2fc95bb2045bb102d3799c064194bae6a0`
- local candidate: `/home/lzqw/PycharmProject/safe100/HUMANOID/artifacts/paper_dual_v35/persistent_geometry_sgd_v94_adf83d4_4x64/candidate.pt`

The rejected binary is not committed. Exact hashes and paths are recorded in
[`checkpoint_index.json`](checkpoint_index.json), with machine-readable metrics
in [`decision_summary.json`](decision_summary.json).

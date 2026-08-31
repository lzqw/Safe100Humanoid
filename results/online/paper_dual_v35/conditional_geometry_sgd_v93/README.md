# v93 conditional geometry adapter

v93 follows the CBF-RL filtered-training idea while addressing the limitation
found in v92: one linear 5-D geometry adapter averaged corrections from opposite
swing feet and barrier phases. The new deployment interface expands the five
current-state CBF geometry values into four mutually exclusive blocks:

1. left foot / unsafe barrier;
2. left foot / safe barrier;
3. right foot / unsafe barrier;
4. right foot / safe barrier.

Each block contains horizontal clearance, vertical clearance, sloped barrier,
and a mask, giving a 16-D deployable geometry input and a 421-D actor. Only the
8,192 new first-layer weights were trained. The historical 405-D actor weights
remained bit-exact, and inactive geometry still reproduces the base policy.

## Training result

Training used four new seeds with 64 paired filter-off/filter-on environments
per seed. It collected 99,548 transitions, including 51 matched-rescue episodes
and 1,934 effective teacher transitions. One full-batch SGD update was applied.

| Metric | Before | After |
|---|---:|---:|
| teacher correction cosine | 0 | **0.641110** |
| teacher weighted distance | 0.135701 | **0.135405** |
| active-state forward KL | 0 | **3.02449e-5** |
| legacy first-layer max change | 0 | **0** |

The offline gate passed without interpolation (`adapter_interpolation_scale=1`).
Training took 291.72 seconds on the local GTX 1660 SUPER.

## Single untouched filter-off gate

As predeclared, exactly one deterministic filter-off gate was run with unseen
seed `201352810` and 64 episodes. The result was **45/64 = 70.3125%**, below the
75% target. The candidate is therefore **rejected**. No filter-on comparison,
base rerun, extra seed, or formal 256-episode gate was added after observing the
result. The global selected checkpoint remains v79.

## Reproducibility and checkpoint index

- source commit: `f94c5a42d2c3d33f9cc3b26ff906264806879554`
- targeted tests: 7 passed
- input v79 checkpoint SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- zero-adapter checkpoint SHA-256: `4a295e1824e933b74aa6d6f0e8ace8eb60244498315fac6ccabfe476031877a0`
- candidate actor SHA-256: `fd186d494a70eb1f507165ae23f9f68a11e01e02cb3dd686a78a3044947c9c7c`
- candidate checkpoint SHA-256: `9231215d7d030874cc0287bd438e0161cf14bdfec3a5a4568eaacf746bc4ec40`
- local candidate: `/home/lzqw/PycharmProject/safe100/HUMANOID/artifacts/paper_dual_v35/conditional_geometry_sgd_v93_f94c5a4_4x64/candidate.pt`

The rejected checkpoint binary is not committed to GitHub. Its exact hash and
local path are retained in [`checkpoint_index.json`](checkpoint_index.json).
Machine-readable training and gate metrics are in
[`decision_summary.json`](decision_summary.json).

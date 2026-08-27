# v110 deployment-state counterfactual rescue trajectory

v110 corrects the post-divergence state/action mismatch exposed by v109. The
matched filter-on rollout is used only to identify rescued episodes. Every
teacher target is then recomputed by the CBF on the corresponding filter-off
deployment state itself. The first real counterfactual intervention is traced
backward for 20 steps with decay `0.9` and followed for 50 steps. Each rescued
episode has total weight one, and only the 5,120 first-layer weights connected
to the ten persistent geometry inputs are trainable.

## Training evidence

Four seeds with 32 paired environments each completed on the shared RTX 4080
SUPER in 129.60 seconds.

| Metric | Result |
|---|---:|
| paired initial states | 128 |
| filter-off successes | 79/128 = 61.72% |
| filter-on successes | 90/128 = 70.31% |
| matched rescue episodes | 30 |
| weighted teacher transitions | 1,982 |
| teacher correction cosine after update | 0.453880 |
| teacher weighted distance | 0.062279 -> 0.062002 |
| reference forward KL | 4.99941e-5 |
| adapter interpolation scale | 0.687256 |
| legacy first-layer max change | 0 |

The corrected same-state targets make the offline direction cleaner than v109,
but the paired training ceiling is still below 75%.

## Aligned filter-off screen

One deterministic 64-episode filter-off screen was run with the same development
seed `201353480` used for v109 candidate comparison. v110 obtained
**36/64 = 56.25%**, with mean reached riser `7.8125`. It is below the fixed 75%
trigger, so no independent gate or additional rollout was run. v110 is rejected
and the selected version remains v79 at 139/193 = 72.02%.

This negative result is informative: removing the state/action mismatch improves
the supervised CBF metric but worsens deployment success. Failure-state local
CBF corrections are therefore not a reliable task-success direction, even with
persistent geometry and temporal backtracing. The next update must optimize a
sequence-level outcome/value objective rather than another local correction
target.

## Reproducibility

- source commit: `bdd57f0669f7406340c346422c09351f393e8258`
- targeted v110 tests: 1 passed
- input v79 checkpoint SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- zero-adapter checkpoint SHA-256: `74f834b1352a12032c84af113381028da7ca3224e7cfff021eee845c3e9069c6`
- candidate actor SHA-256: `aaeba19083c43f86fd53de4d83d1d2064e0bc843bdc96a0badc517ae40501eda`
- candidate checkpoint SHA-256: `473724b36cc416ab0ae5f011d4daee5645aaef9b1579369843476cab050e0ca6`
- remote candidate: `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/deployment_counterfactual_v110_bdd57f0_4x32_s201353400/candidate.pt`

The rejected binary is not committed. Exact paths and hashes are recorded in
[`checkpoint_index.json`](checkpoint_index.json); the full machine-readable
training and screening evidence is included in this directory.

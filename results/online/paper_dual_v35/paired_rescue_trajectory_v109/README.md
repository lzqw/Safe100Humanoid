# v109 paired counterfactual rescue trajectory adapter

v109 tests whether the persistent next-riser geometry can internalize an actual
paired rescue trajectory instead of only imitating instantaneous CBF
interventions. For identical initial states, filter-off observations are the
deployment inputs and the same-time filter-on safe actions are the teacher
targets. The first intervention is traced backward for 20 steps with decay
`0.9`, followed for 50 steps, and every rescued episode has total teacher weight
one. Only the 5,120 weights connecting the ten new geometry inputs to the first
actor layer are trainable; the legacy 405-D policy path is frozen exactly.

## Training evidence

Eight seeds with 32 paired environments each completed on the shared RTX 4080
SUPER in 262.96 seconds.

| Metric | Result |
|---|---:|
| paired initial states | 256 |
| filter-off successes | 161/256 = 62.89% |
| filter-on successes | 184/256 = 71.88% |
| matched rescue episodes | 69 |
| weighted teacher transitions | 4,899 |
| teacher correction cosine after update | 0.392411 |
| teacher weighted distance | 0.126094 -> 0.125850 |
| reference forward KL | 4.99998e-5 |
| adapter interpolation scale | 0.170166 |
| legacy first-layer max change | 0 |

The offline direction checks passed, but both the distance improvement and the
paired filter-on ceiling remained small relative to the deployment target.

## Aligned filter-off screen

One deterministic 64-episode filter-off screen was run with seed `201353480`.
It obtained **41/64 = 64.0625%**, with mean reached riser `7.96875`. This is below
the fixed 75% trigger, so no independent gate or additional rollout was run.
v109 is rejected and the selected version remains v79 at 139/193 = 72.02%.

## Reproducibility

- source commit: `4a687a5dd7874db3e1563759d48e7f63661775f2`
- targeted v109 tests: 2 passed
- input v79 checkpoint SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- zero-adapter checkpoint SHA-256: `07888bbd29ddc681702421d05b4b21e774698b02ed604b80a6378df935970366`
- candidate actor SHA-256: `0ab064fec72b0756b42f09cba818d833d3eaa6c99acea424d0734a5c034accaf`
- candidate checkpoint SHA-256: `bcfc4baa6bb70a1686155654a618dd6007582161e50a414a53da1d516b10b3be`
- remote candidate: `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/paired_trajectory_v109_4a687a5_8x32_s201353400/candidate.pt`

The rejected binary is not committed. Its exact path and hashes are recorded in
[`checkpoint_index.json`](checkpoint_index.json); machine-readable training and
screening evidence is included in this directory.

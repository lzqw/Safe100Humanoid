# v112 state-conditioned paired-outcome contrast

v112 gives the v111 rescued/harmed terminal contrast access to the complete
415-D first-layer state representation. Instead of training only ten geometry
columns, all 212,480 first-layer weights may change, allowing the policy update
to condition CBF usefulness on proprioception, gait history, commands, previous
actions, and persistent stair geometry. The remaining actor is frozen. A KL
trust region is measured over every filter-on state and every transition from
the original filter-off success episodes.

## Training evidence

Four seeds with 32 paired environments each completed on the shared RTX 4080
SUPER in 137.34 seconds.

| Metric | Result |
|---|---:|
| paired initial states | 128 |
| filter-off successes | 88/128 = 68.75% |
| filter-on successes | 86/128 = 67.19% |
| rescued / positive episodes | 28 |
| harmed / negative episodes | 30 |
| weighted teacher transitions | 3,765 |
| trust transitions | 90,951 |
| trainable first-layer parameters | 212,480 |
| teacher correction cosine after update | 0.020416 |
| teacher weighted distance | 0.0609964 -> 0.0609891 |
| reference forward KL | 4.99368e-5 |
| interpolation scale | 0.170654 |

The wider layer can express state dependence, but the paired positive and
negative policy gradients still nearly cancel. The unconstrained update also
exceeded the trust region and was projected to 17.1% of its displacement.

## Aligned filter-off screen

One deterministic 64-episode filter-off screen used development seed
`201353480`, identical to v109-v111 candidate comparison. v112 obtained
**38/64 = 59.375%**. Mean reached riser improved to `8.234375`, but the number
of completed episodes did not improve over v111. The result is below the 75%
trigger, so no independent gate or additional rollout was run. v112 is rejected
and v79 remains selected at 139/193 = 72.02%.

This rules out first-layer capacity as the immediate bottleneck. The next
implementation should represent CBF usefulness explicitly with a gate and a
separate residual expert, rather than forcing one signed regression gradient to
both classify treatment effect and produce the action correction.

## Reproducibility

- source commit: `829420faa5ed30b7ae926e0e8fc4e7a4a8dcac97`
- targeted v112 tests: 1 passed
- input v79 checkpoint SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- zero-adapter checkpoint SHA-256: `bdcfcc9c834ea97b57d14566cbf75bfbff9634a3a3023aebc6d03aafce159ce6`
- candidate actor SHA-256: `084e91614a968b8778dc0930b08cb6315e37426ab40075721d986d82d0a076e3`
- candidate checkpoint SHA-256: `5d57dc1443704f706fcd95e5e52731f5a5883b5f245085c8637b11d083422aa9`
- remote candidate: `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/state_conditioned_outcome_v112_829420f_4x32_s201353400/candidate.pt`

The rejected binary is not committed. Its exact path and hashes are recorded in
[`checkpoint_index.json`](checkpoint_index.json); complete training and screen
evidence is included in this directory.

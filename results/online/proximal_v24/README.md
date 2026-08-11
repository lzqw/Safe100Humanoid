# v24 Contact Completion — Final Result

## Outcome / 结论

The independent v24 pure low-friction experiment is complete. The fixed
round-8 actor did **not** improve target task success:

```text
target success: 72.266% -> 71.484%  (-0.781 pp)
target fall:    27.734% -> 28.516%  (+0.781 pp)
repairs / regressions: 95 / 99
```

The target-success requirement (`>= +3 pp`) failed. The target-fall bound
(`<= +1 pp`) and D0-success bound (`>= -5 pp`) passed. Confidence intervals
were report-only and did not change this point-estimate decision. No
outcome-directed rerun, early stop, rollback, or checkpoint selection was
performed.

独立的 v24 纯低摩擦实验已经完成。固定 round-8 actor 没有提高 target 成功率：
成功率下降 `0.781 pp`，fall 上升 `0.781 pp`，修复/破坏 episode 为 `95 / 99`。
因此任务效果 gate 失败。整个结果按原样保留，没有结果导向重跑或选择 checkpoint。

Together with the byte-frozen v23 lateral result, the registered completion
conclusion is case C:

> Moving-KL proximal PPO provides a bounded online update path, but did not
> improve task success in either representative deployment shift.

综合 byte-frozen v23 lateral 结果，结论属于情况 C：该方法能够进行有界在线更新，
但在两个代表性部署偏移中都没有提高任务成功率。按预注册解释，这条当前算法路线
应在此结束；后续若继续研究，应作为新方法检查 reward/success 一致性、critic
advantage quality、recovery takeover 或部分 Actor 更新，而不能改写 v23/v24。

## Prospective selection / 前瞻场景选择

Before any v24 episode, commit `d57760d` froze 16 candidates from light to
severe, all calibration/formal seeds, the base checkpoint, source hashes, and
the selection rule. Base-only calibration stopped at the first joint
qualifier; no more severe candidate was evaluated.

| Item | Frozen result |
| --- | ---: |
| candidate index / parameter seed | `9 / 51117` |
| foot friction | `0.326` |
| base successes | `360 / 512` (`70.3125%`) |
| falls | `152 / 512` |
| contact/slip falls | `134 / 152` |
| contact/slip purity | `88.1579%` |
| context file SHA-256 | `7e827d08ec3f7561a90ea838c9f008308a7d1a5e6d81ae381fa2eb0c3b4ef275` |
| parameter SHA-256 | `5e79a1cff0d8db75bcdf0072ed8592755c946666dce9affd0ac6c11a091a1864` |

Only `foot_friction` changed. Terrain, command and command dynamics, actuator,
sensor, gait phase, left/right response, navigation, and disturbance-pulse
settings remained nominal. Ordinary CBF intervention was not a failure label.

After calibration, commit `4a1b32f` froze the selected context and the sole
formal execution before any adapted outcome. Formal protocol SHA-256 is
`d52de9034523350dc5ebda1143d26ab02737d31cafccc764d3c9a092c4e6f39b`.

## Fixed eight-round update / 固定八轮更新

v24 reused the v23 implementation and hyperparameters without modification:
one original 405-D actor, one 838-D privileged critic, runtime CBF execution,
raw-action PPO storage, and a moving round-start reference `pi_k`.

| Round | Rollout success | Rollout fall | Moving forward KL | Actor epochs | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 49.46% | 50.54% | 0.003783 | 1 | updated |
| 2 | 38.95% | 61.05% | 0.001018 | 2 | updated |
| 3 | 47.73% | 52.27% | 0.000740 | 2 | updated |
| 4 | 43.56% | 56.44% | 0.000766 | 2 | updated |
| 5 | 40.00% | 60.00% | 0.000635 | 2 | updated |
| 6 | 35.56% | 64.44% | 0.000911 | 2 | updated |
| 7 | 51.69% | 48.31% | 0.000745 | 2 | updated |
| 8 | 46.81% | 53.19% | 0.000552 | 2 | updated |

All 8/8 rounds updated. Maximum moving KL was `0.003782826`, below the `0.01`
hard ceiling. There were zero hard or performance rollbacks, zero
action-routing error, and zero PPO policy-storage error. Round 1 correctly
stopped its second actor epoch after exceeding the `0.003` target. The final
actor is unconditionally round 8, not a best-so-far checkpoint.

## One fresh paired audit / 唯一一次 fresh paired 审计

The deterministic policy mean was evaluated with runtime CBF and identical
base/final initial conditions.

| Domain / metric | Base | Round 8 | Delta | Paired bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Target success (512) | 72.266% | 71.484% | **-0.781 pp** | [-6.055, +4.492] pp |
| Target fall (512) | 27.734% | 28.516% | +0.781 pp | [-4.883, +5.864] pp |
| Target return | 6.0572 | 5.7765 | -0.2807 | [-1.0173, +0.4555] |
| Target reached riser | 8.3594 | 8.2207 | -0.1387 | [-0.3477, +0.0645] |
| D0 success (256) | 88.281% | 89.453% | +1.172 pp | [-3.516, +5.859] pp |
| D0 fall (256) | 3.125% | 3.516% | +0.391 pp | [-2.344, +3.125] pp |
| D0 return | 8.5811 | 8.4009 | -0.1802 | [-0.8538, +0.5000] |

Target slip signal rose `+0.000698`, contact mismatch rose `+0.001239`, CBF
interventions/riser rose `+0.195433`, and mean correction norm rose
`+0.000684`. The last two paired intervals excluded zero. Target recovery
takeover count changed `365 -> 359`. On D0, repairs/regressions were `21 / 18`
and recovery takeover count changed `125 -> 128`.

The independent verifier reconstructed all 768 episode pairs, deltas, repair
and regression counts, gate decisions, actor hash chain, KL ceiling, action
routing, context binding, and immutable v23 anchors. All 38 checks passed.

## Evidence / 证据

- [formal protocol](protocol.json)
- [base-only calibration summary](calibration/calibration_summary.json)
- [selected pure-friction context](calibration/context.json)
- [eight-round training summary](training/training_summary.json)
- [per-round CSV](training/round_metrics.csv)
- [authoritative final result](final/final_test.json)
- [768 paired episode rows](final/paired_episode_metrics.csv)
- [independent verification](final/verification.json)
- [exactly three figure categories](figures/figure_manifest.json)
- [v23/v24 combined result](../proximal_completion/README.md)
- [package hashes](SHA256SUMS)

Large recovery checkpoints remain in the external artifact store. The fixed
round-8 checkpoint SHA-256 is
`beafb298f3a2cc955732bbe290867606142a601b6c82e01606fe784515482e50`.
The committed compact evidence is sufficient to audit the declared result;
the simulator was not rerun to construct the combined v23 row.

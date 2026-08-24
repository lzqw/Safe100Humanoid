# v29 CBF-Teacher Proximal Online Fine-Tuning

## 中文摘要

固定场景为 18 cm 等高台阶、sloped-clearance slope 0.8、recovery window 0.15 m、CBF alpha 10。训练使用原始 405D actor / 838D privileged critic、运行时 CBF 始终开启，并按固定预算尝试 8 轮 raw-action PPO + moving-KL + 成功门控 CBF teacher 更新。8/8 轮的 moving KL 均超过 0.01（0.241–0.249），因此全部按预注册规则硬回滚；round-8 Actor 的 MLP 与 base 逐字节相同，没有按性能选 checkpoint。

Teacher 数据量：共 54,087 个 eligible intervention transitions，权重和 52,920.91，训练 rollout 中 CBF intervention 共 62,966 次（由每轮精确记录的 intervention fraction × 65,536 transitions 重建）。最终结论为 **mixed_or_insufficient_effect**；18 cm CBF-on success 观测变化 -0.039，CBF-off success 观测变化 -0.016，D0 CBF-on success 观测变化 -0.008。由于 MLP 没有变化，这些差异是独立 GPU 仿真重复间的波动，不能归因于学习。后续决策：`stop_without_no_teacher_control`。

## English summary

The fixed target uses uniform 18 cm risers, clearance slope 0.8, a 0.15 m recovery window, and CBF alpha 10. The original 405D actor and 838D privileged critic attempted exactly eight raw-action PPO, moving-KL, and local-success CBF-teacher rounds while the runtime filter remained enabled. All 8/8 proposed updates exceeded the 0.01 hard KL ceiling (0.241–0.249) and were transactionally rolled back. The unconditional round-8 actor therefore has an MLP byte-identical to the base actor.

The final interpretation is **mixed_or_insufficient_effect** and the fixed decision is to stop without a no-teacher control. Because no MLP update survived, the small observed base/final differences are independent GPU-simulation repeat variability rather than learned-policy effects. CSV intervention counts were reconstructed exactly from each logged intervention fraction times 65,536 transitions. Paired bootstrap 95% intervals are reported in `final/final_test.json` for uncertainty only and were not used as a training or conclusion gate.

## 18 cm four-condition audit

| condition | success | fall | toe-riser kick | interventions/riser |
|---|---:|---:|---:|---:|
| pi0_off | 61.1% | 38.9% | 97.3% | 0.0000 |
| pi0_on | 73.0% | 27.0% | 99.6% | 6.2171 |
| pi8_on | 69.1% | 30.9% | 99.0% | 6.3268 |
| pi8_off | 59.6% | 40.4% | 96.9% | 0.0000 |

D0 (13 cm) CBF-on success: pi0 92.6%, pi8 91.8%. Final checkpoint SHA-256: `f6dcd79d71450c8069d7a2c75f281872660e37043818be124e39768571b7a735`.

Only the requested aggregate evidence and figures are tracked here. Checkpoints and per-step telemetry remain outside Git.

# v34 Outcome-Optimized Task-Metric Velocity CBF

## 中文摘要

v33 的伪加速度 HOCBF 与现有“关节位置目标 → 速度级安全投影”接口不匹配：直接替换和重新训练都降低成功率。因此 v34 保留有效的 sloped toe-clearance 速度约束，只把欧氏投影改成纯 Torch、单约束闭式 task-metric 投影。安全 nominal target 逐位原样通过；没有 CPU QP、HOCBF、MPC、控制器切换或新训练 gate。

自动开发严格按冻结协议完成：60 个 candidate 的 64 episodes/context 初筛、前 8 个的 256 episodes/context 精评、前 2 个各自在 F1/F2/F3 从共同 base 完成固定 8 轮 v31 A2（共 6 次训练），再按训练后 development 平均成功率选择统一参数。最终选择 **c000_current**：`barrier_slope=0.8, alpha=10, swing_knee_weight=1, swing_ankle_pitch_weight=1, swing_hip_pitch_weight=1, stance_leg_weight=1, hip_roll_yaw_weight=1, other_joint_weight=1, lambda_x=0, lambda_s=0, top_clearance=0.025, toe_margin=0.08`。

| 方法 | F1 success | F2 success | F3 success | Mean success |
|---|---:|---:|---:|---:|
| v31 A2 + current CBF | 73.05% | 71.88% | 66.41% | 70.44% |
| v31 A2 + optimized CBF | 74.61% | 71.68% | 69.34% | 71.88% |
| new A2 + optimized CBF | 71.68% | 69.92% | 65.04% | 68.88% |

最终 held-out 主结果：new A2 + selected CBF 为 **68.88%**，v31 A2 + current CBF 基线为 **70.44%**，差值 **-1.56 pp**。达到预设 +3 pp 开发目标：**否**。自动选择退化为 current control，因此表中的 v31 A2 + selected 条件不是新 CBF；其独立重复成功率为 71.88%，与基线相差 +1.43 pp，只能视为 GPU 仿真运行波动，不能解释为 CBF 改进。 无论正负，以上均为参数冻结后只运行一次的最终测试结果。

| D0 条件（F1/F2/F3 来源策略平均） | success |
|---|---:|
| v31 A2 + current CBF | 91.54% |
| v31 A2 + optimized CBF | 91.80% |
| new A2 + optimized CBF | 93.88% |
| v31 A2 CBF-off | 88.80% |
| new A2 CBF-off | 89.84% |

辅助 fall/return/riser/time、intervention、correction norm/jerk、toe impulse、unsafe overlap、support-foot slip、post-intervention fall、CBF-off 和逐 context paired effect 均在 `final/combined_results.json`。单次 8×256 smoke 通过：optimized/current 吞吐比 `1.071`；最终 optimized 条件平均 CBF action 计算时间 `2.8186 ms/step`。记录到的开发、训练、选择和最终审计总 wall time 约 `2.64 h`。

真机部署仍使用当前关节位置目标接口，投影在 GPU 端向量化执行。实际控制器应复核控制周期预算、关节速度/位置限制、传感延迟、接触检测和仿真到实机的动力学偏差。

## English summary

v33's pseudo-acceleration HOCBF was mismatched to the existing joint-position-target interface and reduced success both as a direct replacement and after retraining. v34 returns to the validated velocity-level sloped toe-clearance constraint and changes only its projection metric. The implementation is a pure-Torch, vectorized, closed-form single-constraint solve; an already-safe nominal target is returned exactly.

The frozen automated search evaluated 60 candidates, refined the top 8, trained the top 2 in all three contexts with the unchanged eight-round v31 A2 procedure, and selected one global parameter set solely by trained development success. Final identities were created only after that selection was committed, and the held-out audit ran once.

New A2 + selected CBF reaches **68.88%** mean held-out CBF-on success versus **70.44%** for v31 A2 + current CBF, a **-1.56 pp** change. The +3 pp development target was met: **no**. The outcome-only selector fell back to the current control. The v31 A2 + selected row is therefore an independent repeat of the same method (71.88%, +1.43 pp versus the baseline run), which is run-to-run GPU simulation variation rather than a CBF gain. See the aggregate JSON for all paired target/D0, CBF-off, safety, correction, and compute-time metrics.

Source boundary: `f5fa2283bd998bdb8c7d388abe8512d41a92865a`. `SHA256SUMS` binds every published aggregate and figure; raw episode traces and checkpoints remain in the external artifact directory.

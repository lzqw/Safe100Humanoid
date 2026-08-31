# v33 Task-Consistent Acceleration HOCBF-QP

## 中文摘要

v33 保留未修改的当前速度级 CBF（CBF0）、sloped toe-clearance 几何、405-D Actor、838-D privileged Critic 和 v31 A2 在线目标。CBF0 直接约束关节速度，容易通过改变摆动脚前向运动来满足半空间；新方法改为二阶约束

`ḧ + 2ζω ḣ + ω²h ≥ 0`，其中 `ζ=1`，并用 `D + λs I + λx JxᵀJx` 的纯 Torch Sherman–Morrison 闭式投影惩罚前向任务偏差和修正抖动。若 nominal margin 已安全，safe target 与 nominal target 精确相同。

18 个预定参数只在冻结的 v31 A1/A2 策略上开发；最终全局参数为 **ω=8, λx=0, λs=0**，没有 per-context 参数、额外搜索或训练 gate。单次 smoke 的 HOCBF/CBF0 吞吐比为 `1.041`，平均 HOCBF action 计算时间为 `2.2697 ms/step`。

| context | base + current | v31 A2 + current | v31 A2 + HOCBF | newly trained A2 + HOCBF |
|---|---:|---:|---:|---:|
| F1 | 73.24% | 72.66% | 65.04% | 66.80% |
| F2 | 73.63% | 70.31% | 62.50% | 56.84% |
| F3 | 63.67% | 61.72% | 52.73% | 49.61% |

三场景平均成功率：v31 A2 + current 为 **68.23%**，直接替换 HOCBF 后为 **60.09%**（-0.0814），在新 HOCBF 下从共同 base 完成固定 8 轮 A2 微调后为 **57.75%**（相对 v31 current -0.1048）。最高条件为 `base_current`；是否突破 72% 平台：**False**。

A1 current 的 on−off gap 为 -0.0840，新 HOCBF 的 on−off gap 为 -0.1504。逐 context 的 rescue/interference、post-intervention fall、correction norm/jerk、toe impulse、overlap、root roll/pitch、support slip、D0 和 paired bootstrap CI 均在 JSON/CSV 中。逐步 telemetry 与 checkpoint 只保存在外部 artifact，不提交 Git。

实机含义：该实现不引入 CPU QP solver，投影保持 GPU 向量化；控制周期还需为上表 smoke 实测计算时间、仿真到实机动力学偏差与传感延迟留出裕量。

## English summary

v33 leaves CBF0 and all v31/v32 artifacts unchanged. It replaces the velocity-level Euclidean projection only in a new mode with an acceleration HOCBF and a task-consistent weighted one-constraint QP. The pure-Torch Sherman–Morrison solve penalizes forward-foot acceleration changes and temporal correction changes, while an already-safe nominal target remains exactly unchanged.

The globally selected parameters are **ω=8, λx=0, λs=0**. Frozen-policy A1/A2 comparisons, three unconditional eight-round A2 runs from the common base, target/D0 paired audits, and at most 2,000 bootstrap samples follow the prospectively fixed protocol without outcome-dependent gates or checkpoint selection.

Mean target success changes from 68.23% for v31 A2 + current CBF to 60.09% after the direct HOCBF replacement and 57.75% after new-HOCBF refinement. The measured smoke throughput ratio is 1.041; inspect the aggregate files for safety, balance, D0, and compute-time tradeoffs before real-hardware use.

Source boundary: `ab23421d510f12cc0af9b7c73e4274735096d7c9`. `SHA256SUMS` binds every published aggregate and figure.

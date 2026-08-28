# v131 Deterministic-Aligned Low-Noise Paper PPO

v131 从 v129 selected actor 继续论文核心的 100% safety-filtered PPO：simulator
执行 filtered action，PPO storage 保存 nominal action，奖励保持 Eq. (27)
unit-balanced dual reward。唯一算法变量是将冻结 Gaussian training std 从 `0.05`
降到 `0.03`，探索方差变为 v129 的 `36%`，使训练采样动作更接近部署时的
deterministic mean。连续 Adam PPO、2 epochs × 4 minibatches、round-level KL
学习率控制均保持不变，不使用 moving-KL loss、teacher、DR 或 rollback。

## 结果

- 64 environments × 1,024 steps × 6 rounds = 393,216 transitions。
- RTX 4080 SUPER 训练用时 215.25 秒（3 分 35.25 秒）。
- 最佳 aligned filter-on rollout：round 5 使用 round-4 checkpoint，
  `101/135 = 74.81%`。
- 所选 checkpoint 的唯一 deterministic filter-off screen：
  **44/64 = 68.75%**，跌倒 20/64，平均到达 riser 7.7969。

| Rollout | Checkpoint | Filter on | LR used → next | Moving KL | Selected |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0 | 95/136 (69.85%) | 2.50e-6 → 1.44e-6 | 0.001799 | yes |
| 2 | 1 | 92/136 (67.65%) | 1.44e-6 → 1.54e-6 | 0.000530 | no |
| 3 | 2 | 95/137 (69.34%) | 1.54e-6 → 1.67e-6 | 0.000506 | no |
| 4 | 3 | 97/135 (71.85%) | 1.67e-6 → 1.72e-6 | 0.000568 | yes |
| 5 | 4 | **101/135 (74.81%)** | 1.72e-6 → 1.79e-6 | 0.000556 | **yes** |
| 6 | 5 | 94/137 (68.61%) | 1.79e-6 → 2.03e-6 | 0.000463 | no |

screen 未达到预声明的 `48/64`，因此没有运行独立 gate，也没有追加其他 seed、
checkpoint 或 filter-on 评估。v131 被拒绝；v129 的 `47/64 = 73.44%` 开发峰值和
v79 的 `139/193 = 72.02%` 正式最佳保持不变。结果说明降低探索噪声能保持稳定的
KL-controlled 训练，但不能单独解决 filtered rollout 到 deterministic filter-off
deployment 的内化差距。

## 溯源

- 代码提交：`d2df47a82208863716ab5c380198f1e049358b21`。
- 唯一针对性测试：
  `test_v131_reduces_exploration_variance_while_retaining_nonzero_sampling`，
  1 passed。
- base v129 checkpoint SHA-256：
  `71c5df2ed5bdfcca01eb6ee4302f0116bdca2482a2bd04486f5e239383a33ec8`。
- selected round-4 checkpoint SHA-256：
  `5fd6c343c0df848fcceaad9735509a0d6f06492de8b91e4cc265b3c4430ed4ce`。
- selected actor SHA-256：
  `86de165b8a3b2d881a878c8a1aec68c386b14dbe671438b1218ede923d7f64e2`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/deterministic_aligned_v131_d2df47a_64x1024x6_s201355200`。
- 全部训练与 screen JSON/CSV 已提交；未通过 gate 的模型二进制未提交到 Git。

实现：`src/tasks/stairs_cbf/paper_deterministic_aligned_v131.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

# v132 Scaled Paper-Style Continuation

v132 从 v129 selected actor 继续完全相同的论文式 100% safety-filtered PPO：
simulator 执行 filtered action，PPO storage 保存 nominal action，奖励使用 Eq. (27)
unit-balanced dual reward，Gaussian std 保持 `0.05`。连续 Adam PPO、
2 epochs × 4 minibatches、round-level KL 学习率控制均不变，不加入 auxiliary loss、
teacher、DR 或 rollback。唯一变化是使用 128 environments 训练 8 轮，将 v129 的
transition count 精确翻倍。

## 训练结果

- 128 environments × 1,024 steps × 8 rounds = 1,048,576 transitions。
- RTX 4080 SUPER 训练用时 287.82 秒（4 分 47.82 秒）。
- 最佳 aligned filter-on rollout：round 8 使用 round-7 checkpoint，
  `198/269 = 73.61%`。它与 round 4 的成功率相同，按更高 mean-riser tie-break 选中。

| Rollout | Checkpoint | Filter on | LR used → next | Moving KL | Selected |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0 | 189/270 (70.00%) | 2.50e-6 → 2.41e-6 | 0.000645 | yes |
| 2 | 1 | 194/264 (73.48%) | 2.41e-6 → 2.83e-6 | 0.000435 | yes |
| 3 | 2 | 193/267 (72.28%) | 2.83e-6 → 3.51e-6 | 0.000390 | no |
| 4 | 3 | 198/269 (73.61%) | 3.51e-6 → 4.07e-6 | 0.000447 | yes |
| 5 | 4 | 189/273 (69.23%) | 4.07e-6 → 4.57e-6 | 0.000476 | no |
| 6 | 5 | 198/271 (73.06%) | 4.57e-6 → 4.50e-6 | 0.000621 | no |
| 7 | 6 | 192/267 (71.91%) | 4.50e-6 → 4.27e-6 | 0.000664 | no |
| 8 | 7 | **198/269 (73.61%)** | 4.27e-6 → 5.00e-6 | 0.000432 | **yes** |

## 部署结果

- 唯一 development screen（seed `201355380`）：
  **50/64 = 78.125%**，首次超过预声明的 `48/64` gate-trigger 门槛。
- 因此运行且只运行一个独立 gate（seed `201355390`）：
  **46/64 = 71.875%**，低于正式 `48/64` 接受门槛两个 episode。
- 两个 evaluation seeds 描述性合计为 `96/128 = 75.00%`；由于 gate 是在 screen
  通过后条件触发，正式判定仍以独立 gate 为准，不能据 pooled 数值宣称通过。

v132 是目前最接近目标的候选，也是第一个 deterministic filter-off screen 超过 75%
的论文式训练版本；但独立 gate 未通过，因此不替换正式最佳 v79。没有运行第三个 seed、
其他 checkpoint 或 filter-on 补充评估。证据表明扩大论文式 full-filter on-policy
样本规模是迄今第一个改善部署表现的方向，但仍需提高跨 seed 稳定性。

## 溯源

- 代码提交：`194917e8b8b02713a083bcb61bdbc3509fc48696`。
- 唯一针对性测试：
  `test_v132_doubles_v129_transition_scale_without_changing_rollout_length`，
  1 passed in 18.16s。
- base v129 checkpoint SHA-256：
  `71c5df2ed5bdfcca01eb6ee4302f0116bdca2482a2bd04486f5e239383a33ec8`。
- selected round-7 checkpoint SHA-256：
  `a7fdd4d07dc79f1f001b09ff3638bcc5de000c3804f369dcf13ed61ebd18bde3`。
- selected actor SHA-256：
  `ae031e720b6c8e18fd855da10bbd0e3bc80160357ef93c3055c7cfefef126093`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/scaled_continuation_v132_194917e_128x1024x8_s201355300`。
- 全部训练、screen 和独立 gate JSON/CSV 已提交；gate 未通过，所以模型二进制未提交。

实现：`src/tasks/stairs_cbf/paper_scaled_continuation_v132.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

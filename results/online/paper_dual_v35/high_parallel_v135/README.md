# v135 Higher-Parallel Paper PPO

v135 区分“更多 sequential PPO updates”和“更大的同步 on-policy batch”。它从固定 v129
selected checkpoint 重新开始，保持与 v132 相同的 8 轮、std `0.05`、100% filtered
execution、Eq. (27) unit-balanced reward、2 epochs × 4 minibatches Adam PPO 和
round-level KL 控制，只把 environments 从 128 提到 192。每轮同步 batch 为 196,608
transitions，总计 1,572,864，是 v129 的 3 倍，但 sequential update count 不变。

## 结果

- RTX 4080 SUPER 训练用时 287.28 秒（4 分 47.28 秒）。
- 最佳 aligned filter-on rollout：round 6 使用 round-5 checkpoint，
  `281/390 = 72.05%`。
- 所选 checkpoint 的唯一 deterministic filter-off screen：
  **46/64 = 71.875%**，跌倒 18/64，平均到达 riser 8.0313。

| Rollout | Checkpoint | Filter on | LR used → next | Moving KL | Selected |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0 | 279/409 (68.22%) | 2.50e-6 → 2.40e-6 | 0.000649 | yes |
| 2 | 1 | 273/404 (67.57%) | 2.40e-6 → 2.65e-6 | 0.000493 | no |
| 3 | 2 | 280/410 (68.29%) | 2.65e-6 → 3.31e-6 | 0.000369 | yes |
| 4 | 3 | 285/405 (70.37%) | 3.31e-6 → 3.98e-6 | 0.000415 | yes |
| 5 | 4 | 290/405 (71.60%) | 3.98e-6 → 4.34e-6 | 0.000505 | yes |
| 6 | 5 | **281/390 (72.05%)** | 4.34e-6 → 4.78e-6 | 0.000495 | **yes** |
| 7 | 6 | 285/403 (70.72%) | 4.78e-6 → 5.00e-6 | 0.000462 | no |
| 8 | 7 | 293/412 (71.12%) | 5.00e-6 → 5.00e-6 | 0.000442 | no |

screen 差两个 episode 才达到 `48/64`，因此没有运行独立 gate，也没有追加 checkpoint
或 seed。192-env batch 的训练率比 v132 更平稳，但 filtered 峰值与 deterministic
deployment 都更低；所以 paper-style synchronous parallelism 从 64→128 的增益不能继续
单调外推到 192。v132 保持最强候选，v79 保持正式最佳。

## 溯源

- 代码提交：`c0ca7e70cddb433ed4b4beef974aebf165353e89`。
- 唯一针对性测试：
  `test_v135_triples_synchronous_v129_scale_without_more_updates`，
  1 passed in 17.85s。
- base v129 checkpoint SHA-256：
  `71c5df2ed5bdfcca01eb6ee4302f0116bdca2482a2bd04486f5e239383a33ec8`。
- selected round-5 checkpoint SHA-256：
  `dbbbf0d05800572316c43693f605609dcb20e74e66f779bb861b04c298df33a5`。
- selected actor SHA-256：
  `861dabeac8f62c67602d54be2ea7d52580ba33e2b153b73f3458b1f988bf41b9`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/high_parallel_v135_c0ca7e7_192x1024x8_s201355600`。
- 全部训练与 screen JSON/CSV 已提交；未通过 screen，所以模型二进制未提交。

实现：`src/tasks/stairs_cbf/paper_high_parallel_v135.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

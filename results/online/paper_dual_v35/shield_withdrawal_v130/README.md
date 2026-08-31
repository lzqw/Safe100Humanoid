# v130 Safety-Filter Withdrawal Consolidation

v130 从 v129 的 round-7 actor 开始，用 5 轮把训练执行的 safety-filter fraction 按
`1.0→0.75→0.5→0.25→0` 线性撤除，同时保留 counterfactual Eq. (27) dual reward、
连续 Adam PPO 和 v129 的 round-level KL 学习率控制。训练不做 rollback；只有包含至少
64 个 filter-off episode 的既有 training rollout 才能参与 checkpoint 选择。

## 结果

- 128 environments × 1,024 steps × 5 rounds = 655,360 transitions。
- RTX 4080 SUPER 训练用时 153.54 秒。
- 最佳 aligned filter-off rollout：round 3 使用 round-2 checkpoint，
  `90/124 = 72.58%`。
- 最终 0% filter rollout：`183/266 = 68.80%`。
- 所选 checkpoint 的唯一 deterministic filter-off screen：
  **32/64 = 50.00%**，跌倒 32/64，平均到达 riser 7.75。

| Round | Filter fraction | Filter on | Filter off | KL | Selection |
|---:|---:|---:|---:|---:|:---:|
| 1 | 1.00 | 197/265 (74.34%) | 0/0 | 0.001686 | ineligible |
| 2 | 0.75 | 142/206 (68.93%) | 44/64 (68.75%) | 0.000428 | selected then replaced |
| 3 | 0.50 | 98/138 (71.01%) | **90/124 (72.58%)** | 0.000390 | **selected** |
| 4 | 0.25 | 46/67 (68.66%) | 129/195 (66.15%) | 0.000433 | rejected |
| 5 | 0.00 | 0/0 | 183/266 (68.80%) | 0.000424 | rejected |

screen 未达到预声明的 48/64，因此没有运行独立 gate 或追加其他 checkpoint/seed。
v130 被拒绝，v129 的 47/64 开发 screen 和 v79 正式最佳均保持不变。结果说明直接撤除
shield 即使在 stochastic training rollout 中维持约 69%–73%，仍会破坏 deterministic
mean policy；后续不应继续相同的 filter-fraction 退火。

## 溯源

- 代码提交：`92196aa39609f881deb52a3a04542eda9656b43c`。
- 唯一针对性测试：
  `test_v130_selects_only_sufficient_filter_off_training_rollouts`，1 passed。
- base v129 checkpoint SHA-256：
  `71c5df2ed5bdfcca01eb6ee4302f0116bdca2482a2bd04486f5e239383a33ec8`。
- selected round-2 checkpoint SHA-256：
  `7a199623270e5961b04c76b5955644cbb5e13ef66be3a8030f5338040494044a`。
- selected actor SHA-256：
  `4b377dbb3249ecdcfe65c15e11c446d6a0cbc0f0795cc00293c0cc556eab638d`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/shield_withdrawal_v130_92196aa_128x1024x5_s201355100`。
- 全部训练与 screen JSON/CSV 已提交；未通过 gate 的模型二进制未提交到 Git。

实现：`src/tasks/stairs_cbf/paper_shield_withdrawal_v130.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

# v133 Second Scaled Paper-Style Stage

v133 从 v132 selected actor 继续第二个完全相同的论文式 full-filter stage：128
environments × 1,024 steps × 8 rounds，Gaussian std `0.05`，连续 2 epochs × 4
minibatches Adam PPO，round-level KL 控制。reward、filter、objective、optimizer、
teacher/DR/rollback 设置均不变；目的是检验 v132 的正向部署证据能否随同一 objective
继续扩大样本而稳定增强。

## 结果

- 本阶段 1,048,576 transitions；v132+v133 scaled stages 累计 2,097,152。
- RTX 4080 SUPER 训练用时 265.20 秒（4 分 25.20 秒）。
- 最佳 aligned filter-on rollout：round 3 使用 round-2 checkpoint，
  `197/263 = 74.90%`。
- 所选 checkpoint 的唯一 deterministic filter-off screen：
  **41/64 = 64.06%**，跌倒 23/64，平均到达 riser 7.9844。

| Rollout | Checkpoint | Filter on | LR used → next | Moving KL | Selected |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0 | 194/274 (70.80%) | 2.50e-6 → 2.54e-6 | 0.000582 | yes |
| 2 | 1 | 187/264 (70.83%) | 2.54e-6 → 2.72e-6 | 0.000523 | yes |
| 3 | 2 | **197/263 (74.90%)** | 2.72e-6 → 2.99e-6 | 0.000495 | **yes** |
| 4 | 3 | 186/267 (69.66%) | 2.99e-6 → 3.18e-6 | 0.000533 | no |
| 5 | 4 | 190/268 (70.90%) | 3.18e-6 → 3.67e-6 | 0.000448 | no |
| 6 | 5 | 198/272 (72.79%) | 3.67e-6 → 4.01e-6 | 0.000505 | no |
| 7 | 6 | 192/272 (70.59%) | 4.01e-6 → 4.75e-6 | 0.000426 | no |
| 8 | 7 | 198/271 (73.06%) | 4.75e-6 → 4.73e-6 | 0.000606 | no |

screen 未达到预声明的 `48/64`，因此没有运行独立 gate，也没有追加其他 checkpoint、
seed 或 filter-on 评估。第二个相同 million-transition stage 虽保持稳定 KL，并达到
74.90% filtered rollout，但 deterministic deployment 从 v132 的 50/64 降到 41/64。
这表明 v132 是当前 objective 下的局部部署峰值，不能通过继续相同训练稳定增强；v132
仍是最强候选，v79 仍是正式最佳。

## 溯源

- 代码提交：`ae093e7a65737e8bef00e163d86b9075a7091c50`。
- 唯一针对性测试：
  `test_v133_extends_the_same_scaled_objective_to_a_second_stage`，
  1 passed in 16.37s。
- base v132 checkpoint SHA-256：
  `a7fdd4d07dc79f1f001b09ff3638bcc5de000c3804f369dcf13ed61ebd18bde3`。
- selected round-2 checkpoint SHA-256：
  `1de8a78711b8abf1297279c33ba2438a55eb37a1939fdf45c81e1e340357902f`。
- selected actor SHA-256：
  `0af82956c1553ac2382109cd8e8b9cdb97fc13f26b86140cc7a8a907d298fb9c`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/scaled_stage_two_v133_ae093e7_128x1024x8_s201355400`。
- 全部训练与 screen JSON/CSV 已提交；未通过 screen，所以模型二进制未提交。

实现：`src/tasks/stairs_cbf/paper_scaled_stage_two_v133.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

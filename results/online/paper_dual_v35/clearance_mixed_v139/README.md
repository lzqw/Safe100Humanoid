# v139 Deployment-Majority Mixed Clearance Continuation

v139 从 v138 selected actor 继续训练，保留论文式 next-riser foot-clearance
reference 的 `8 cm` margin、Eq. (27) unit-balanced dual reward、标准 Adam PPO、
`std=0.05` 和 round-level KL 学习率控制。唯一算法变化是把训练执行分布固定为
`25% filter-on / 75% filter-off`，分别归一化两个组的 advantage，并按与部署一致的
filter-off 子组选择 checkpoint。

## 正式结果

- 唯一 deterministic filter-off development screen（seed `201356080`）：
  **50/64 = 78.125%**，通过预声明的 `48/64` gate-trigger 门槛。
- 因此运行且只运行一个独立 gate（seed `201356090`）：
  **49/64 = 76.5625%**，通过正式 `48/64` 接受门槛。
- 两个 seed 描述性合计为 **99/128 = 77.34375%**；正式接受依据仍是独立 gate
  自身达到 `49/64`，不是 pooled 结果。
- v139 是首个通过既定独立 gate 的 F2 18.4 cm、deterministic mean、runtime-filter-off
  候选，正式替换此前的 v79 最佳记录。没有运行第三个 seed、其他 checkpoint 或
  filter-on 补充评测。

## 训练结果

- 128 environments × 1,024 steps × 4 rounds = 524,288 transitions。
- RTX 4080 SUPER 训练用时 120.62 秒。
- 最佳 deployment-aligned filter-off rollout：round 2 使用 round-1 checkpoint，
  `132/195 = 67.69%`；后两轮没有超过它。

| Rollout | Checkpoint | Filter off | Filter on | LR used → next | Moving KL | Selected |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0 | 135/201 (67.16%) | 42/66 (63.64%) | 2.50e-6 → 2.29e-6 | 0.000714 | yes |
| 2 | 1 | **132/195 (67.69%)** | 47/68 (69.12%) | 2.29e-6 → 2.86e-6 | 0.000381 | **yes** |
| 3 | 2 | 123/202 (60.89%) | 48/69 (69.57%) | 2.86e-6 → 3.38e-6 | 0.000431 | no |
| 4 | 3 | 131/200 (65.50%) | 47/68 (69.12%) | 3.38e-6 → 4.23e-6 | 0.000383 | no |

## 部署明细

| Evaluation | Seed | Success | Falls | Mean reached riser | Unsafe overlap / riser |
|---|---:|---:|---:|---:|---:|
| Development screen | 201356080 | **50/64 (78.125%)** | 14/64 | 8.1406 | 0.8138 |
| Independent gate | 201356090 | **49/64 (76.5625%)** | 15/64 | 8.0313 | 0.9903 |

两次评测都使用固定 F2 18.4 cm 楼梯、原始 405-D actor、deterministic policy mean、
无 domain randomization、runtime filter off。screen 与 gate 的初态签名不同，gate 使用
未参与 screen 或训练的独立 seed。

## 溯源与文件

- 代码提交：`fdc2a63a484651c4e0fd73cb0227fdba935b1058`。
- 唯一针对性测试：
  `test_v139_uses_deployment_majority_mixture_and_keeps_v138_clearance`，
  1 passed in 15.21s；未运行 full suite。
- base v138 checkpoint SHA-256：
  `7a3899c515d5afd93f79f4db251feab4cd59f003e7150711e506ef5850604c63`。
- selected round-1 checkpoint SHA-256：
  `323f1e00b58d379b8746c0191a44272f2e1df134139050417c56e733cc484728`。
- selected actor SHA-256：
  `4a4926d9227c31fb239ceead6c39bed61304d1f2c7e3a47aea510e060cee2acd`。
- deterministic actor SHA-256：
  `601a60e066ca1fb805e2f8cd27d4bfe44d79256b93040d23495e5ea8699d8386`。
- 通过 gate 的模型二进制：
  [`checkpoints/selected_round_01.pt`](checkpoints/selected_round_01.pt)。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/clearance_mixed_v139_fdc2a63_128x1024x4_s201356000`。
- `training/`、`screen_seed201356080/` 与 `gate_seed201356090/` 包含完整 JSON/CSV
  原始结果；`decision_summary.json` 和 `checkpoint_index.json` 给出正式判定与模型身份。

实现：`src/tasks/stairs_cbf/paper_clearance_mixed_v139.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

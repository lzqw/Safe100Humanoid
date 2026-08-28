# v137 Swing-Foot CBF Toe-Off Preactivation

v137 针对 v132 的一个具体时序缺口：旧 CBF 只有在接触传感器已经判定脚离地后才选择
swing foot，因此不会约束真正决定抬脚的 toe-off 指令。根据论文的 humanoid stair CBF
仍以 body-frame swing-foot position 和下一台阶 hyperplane 为 reduced state，本版本不增加
推测性的 base-balance CBF，只在双脚仍接触且 gait clock 已进入某只脚的 swing phase 时，
提前选择该 scheduled swing foot；若存在实际离地脚，则物理接触状态始终优先。

论文来源：[CBF-RL: Safety Filtering Reinforcement Learning in Training](https://arxiv.org/html/2510.14959v6)。

## 训练结果

- 固定 v132 的 PPO、reward、模型、F2 18.4 cm 场景和 transition budget：
  128 environments × 1,024 steps × 8 rounds = 1,048,576 transitions。
- RTX 4080 SUPER 用时 226.44 秒（3 分 46.44 秒）。
- preactivation 确实生效：CBF reward active fraction 平均从 v132 的 27.01% 提高到
  34.95%，intervention fraction 平均从 9.78% 提高到 12.64%；8 轮 swing selection
  mismatch 均为 0。
- 但所有 actor 更新后的 aligned filter-on rollout 都低于初始 rollout。因此选择
  round-0（未更新）checkpoint：`193/268 = 72.01%`，mean reached riser 8.2127。

| Rollout | Checkpoint | Filter on | Mean riser | CBF active | Intervention | Selected |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0 | **193/268 (72.01%)** | 8.2127 | 35.39% | 12.62% | **yes** |
| 2 | 1 | 189/280 (67.50%) | 7.8429 | 34.67% | 12.47% | no |
| 3 | 2 | 195/287 (67.94%) | 7.7700 | 34.64% | 12.44% | no |
| 4 | 3 | 192/280 (68.57%) | 7.8321 | 34.63% | 12.47% | no |
| 5 | 4 | 188/280 (67.14%) | 7.7179 | 34.78% | 12.65% | no |
| 6 | 5 | 171/286 (59.79%) | 7.5629 | 36.18% | 13.41% | no |
| 7 | 6 | 186/278 (66.91%) | 7.7518 | 34.57% | 12.55% | no |
| 8 | 7 | 189/285 (66.32%) | 7.8246 | 34.79% | 12.51% | no |

## 部署结果与判定

唯一 deterministic filter-off development screen（seed `201355880`）：

- **43/64 = 67.1875%**；未达到预声明 `48/64` 门槛。
- 21/64 跌倒，mean reached riser 7.9688。
- 因此没有运行独立 gate、其他 seed、其他 checkpoint 或补充 filter-on 评测。

结论：scheduled toe-off preactivation 修复了 CBF 激活时序并显著提高了训练期间的 active
与 intervention 覆盖，但 PPO 更新方向全部退化，不能改善 deterministic filter-off 部署。
v137 拒绝；v132 仍是最强 development candidate，正式最佳仍为 v79 aligned filter-off
`139/193 = 72.02%`。

## 溯源

- 实现提交：`eb0c738cf8440dcb930230a94d18e8bed9581005`。
- 唯一针对性测试：
  `test_v137_preactivates_only_toe_off_and_keeps_airborne_foot_priority`，
  4080 上 `1 passed in 17.15s`；未运行完整测试套件。
- base v129 checkpoint SHA-256：
  `71c5df2ed5bdfcca01eb6ee4302f0116bdca2482a2bd04486f5e239383a33ec8`。
- selected round-0 checkpoint SHA-256：
  `cb356124df294f4cdccc839da59db23b130c1c359e15019aceeb3addfa9c3c29`。
- selected actor SHA-256：
  `c197a3470f346cb46d4e99b0de6339c168ccd0c6d84c7fc3fb45238b56545c79`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/swing_preactivation_v137_eb0c738_128x1024x8_s201355800`。
- 全部训练和 screen JSON/CSV 已提交；模型未通过 screen，所以二进制未提交。

实现入口：`src/tasks/stairs_cbf/paper_swing_preactivation_v137.py`、
`src/tasks/stairs_cbf/cbf_math.py` 与 `experiments/scripts/refine_paper_dual_v35.py`。

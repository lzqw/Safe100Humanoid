# v134 Uniform Actor-SWA Consolidation

v133 表明继续相同 PPO 会越过 v132 的 deployment 峰值。v134 因此不做新训练，而是
对 v132 round 1–7 的七个同轨迹 actor snapshot 做一次预声明 uniform SWA。只平均
373,644 个 `mlp.*` mean-policy 参数；normalizer 与 frozen `std=0.05` 必须跨 snapshot
逐位一致并保持 selected round-7 值。每个 snapshot 权重固定为 `1/7`，不搜索 averaging
window 或权重。输出 checkpoint 标记为 deployment-only，禁止误用于续训。

## 结果

- 七个 source checkpoint SHA-256 全部精确匹配预声明值。
- non-MLP actor state 完全相同并逐位保留。
- SWA 到 selected round-7 actor 的 MLP L2 distance：`0.00449766`。
- 唯一 deterministic filter-off screen（seed `201355580`）：
  **40/64 = 62.50%**，跌倒 24/64，平均到达 riser 7.7344。

screen 未达到 `48/64`，因此没有运行独立 gate，也没有尝试其他 snapshot window、权重或
seed。SWA 明显低于 v132 的 50/64 screen 和 46/64 independent gate，说明 v132 的有效
deployment 行为不是 PPO 轨迹在参数空间的共同中心；继续做 post-hoc averaging 搜索没有
依据。v132 保持最强候选，v79 保持正式最佳。

## 溯源

- 代码提交：`49f20b16e42db4521662975043a15442b830033b`。
- 唯一针对性测试：
  `test_v134_uniformly_averages_only_actor_mlp_snapshots`，1 passed in 17.54s。
- SWA checkpoint SHA-256：
  `d1f2fd282d301069dce6aeda8926c04130a989fa183e54f23b70ea029dad3700`。
- SWA actor SHA-256：
  `6acd1fded2e95284eed9b503c96ecc4f8945adf2489dedfe844a75b3d76fe816`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/uniform_swa_v134_49f20b1`。
- 完整 source 路径/SHA、构建诊断、screen JSON/CSV 已提交；未通过 screen，模型二进制
  未提交到 Git。

实现：`src/tasks/stairs_cbf/paper_uniform_swa_v134.py` 与
`experiments/scripts/build_paper_swa_v134.py`。

# v128 早期起点、100% Safety-Filtered 连续 PPO

v128 补齐 v127 后确认的论文实现缺口：不再从成熟的 v79 actor 做局部修补，而是从
原始 nominal checkpoint 开始，在固定 F2（18.4 cm）上连续训练。每个 rollout 的
simulator action 都经过 safety filter，PPO storage 仍保存 nominal policy action，奖励
使用 Eq. (27) 风格的 foot-task proximity 与 unit-balanced velocity margin。actor 使用
标准 clipped PPO、Adam、2 epochs × 4 minibatches，不使用 teacher、moving-KL、事务
回滚或 DR。

训练期间的 aligned filter-on rollout 已经是 on-policy 数据，因此 v128 直接用成功率、
平均登阶进度和轮次依次选择 checkpoint；这不改变连续训练轨迹，也没有增加 rollout。
训练结束后只对所选 checkpoint 做一次 64-episode deterministic filter-off screen。

## 训练结果

- 64 environments × 1,024 steps × 10 rounds = 655,360 transitions。
- RTX 4080 SUPER 用时 345.77 秒（5 分 45.77 秒）。
- 起点 filtered rollout：87/128 = 67.97%。
- 最佳 aligned rollout：round 7 使用 round-6 checkpoint，92/131 = 70.23%，相对起点
  +2.26 pp。
- 最后一轮 aligned rollout：81/127 = 63.78%。
- forward KL 首轮为 0.002553，随后为 0.000550–0.000798；训练没有数值发散，但成功率
  仍在 61.19%–70.23% 间振荡。

| Rollout round | Checkpoint round | Filter on | Moving KL | Selected |
|---:|---:|---:|---:|:---:|
| 1 | 0 | 87/128 (67.97%) | 0.002553 | yes |
| 2 | 1 | 90/129 (69.77%) | 0.000798 | yes |
| 3 | 2 | 91/131 (69.47%) | 0.000795 | no |
| 4 | 3 | 83/133 (62.41%) | 0.000728 | no |
| 5 | 4 | 84/133 (63.16%) | 0.000576 | no |
| 6 | 5 | 90/129 (69.77%) | 0.000594 | no |
| 7 | 6 | **92/131 (70.23%)** | 0.000630 | **yes** |
| 8 | 7 | 83/133 (62.41%) | 0.000561 | no |
| 9 | 8 | 82/134 (61.19%) | 0.000550 | no |
| 10 | 9 | 81/127 (63.78%) | 0.000580 | no |

## 部署筛选与结论

唯一新 seed `201354980` 的 deterministic filter-off screen 为 **36/64 = 56.25%**，
跌倒率 43.75%，平均到达 riser 7.5469。它没有达到预声明的 48/64 gate，因此没有运行
独立 gate，也没有追加 filter-on 或其他 seed。

该结果表明，早期起点和连续标准 PPO 修正了此前“只在成熟策略附近短续训”的实验
缺口，但在当前 64-env、655k-transition 规模下，filtered rollout 的小幅改善没有内化
为 filter-off 部署能力。v128 被拒绝，正式最佳仍为 v79 的训练内对齐 139/193 =
72.02%，开发 screen 最佳仍为 47/64 = 73.44%。

## 溯源

- 代码提交：`5f66a7873429d38403b7c9f6f85d63f31848dda3`。
- 唯一针对性测试：
  `test_v128_selects_only_the_best_aligned_filtered_training_rollout`，1 passed。
- 原始 base checkpoint SHA-256：
  `cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8`。
- selected round-6 checkpoint SHA-256：
  `0075f6436666a3a73fca0048ad4a849637ee610c71c6e598393c9ee43a1b8e1d`。
- selected actor SHA-256：
  `4561c182c42776d37daf6393d1422723fe288e8b6d2f9687bc5a3f38f5480acf`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/early_continuous_v128_5f66a78_64x1024x10_s201354900`。
- `training/` 与 `screen_seed201354980/` 保存全部原始 JSON/CSV；checkpoint 路径和
  哈希见 `checkpoint_index.json`。模型二进制未重复提交到 Git。

实现：`src/tasks/stairs_cbf/paper_early_start_v128.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

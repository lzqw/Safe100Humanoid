# v129 KL-Controlled Continuous Paper PPO

v129 从当前最佳 v79 actor 继续论文核心的 100% safety-filtered PPO：simulator 执行
filtered action，PPO storage 保存 nominal action，奖励使用 Eq. (27) unit-balanced
foot-task dual reward。与此前 full-filter 实验不同，actor 使用连续 2 epochs × 4
minibatches Adam PPO，不做事务回滚，也不加入 moving-KL loss。

v128 首轮更新 KL 为 0.00255，后续成功率明显振荡。v129 因此只给下一轮学习率增加
一个预声明控制器：目标 forward KL 为 `6e-4`，每轮 LR 倍率限制在 `[0.5, 1.25]`，
总范围限制在 `[5e-7, 5e-6]`。控制器不修改已经完成的 actor update，也不根据结果
回滚策略。

## 训练结果

- 64 environments × 1,024 steps × 8 rounds = 524,288 transitions。
- RTX 4080 SUPER 用时 281.66 秒（4 分 41.66 秒）。
- 起点 filtered rollout：98/138 = 71.01%。
- 最佳 aligned rollout：round 8 使用 round-7 checkpoint，108/128 = **84.38%**，
  相对起点 +13.36 pp。
- 首轮 KL 0.002847 触发 LR 5e-6→2.5e-6；之后 KL 保持在
  0.000437–0.000632，围绕目标稳定。

| Rollout | Checkpoint | Filter on | LR used → next | Moving KL | Selected |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0 | 98/138 (71.01%) | 5.00e-6 → 2.50e-6 | 0.002847 | yes |
| 2 | 1 | 88/138 (63.77%) | 2.50e-6 → 2.84e-6 | 0.000465 | no |
| 3 | 2 | 94/147 (63.95%) | 2.84e-6 → 3.08e-6 | 0.000511 | no |
| 4 | 3 | 96/138 (69.57%) | 3.08e-6 → 3.60e-6 | 0.000437 | no |
| 5 | 4 | 95/135 (70.37%) | 3.60e-6 → 3.90e-6 | 0.000513 | no |
| 6 | 5 | 91/136 (66.91%) | 3.90e-6 → 3.80e-6 | 0.000632 | no |
| 7 | 6 | 89/136 (65.44%) | 3.80e-6 → 3.84e-6 | 0.000588 | no |
| 8 | 7 | **108/128 (84.38%)** | 3.84e-6 → 4.09e-6 | 0.000529 | **yes** |

## 部署筛选

所选 round-7 checkpoint 在唯一新 seed `201355080` 上的 deterministic filter-off
screen 为 **47/64 = 73.44%**，跌倒 17/64，平均到达 riser 8.2031。它只差一个成功
回合达到预声明的 `48/64` gate，因此按约定没有运行独立 gate，也没有追加其他
checkpoint、filter-on 或 seed。

v129 相比 v128 的 36/64 明显恢复，并追平已有最高开发 screen，但没有产生超过门槛的
新证据，不能替换正式最佳 v79。结果同时说明 round-level KL 控制确实解决了更新尺度
失稳，并让 full-filter training 达到 84.38%；剩余差距集中在 filtered success 向
filter-off deployment 的内化，而不是数值稳定性。

## 溯源

- 代码提交：`2e8da8b13acb6cfa14b76f04d2c1804b51850589`。
- 唯一针对性测试：
  `test_v129_controls_next_round_lr_without_rollback_or_actor_mutation`，1 passed。
- v79 base checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`。
- selected round-7 checkpoint SHA-256：
  `71c5df2ed5bdfcca01eb6ee4302f0116bdca2482a2bd04486f5e239383a33ec8`。
- selected actor SHA-256：
  `c197a3470f346cb46d4e99b0de6339c168ccd0c6d84c7fc3fb45238b56545c79`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/continuous_kl_v129_2e8da8b_64x1024x8_s201355000`。
- 所有训练与 screen JSON/CSV 已提交；失败 gate 的模型二进制未重复提交到 Git。

实现：`src/tasks/stairs_cbf/paper_continuous_kl_v129.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

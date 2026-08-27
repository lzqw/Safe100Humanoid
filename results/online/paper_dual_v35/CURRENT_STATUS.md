# Paper-dual 当前结果索引（2026-08-27）

当前目标是 F2 18.4 cm、部署时 filter-off 成功率至少 75%。截至本次记录，目标尚未达到，
当前最好的训练内对齐结果仍是 v79 的 **139/193（72.02%）**。

| 版本 | 关键结论 | GitHub 目录 |
|---|---|---|
| v79 | 从 61.11% 提升到 72.02%，但低于 75% 门槛；仍是当前最佳 | [`mixed_unit_balanced_v79/`](mixed_unit_balanced_v79/) |
| v80 | pooled base 为 68.04%，proposal 为 68.32%，仅 +0.28 pp，判定统计持平 | [`mixed_cont_v80/`](mixed_cont_v80/) |
| v81 | 256/192 environments 都在首次 actor full-batch update OOM；0 次 optimizer step，无候选模型 | [`pooled_scaling_v81_oom/`](pooled_scaling_v81_oom/) |

v79 最佳 checkpoint 保留在 4080：
`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/mixed_unit_balanced_v79_d65f0b6_s201352619/round_03.pt`，
SHA-256 为 `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`。
因为它没有通过 75% gate，模型二进制未提交到 Git；配置、逐轮结果、哈希和选择依据均已提交。

本次记录时没有 v81 训练进程仍在运行。下一步是把 actor full-batch update 改为 microbatch
梯度累积：分块反向传播，但只执行一次 SGD step，以保留 256-environment 的全批平均梯度，
同时把峰值显存降到接近 128-environment 的水平。预计代码调整 10--15 分钟，随后单次训练约
3--4 分钟。


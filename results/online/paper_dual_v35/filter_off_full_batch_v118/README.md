# v118 full-batch filter-off residual PPO

v118 保留 v117 的 filter-off episodic objective，但把 44 次 Adam minibatch 改成一次
覆盖 88,571 transition 的 direction-preserving SGD。四组探索为 169/256 成功。

投影后的 clipped surrogate 从约 0 提升到 `+0.001179`，KL 为 `0.004853`，离线 gate
通过，证明 full-batch SGD 修复了 v117 的负 surrogate。然而唯一新 seed 的 deterministic
filter-off screen 只有 **40/64 = 62.5%**，mean reached riser `7.922`，因此不运行
独立 gate、候选拒绝。

该结果把问题从“优化器方向扭曲”进一步定位到“pooled episode-outcome gradient 对训练
seeds 过拟合”。下一步预留一整个 rollout seed 做 surrogate validation，只有 train 与
validation 都改善才保留 proposal，否则事务回滚。

实现 commit `f2e3b11c28ae24151fdca2710ff0c551970e98be`；耗时 135.98 秒。

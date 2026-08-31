# v119 held-out transactional residual PPO

v119 将最后一个完整 64-env rollout seed 从训练中隔离。前三个 seed 做一次 full-batch
SGD；只有 train 与 held-out surrogate 都改善才保留 proposal，否则在 screen 前恢复
exact zero residual。

train clipped surrogate 从约 0 提升到 `+0.001675`，但 held-out seed 从约 0 下降到
`-0.001449`。事务 gate 因此拒绝 proposal；initial 与 final residual SHA 均为
`da250c...a17a`，screen 的 mean residual norm 精确为 0。

回滚后的原始 v79 screen（seed `201354080`）为 **46/64 = 71.875%**，mean reached
riser `8.406`。没有达到 75%，不运行独立 gate。

v119 证明二元成功/失败 residual gradient 在完整 held-out rollout seed 上会反向。
下一步保留该事务协议，但用终局 reached-riser 连续 credit 代替高方差二元标签。

实现 commit `2d49d9910ed37404a2d5add7c7181da9957ca879`；总耗时 95.64 秒。

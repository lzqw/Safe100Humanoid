# v120 held-out terminal-progress residual PPO

v120 保留 v119 的 full-batch SGD、完整 seed validation 和事务回滚，只把二元成功/失败
advantage 换成每个 seed 内标准化的 `terminal reached-riser + success bonus`。每个
episode 仍等权。

train progress surrogate 从约 0 提升到 `+0.000742`，但 held-out seed 为
`-0.001842`，因此 proposal 被拒绝并恢复 exact zero residual；initial/final SHA
均为 `8c09b4...e7ca`。

回滚后的 v79 screen（seed `201354180`）为 **46/64 = 71.875%**，mean reached
riser `8.000`，mean residual norm 为 0；未运行独立 gate。

连续终局进度没有解决跨 seed 梯度反向，说明 residual exploration 的 episode-level
score-function gradient 本身方差仍过高。后续不应继续更换同类标量 credit；需要学习
state-action critic/GAE 或扩大真正独立的 on-policy batch 后再更新。

实现 commit `c95455e0ba4a26cf856207556a1e647d18466606`；耗时 106.75 秒。

# v121 held-out pretrained-critic GAE residual PPO

v121 冻结 v79 actor，并将 v79 的 838-D critic 以 10 个零列精确扩展到 848-D。
在真实 filter-off deployment trajectory 上记录逐步 reward/value/done，用
`gamma=0.99`、`lambda=0.95` 计算完整首 episode GAE；每个 rollout seed 单独标准化，
每个 episode 等权。最后一个完整 seed 仍只用于事务接受，不参与更新。

4x64 rollout 共得到 165 success / 91 failure、105,214 个完整 transition，其中
88,846 个 persistent-geometry transition 进入 residual objective。一次 full-batch SGD
经 KL=0.005 投影后，train clipped surrogate 从 `-0.004977` 提升到 `-0.003434`，
但 held-out seed 从 `-0.030414` 降到 `-0.032271`，因此 proposal 被拒绝并恢复 exact
zero residual；initial/final residual SHA 均为 `5bcb1d...348f`。

回滚后的 v79 screen（seed `201354280`）为 **41/64 = 64.0625%**，mean reached
riser `7.921875`，mean residual norm 为 0。低于 75% 门槛，因此未运行独立 gate，
全局最佳仍为 v79 的 139/193 = 72.02%。

结果说明预训练 state-value baseline 能产生有限且数值稳定的 GAE，但没有消除 residual
action gradient 的跨 seed 反转。下一步需要同初始状态的 paired/antithetic residual
action advantage，直接消除 initial-state rollout 方差，而不是继续更换单轨迹 credit。

实现 commit `aac480de6e61a2cb732bb8885de0a45921548a47`；运行耗时 109.30 秒。

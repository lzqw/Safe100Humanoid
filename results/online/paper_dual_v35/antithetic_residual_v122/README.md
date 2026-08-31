# v122 paired-antithetic filter-off residual PPO

v122 对每个 seed 和每个初始状态运行 `+noise/-noise` 两条镜像 filter-off 轨迹。探索
噪声使用独立同 seed generator，环境随机数不会破坏镜像；256 对初始状态的签名全部
一致。advantage 使用两分支的 `reached-riser + success bonus` 差值，在每个 seed 内
标准化，并让每个 pair 的正负 branch episode 等权。

512 条轨迹中，正分支 171 success、负分支 170 success；64 对差值为正、68 对为负、
124 对为零。一次 full-batch SGD 经 KL=0.005 投影后，train clipped surrogate 从约 0
提高到 `+0.001510`。held-out unclipped surrogate 同样为正 `+0.001113`，但 clipped
surrogate 为 `-0.000133`，所以按预声明规则拒绝并恢复 exact zero residual；initial/final
SHA 均为 `5484f4...503d`。

回滚后的 v79 screen（seed `201354380`）为 **42/64 = 65.625%**，mean reached riser
`7.625`，mean residual norm 为 0。低于 75%，未运行独立 gate；全局最佳仍为 v79 的
139/193 = 72.02%。

paired antithetic 已把 held-out 的一阶方向从负变正，当前失败来自较大 ratio 被 PPO
clip 后反转，而不是原始方向再次跨 seed 反转。下一步保留完全相同 objective，只在
预声明几何 scale 上选择 train/held-out clipped surrogate 都改善的最大安全步长。

实现 commit `37508577f7017628bc0a030fb041f25188d5cf9f`；运行耗时 182.74 秒。

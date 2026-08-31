# v123 jointly calibrated paired-antithetic residual PPO

v123 保留 v122 的 matched `+noise/-noise` objective，只增加预声明的 8 档几何 scale：
`1, 1/2, ..., 1/128`。选择规则是在 train 和完整 held-out seed 上，clipped surrogate
都至少改善 `1e-6` 的最大 scale；若不存在则精确回滚。

256 对初始状态的签名全部一致。正分支 162 success、负分支 168 success；66 对差值
为正、73 对为负、117 对为零。train-KL proposal 在 held-out 上为负；`scale=0.5`
的 held-out gain 仅 `6.52e-7`，低于预声明 margin；`scale=0.25` 首次同时满足条件，
train/held-out clipped surrogate 分别为 `+0.000658` 和 `+0.00000618`，KL 均约
`0.000309`。因此 offline gate 通过，候选没有回滚。

但唯一全新 deterministic filter-off screen（seed `201354480`）只有
**38/64 = 59.375%**，mean reached riser `7.828125`，mean residual norm
`0.000401`。低于 75%，未运行独立 gate，也没有替换 v79；全局最佳仍为
139/193 = 72.02%。

v123 证明 paired surrogate 可以跨 seed 且在 clipping 后共同改善，但这仍不足以预测
deterministic mean-policy deployment success。下一步应直接优化 deterministic residual
参数的 paired return（parameter-space antithetic/direct search），不再继续收缩同一个
stochastic action-space PPO proposal。

实现 commit `750281da5116f40b990824744cb3d68b798c5dbc`；运行耗时 195.92 秒。

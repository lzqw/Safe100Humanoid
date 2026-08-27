# v126 tolerant local deterministic parameter ES

v126 保留 v125 的 deterministic parameter-return objective、`sigma=0.005`、held-out
cosine `>=+0.05` 和 candidate norm `0.001`，只把三个 train gradient 的平均 pairwise
cosine 容忍线从 0 固定为 `-0.05`。

256 对轨迹全部具有 matched initial-state signature 和镜像参数方向。正负分支均为
172/256 success；56 对差值为正、69 对为负、131 对为零，探索 residual norm 约
`0.0104`。train pairwise cosine 为 `-0.02283`，通过宽松线；但 train mean 与完整
held-out gradient cosine 为 `-0.09503`，远低于 +0.05，因此仍精确回滚。

回滚后的 v79 screen（seed `201354780`）达到 **47/64 = 73.4375%**，mean reached
riser `8.0625`，但 mean residual norm 为 0，所以不能归因于 v126 候选。仍差一个
episode 才到 75%，未运行独立 gate；全局正式最佳继续为 v79 的 139/193 = 72.02%。

v124-v126 在宽、局部和宽松 train-consensus 三种预声明设置下都出现 held-out 参数梯度
反向，继续调 sigma/cosine threshold 没有证据支持。后续停止 parameter ES 路线，回到
paper-dual actor，并研究跨 seed 稳定的 state-conditioned value/occupancy correction。

实现 commit `ad66bfb4d42fc89eaea3b62a92bf0b294afb658c`；运行耗时 138.70 秒。

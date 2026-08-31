# v124 deterministic parameter-space antithetic ES

v124 不再优化 stochastic action log-prob。它使用零初始化的 10-D persistent-geometry
线性 residual（132 个参数），为每个环境固定一组参数方向，并从同一初始状态运行
deterministic `theta+sigma*u` / `theta-sigma*u` 完整轨迹。终局进度差直接形成
parameter-space ES gradient；前三个 seed 求平均，最后一个 seed 只检查方向 cosine。

256 对轨迹全部具有 matched initial-state signature 和逐位相同的镜像参数方向。正负
分支分别为 173/256 与 168/256 success；68 对差值为正、64 对为负、124 对为零。
`sigma=0.02` 产生约 `0.0393` 的平均探索 residual norm。三个 train gradient 的平均
pairwise cosine 为 `+0.05090`，但 held-out gradient cosine 为 `-0.08477`，低于
预声明 `+0.05` 门槛，因此没有生成非零候选并精确回滚。

回滚后的 v79 screen（seed `201354580`）为 **45/64 = 70.3125%**，mean reached
riser `7.984375`，mean residual norm 为 0。低于 75%，未运行独立 gate；全局最佳仍
为 v79 的 139/193 = 72.02%。

直接 deterministic parameter return 仍在当前较宽扰动半径上跨 seed 反向。下一步只
缩小 parameter sigma 到局部范围，使探索 residual norm 从约 0.039 降到约 0.010；
其余目标、gate 和候选幅度保持不变。

实现 commit `30d596fdc51ea204124938768bba1326de6c0c98`；运行耗时 162.23 秒。

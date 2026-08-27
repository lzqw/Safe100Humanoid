# v113 paired treatment gate + bounded residual

v113 针对 v111/v112 的正负 outcome correction 抵消问题，把训练拆成两个独立头：

1. gate 从 filter-off 的 415-D 可部署状态特征预测同初态下开启 CBF 是否会把失败变成成功；
2. residual 只拟合 matched-rescue episode 上的同状态 CBF correction，部署时仅在
   gate 超过 `0.5` 且 persistent geometry 激活时使用。

v79 主 actor 完全冻结，解析 CBF 在部署 screen 中完全关闭。

## 训练与结果

4×64 个 paired initial states 中，filter-off 为 177/256，filter-on 为 195/256；
包含 54 个 rescued 和 36 个 harmed episode。gate 使用 18,176 条状态、residual 使用
3,561 条 rescue trace，另有 80,118 条 filter-off success trust 状态。

| 指标 | 训练前 | 训练后 |
|---|---:|---:|
| gate balanced BCE | 0.707533 | 0.686290 |
| gate balanced accuracy | 49.85% | 54.46% |
| residual weighted target distance | 0.057575 | 0.033427 |

residual 本身能拟合 teacher，但 treatment gate 的正类均值 `0.5017`、负类均值
`0.4904`，分离度不足。唯一 64-episode filter-off screen（seed `201353580`）为
**42/64 = 65.625%**，mean reached riser 为 `7.703`，因此没有触发独立 gate。

## 结论

显式拆分 gate 和 residual 消除了全局符号抵消，但当前局部可部署状态对终局 CBF
treatment effect 的可预测性太弱，未改善任务成功率。候选拒绝，正式最佳继续保留
v79 的 139/193 = 72.02%。下一步不重复本配置，应改用因果时序上下文，或回到仅差
一个 episode 的 v92 方向做预先限定的幅度筛选。

训练总耗时 202.82 秒；实现 commit 为 `85556ef348bef0b5d312f292137caed3a062589e`。
模型二进制未上传，精确路径与 SHA-256 见 `checkpoint_index.json`。

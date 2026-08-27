# v114 held-out v92 adapter scale calibration

v114 回到历史上唯一达到 47/64 的 v92 Eq. (23) observable CBF adapter，不重新训练
方向，只在一次 256-environment filter-off screen 中比较预先声明的四档 actor-delta
幅度。每档 64 episode；只有最佳档达到 75% 才允许运行一次独立 gate。

| v92 delta scale | Filter-off screen | Mean reached riser |
|---:|---:|---:|
| 0× | **46/64 = 71.875%** | 7.859 |
| 0.5× | 39/64 = 60.938% | 7.609 |
| 1× | 41/64 = 64.063% | 8.000 |
| 1.5× | 45/64 = 70.313% | **8.406** |

成功率选择最终回到 `0×`，即原始 v79。所有 adapter scale 都低于 75%，所以独立
gate 自动跳过。v92 方向在本次 seed 上虽然 1.5× 能让平均进度更远，但不能把更多
episode 转化为登顶，幅度校准路线拒绝。

整轮耗时 39.59 秒，source commit `1b4cb8e87276ceeb6cc356be5b992cc623514672`。
候选模型未上传；精确路径和 SHA-256 见 `checkpoint_index.json`。

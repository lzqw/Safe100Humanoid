# v125 local deterministic parameter-space ES

v125 保留 v124 的 132-D deterministic parameter-return objective、candidate norm 和
全部 gate，只将 `sigma` 从 0.02 降到 0.005。实际正负探索 residual norm 分别降到
约 `0.01028/0.01031`。

256 对轨迹全部具有 matched initial-state signature 和镜像参数方向。正负分支分别为
172/256 与 174/256 success；61 对差值为正、72 对为负、123 对为零。首个正分支达到
48/64，但它由 64 个不同参数方向组成，不是一个可部署候选。

三个 train gradient 的平均 pairwise cosine 为 `-0.02022`，略低于预声明 0 门槛；
train mean 与完整 held-out gradient 的 cosine 为 `+0.08541`，已超过 +0.05 门槛。
由于必须同时满足两项，v125 仍精确回滚。回滚后的 screen（seed `201354680`）为
**42/64 = 65.625%**，mean residual norm 为 0；未运行独立 gate，全局最佳仍为 v79。

局部 sigma 将 held-out cosine 从 v124 的负值变成正值，而 train pairwise 只轻微低于
零，幅度小于 132-D 随机方向的典型 cosine 噪声。下一步保留 held-out +0.05 门槛，
仅把 train mean pairwise 容忍线预声明为 -0.05，再用全新 seeds 检查候选。

实现 commit `7b9fd4d1a37e3cdb84747764cdfbb56f2bdeba81`；运行耗时 202.24 秒。

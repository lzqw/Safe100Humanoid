# v72 单步 full-batch SGD 的 paper-style CBF-dual PPO

v72 保持 v71 的论文式训练数据流不变：25% 环境执行 CBF filter、75% 环境执行
nominal action，on/off advantage 分组归一化，teacher 为 A0。唯一核心变化是把每轮
8 个 Adam actor minibatch step 改为 **1 个 full-batch SGD step**；全局梯度裁剪
只缩放完整梯度，不改变它的方向。

训练从同一个 405-D F2 filter-off 最佳 base 出发，使用 `raw_moderate` 双奖励、
斜坡 x-z CBF、4×1024 steps、128 environments、actor LR `1e-4`、moving KL
`0.5`，未开启 DR。RTX 4080 SUPER 总训练时间为 **136.79 秒**。

## 训练结果

第 N 轮 rollout 在更新前执行，严格评估第 N−1 轮 checkpoint：

| Rollout | 实际 checkpoint | Filter off | Filter on | 总成功率 |
|---:|---:|---:|---:|---:|
| 1 | base | 133/196 (67.86%) | 47/68 (69.12%) | 180/264 (68.18%) |
| 2 | round 1 | 131/195 (67.18%) | **47/64 (73.44%)** | 178/259 (68.73%) |
| 3 | round 2 | **136/190 (71.58%)** | 48/69 (69.57%) | **184/259 (71.04%)** |
| 4 | round 3 | 120/197 (60.91%) | 47/70 (67.14%) | 167/267 (62.55%) |

最佳更新后 checkpoint 是 round 2：同一次运行内 filter-off 相对 base 提高
**3.72 pp**，达到 **71.58%**。它仍低于预设的 75% 训练门槛；round 3 又下降到
60.91%，说明连续固定步长更新仍可能越过有效区域。因此没有运行额外 128-episode
deterministic gate，也没有上传已拒绝的 checkpoint，当前正式最佳模型不变。

## 优化诊断

四轮均准确执行一次 full-batch SGD actor update。更新后的 moving forward KL 为
`2.63e-5`、`4.21e-5`、`3.77e-5`、`4.25e-5`，action mean shift 仅
`3.47e-4`–`4.28e-4`；相比 v71 的多 minibatch Adam，已经达到预期的保守
trust-region 尺度。pre-clip gradient norm 为 `3.76`–`4.65`，虽仍触发全局裁剪，
但本方法只施加一次标量缩放，完整 batch 的梯度方向得到保留。

该结果说明优化几何修正有效，但固定的连续更新缺少基于下一次对齐 rollout 的接受/
回滚机制。下一步将从 round-2 候选附近采用更小的自适应步长，并只接受实际提升的
checkpoint，避免 round-3 式退化。

## 文件与溯源

- source commit：`de28d67847c82182b8d03d0c3f6c3110c61202ce`
- base checkpoint SHA-256：
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`
- best aligned round-2 checkpoint SHA-256（已拒绝，未上传模型）：
  `3285223174b01c97009db54361042c4c3d2d87054ca2156c84769f6d13ceccbc`
- best aligned round-2 actor SHA-256：
  `672dfdb3af5f19c870313183b461e90cdd4198bff59d41060ce1553eb24fea32`
- final、未对齐 checkpoint SHA-256：
  `3c7f90fdbda494263fac7892cb901ce040fddafa40f583b5cccb9fdd9cb48868`
- 4080 原始输出目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/full_batch_sgd_v72_de28d67_s201352619`
- `training/training_summary.json`：完整配置、checkpoint/actor 哈希及总耗时。
- `training/round_metrics.{json,csv}`：四轮完整指标和优化审计。
- `decision_summary.json`：机器可读门槛结论。

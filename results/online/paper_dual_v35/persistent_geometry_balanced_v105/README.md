# v105 persistent geometry 相对梯度平衡

v104 证明完整 paper-dual PPO 几乎不使用新增 10-D persistent geometry。v105 保留
415-D 完整 actor PPO 和全局 gradient clipping，只在 backward hook 中将 geometry
首层梯度块提升到 legacy 405-D 首层块的相同 L2 norm，倍率上限 32×。倍率每轮由
实测梯度自动确定，不进行 scale 搜索。

训练复用 v79 最佳 checkpoint、v104 seed、F2 18.4 cm、25% filter-on / 75%
filter-off mixed execution、Eq. (27) unit-balanced reward、128 environments、1024
steps、6 轮 full-batch SGD，actor LR `1e-4`。共处理 786,432 transitions，在 RTX
4080 SUPER 上耗时 **160.21 秒**。

## 对齐结果

| Rollout | checkpoint | Filter off | Filter on | geometry scale | moving KL |
|---:|---:|---:|---:|---:|---:|
| 1 | round 0 | 140/198 (70.71%) | 48/73 (65.75%) | 12.16× | 3.88e-5 |
| 2 | round 1 | 124/202 (61.39%) | 50/65 (76.92%) | 10.05× | 3.56e-5 |
| 3 | round 2 | **138/194 (71.13%)** | 50/65 (76.92%) | 9.45× | 3.24e-5 |
| 4 | round 3 | 133/194 (68.56%) | 48/67 (71.64%) | 9.26× | 3.29e-5 |
| 5 | round 4 | 124/199 (62.31%) | 52/69 (75.36%) | 8.78× | 4.20e-5 |
| 6 | round 5 | 132/196 (67.35%) | 45/63 (71.43%) | 10.68× | 4.21e-5 |

每轮缩放后的 geometry/legacy gradient norm ratio 均为 1.0，数值和 action routing
稳定。最佳更新后 actor 比本次基线高 **0.43 percentage points**，但仍低于 v79
历史 72.02% 和 75% gate。没有运行独立 gate，v105 被拒绝，全局选择仍是 v79。

## 诊断与结论

对最佳 `round_02.pt` 与 `round_00.pt` 比较：

- geometry 首层权重最大绝对值：`7.3310e-7`
- legacy 首层权重最大变化：`3.1665e-7`
- 更深 MLP 层最大变化：`6.2175e-6`

仅两轮后 geometry 权重已达到 v104 五轮结果的 3.28 倍，并超过 legacy 首层变化，
所以 v105 确实解决了“新增输入梯度饥饿”。成功率却仅提高 0.43 pp，说明剩余问题
不是 geometry 输入是否得到更新，而是普通 per-transition PPO advantage 没有提供与
最终登顶 outcome 稳定一致的几何方向。下一步需要 episode-outcome-aligned auxiliary
objective，而不是继续放大同一梯度。

## 文件与溯源

- source commit：`8e3c0a23fb6a6033115e64a86f42d76e954daf4b`
- base checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- best aligned checkpoint：`round_02.pt`
- best aligned checkpoint SHA-256：
  `e1debe45edd0cc8db6f6ff1278972b17ddb66ee16474fabae84cba85d18257b6`
- best aligned actor SHA-256：
  `1c74f980f54737cd7a138ed88ecc4f8320a3c1e333612b5c84c7d816a22cf522`
- final unaligned checkpoint SHA-256：
  `fc5365a8913ffa4a24e0f558c8172f703f410f4be3add26fb348ffc34112de48`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/persistent_geometry_balanced_v105_8e3c0a2_s201353360`

未通过 gate 的模型二进制未上传 GitHub；完整训练 JSON/CSV、checkpoint 哈希和选择
依据均保存在本目录。

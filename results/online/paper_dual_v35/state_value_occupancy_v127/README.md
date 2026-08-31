# v127 state-value occupancy-corrected paper-dual PPO

v127 针对论文训练期全过滤、部署期无过滤的状态分布差异：在同一 50/50 mixed
rollout 中，用 actor 的 128-D 因果 hidden state 与 pretrained critic value 组成
129-D 特征，按完整环境轨迹做 2-fold cross-fit，估计
`d_filter_off(s) / d_filter_on(s)`。actor 只使用加权后的 filtered GAE；filter-off
transition 只用于占用估计，actor advantage 精确为 0；critic 仍使用全部 transition。

训练从 v79 开始，128 environments、4×1024 steps、每轮两个 gradient chunks、一次
globally-clipped SGD step，并用下一轮 filter-off rollout 做事务接受/回滚。RTX 4080
SUPER 训练耗时 **97.07 秒**，唯一针对性测试在 4080 项目环境中通过。

## 训练结果

| Rollout | 实际 actor | Filter off | Filter on | Occupancy BA | 决策 |
|---:|---|---:|---:|---:|---|
| 1 | v79 base | 92/129 = 71.32% | 99/132 = 75.00% | 51.53% | baseline |
| 2 | proposal 1 | 80/128 = 62.50% | 95/137 = 69.34% | 52.01% | reject / rollback |
| 3 | v79 retry | 86/133 = 64.66% | 91/139 = 65.47% | 52.14% | 产生 proposal 2 |
| 4 | proposal 2 | 79/134 = 58.96% | 92/135 = 68.15% | 51.22% | reject / rollback |

同一 v79 actor 的两个 filter-off rollout 合并为 **178/262 = 67.94%**。两个更新后
proposal 都明显低于该 anchor，事务门完整回滚；最终 selected actor SHA 与 v79 完全
相同。occupancy classifier 只有 51.2%–52.1% balanced accuracy，说明当前 129-D
在线状态/value 表示几乎无法稳定区分两种 occupancy；有信号的轮次有效样本率为
63.1%–71.4%，但重加权后的 filtered PPO 仍未改善部署成功率。

## 唯一 filter-off screen

新 seed `201354880` 的 deterministic 64-episode screen 为
**41/64 = 64.0625%**，fall 23/64，mean reached riser `7.484375`。它低于固定的
48/64 触发线，因此没有运行独立 gate，也没有额外 rollout。v127 被拒绝，正式最佳
仍为 v79 的 **139/193 = 72.02%**。

## 溯源

- source commit：`49cd14eda958af58784b4be6df4c50160141e32b`
- input v79 checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected actor SHA-256：
  `b0a717cef34d128e4175226b86780e5210cc5287558d78cb5a44e095e37fb600`
- selected wrapper checkpoint SHA-256：
  `97c8e08b5924de7f620e56fd0ceb4259220f42a7c0824acf799a6832a5352f15`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/state_value_occupancy_v127_49cd14e_128x1024_s201354800`

checkpoint 二进制未提交；路径和哈希见
[`checkpoint_index.json`](checkpoint_index.json)。完整逐轮 JSON/CSV 位于
[`training/`](training/)，screen 的 summary 和 64 条 episode 记录位于
[`screen_seed201354880/`](screen_seed201354880/)。


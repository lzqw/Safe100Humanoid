# v82 256-environment microbatch full-gradient 训练

v82 修复了 v81 的 full-batch actor OOM，同时保留 v72 的核心约束：一个完整 rollout 只产生
一次 globally-clipped SGD actor update。实现把 256×1024 transitions 随机排列后分成两个
等大的 chunks；每个 chunk 的 population-mean loss 乘 `1/2` 后反向传播，两个梯度相加，
最后只裁剪一次并执行一次 optimizer step。

训练从 v79 best checkpoint 出发，沿用 Eq. (27) unit-balanced reward、25/75 mixed
execution、group-balanced advantages、pooled transactional acceptance 和约 `5e-5` actor LR。
RTX 4080 SUPER 上的 4-round 训练总耗时 **137.25 秒**，未再 OOM。

| Rollout | Filter off | Filter on | Actor LR | 决策 |
|---:|---:|---:|---:|---|
| 1（base） | 251/397（63.22%） | 94/139（67.63%） | 5.000e-5 | anchor |
| 2 | 249/390（63.85%） | 97/135（71.85%） | 5.000e-5 | accepted |
| 3 | 274/404（67.82%） | 104/140（74.29%） | 5.000e-5 | accepted |
| 4 | 265/382（69.37%） | 96/131（73.28%） | 4.865e-5 | accepted / selected |

同一运行内 filter-off 从 63.22% 提升到 69.37%，即 **+6.15 pp**；每轮 telemetry 均确认
`actor_minibatches_completed=2` 且 `actor_optimizer_updates_completed=1`。不过最终结果仍低于
75% gate，也低于 v79 的 72.02%，因此没有追加部署验证，当前全局最佳不变。

## 文件与溯源

- source commit：`7faad93cc4133f8a250d78d742593855dac06028`
- input checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected aligned checkpoint：`round_03.pt`
- selected checkpoint SHA-256：
  `b483e95ab345b7936d7b0f8b360e758a01ccd96d77d1ac08f3a915714001c16a`
- selected actor SHA-256：
  `e13d5290a1b550b65d523797db4b7f94c27ec35815965054c17cf1939ea86165`
- final unaligned `round_04.pt` SHA-256：
  `04a366d806c426bde1904d4243307b0426b2ef3caa8fcd086fc402817dabba1e`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/pooled_256_v82_7faad93_s201352625`

`training/` 保存未经改写的 `round_metrics.json`、`round_metrics.csv` 和
`training_summary.json`；`decision_summary.json` 保存紧凑结论。未通过 75% gate，因此没有
把约 10 MB 的 checkpoint 二进制提交到 Git，checkpoint 的准确路径和哈希已保留。


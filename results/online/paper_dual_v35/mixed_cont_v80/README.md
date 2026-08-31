# v80 v79-best 的半步长事务 continuation

v80 从 v79 best aligned round-3 checkpoint 出发，保持 Eq. (27) unit-balanced
reward 和 25/75 mixed execution，将 actor LR 从 `1e-4` 降到 `5e-5`，并在新 seed
`201352624` 上启用 filter-off rollout 事务回滚。128 environments、4×1024 steps
在 RTX 4080 SUPER 上耗时 **134.48 秒**。

| Rollout | Actor | Filter off | Filter on | 原运行决策 |
|---:|---|---:|---:|---|
| 1 | v79 best base | 127/199 (63.82%) | 45/66 (68.18%) | anchor |
| 2 | proposal 1 | 118/195 (60.51%) | 48/70 (68.57%) | rejected / rollback |
| 3 | 同一 base 重试 | 137/189 (72.49%) | 52/67 (77.61%) | 产生 proposal 2 |
| 4 | proposal 2 | 138/202 (68.32%) | 46/64 (71.88%) | accepted by old anchor |

同一个 base actor 的两次 filter-off rollout 相差 **8.67 pp**。v80 运行时门槛只
保留第一次 63.82% anchor，所以把 68.32% proposal 判为改善；把两次相同 actor
的证据正确合并后，base 为 **264/388（68.04%）**，proposal 只高 **0.28 pp**，
应解释为统计持平，而不是可靠提升。所有结果都低于 75%，未追加部署 gate 或上传
checkpoint，当前最佳不变。

代码随后在 commit `720d41c` 修复：同一 accepted actor 的重复 rollout 会累计
success/episode counts，后续 proposal 与 pooled anchor 比较，并在结果中记录累计
样本数。该修复避免一次异常低或高的 rollout 长期偏置事务门槛。

## 文件与溯源

- training source commit：`d65f0b6a18d02b085338cf54af14079b8cd05851`
- pooled-anchor fix commit：`720d41ce7ff8d57a0384e955ef4929d21b675431`
- input checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- input actor SHA-256：
  `b0a717cef34d128e4175226b86780e5210cc5287558d78cb5a44e095e37fb600`
- v80 aligned proposal-2 checkpoint SHA-256（未上传模型）：
  `770bdbe6e28ef4362e12d36313828179b35bfb9a8cf557f3cf4768eb8809b983`
- proposal-2 actor SHA-256：
  `d68c754b23cdbf6babb64329cd662906c8a952fa2bbe1353b124adc4ab3252d0`
- final unaligned checkpoint SHA-256：
  `c2a91db17d1640855270a2bb5e3d5e7b99d10d90027779b9e6a8fdbbbbd14b72`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/mixed_cont_v80_d65f0b6_s201352624`
- `training/` 保存原始 JSON/CSV；`decision_summary.json` 保存 pooled 事后结论。

下一步不再依赖约 200 个 episode 的单轮 raw rate；使用 256 environments 增加每轮
完成 episode 数，并采用修复后的 pooled anchor，在相近训练时间内降低选择噪声。

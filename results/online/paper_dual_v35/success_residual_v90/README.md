# v90 成功干预 bounded deterministic residual（本机筛选）

v90 保留 v89 的“完整成功 episode 且 deterministic mean 实际触发 CBF”路由，但不再
克隆完整 safe mean，而是使用论文 A2 风格目标
`reference_mean + 0.25 * deterministic_CBF_correction`。由于 4080 被外部 CARLA
supervisor 持续占用，本轮在本机 GTX 1660 SUPER 上用 32 environments 做方向筛选；它
不是正式 256-env 证据或 deployment gate。

| Rollout | Filter on | 决策 |
|---:|---:|---|
| 1（base） | 53/72（73.61%） | anchor |
| 2 | 47/67（70.15%） | rejected / rollback |
| 3（base 重试） | 52/70（74.29%） | pooled evidence |
| 4 | 43/71（60.56%） | rejected / rollback |

同一 base actor pooled 为 **105/142（73.94%）**；两个 v90 proposal 均明显回落，故不
占用 4080 做正式放大，也不运行 filter-off gate。全局最佳仍是 v79 的 72.02% 正式结果。

## 关键发现

bounded target 按预期把 policy-to-target distance 从 v89 的约 0.52 降到
**0.1276–0.1333**。但 32-env actor gradient norm 仍达 **7.43–9.42** 且每轮都被裁剪，
而 teacher loss 只有约 0.0006；这说明 noisy PPO actor gradient 仍支配更新。下一步禁用
PPO/entropy actor 梯度，只保留成功干预 bounded residual 与 moving reference KL。

## 溯源

- implementation/source commit：`2c5e920`
- input checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected actor SHA-256：`b0a717cef34d128e4175226b86780e5210cc5287558d78cb5a44e095e37fb600`
- hardware：本机 NVIDIA GeForce GTX 1660 SUPER，32 environments
- 原始目录：`/home/lzqw/PycharmProject/safe100/HUMANOID/artifacts/paper_dual_v35/success_residual_v90_2c5e920_s201352636_n32_local1660`

完整原始 JSON/CSV 位于 `training/`；未过筛选，checkpoint 二进制不提交 Git。

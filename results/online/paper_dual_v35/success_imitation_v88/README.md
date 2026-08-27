# v88 成功轨迹安全动作模仿

v88 在论文式 100% safety-filtered rollout 上保留 nominal PPO action、task+CBF reward 和
moving KL，同时只对完整到顶 episode 加入 population-scaled Smooth-L1 模仿损失。训练使用
256 environments、1024 steps、2 个 gradient chunks、每轮一次 SGD step；4 轮总耗时
**111.81 秒**，没有 OOM。

| Rollout | Filter on | 决策 |
|---:|---:|---|
| 1（base） | 384/537（71.51%） | anchor |
| 2 | 377/547（68.92%） | rejected / rollback |
| 3（base 重试） | 370/541（68.39%） | pooled evidence |
| 4 | 379/544（69.67%） | rejected / rollback |

两个 v88 proposal 都未优于 base；同一 base actor 的 pooled evidence 为
**754/1078（69.94%）**。训练 filter-on 未达到 75%，因此没有追加 filter-off deployment
gate，当前全局最佳仍是 v79 的 72.02%。

## 关键发现

成功 episode 内有约 60.5%–62.7% 的全部 rollout transitions，但其中仅约 8.5%–8.8%
实际发生 CBF intervention。v88 对其余未干预 transition 也使用 stochastic sampled action
作为目标，因此 policy-to-target 距离约为 **0.203**，主要反映探索噪声而不是安全修正。
下一版应只学习“成功且实际干预”的 transition，并改用同状态 deterministic safe mean，
避免克隆 stochastic action noise。

## 溯源

- implementation/source commit：`5920ef7`
- input checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected checkpoint SHA-256：`6e3c07140aaa7b548d1477bb9f0b2d463e624f55c8fee2a62314a1fced3a69dd`
- selected actor SHA-256：`b0a717cef34d128e4175226b86780e5210cc5287558d78cb5a44e095e37fb600`
- 4080 原始目录：`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/success_imitation_v88_5920ef7_s201352634`

完整原始 JSON/CSV 位于 `training/`。checkpoint 未过 gate，因此模型二进制不提交 Git。

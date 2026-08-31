# v86 论文式 100% filtered execution

v86 使用 100% safety-filtered training execution，同时 PPO storage 继续保存 nominal policy
action 和 task+CBF reward。候选沿用 v85 的 `margin=0.1 / proximity=2.0`，256 environments、
2 gradient chunks、4 rounds 耗时 **150.32 秒**。

| Rollout | Filter on | 决策 |
|---:|---:|---|
| 1（base） | 390/543（71.82%） | anchor |
| 2 | 380/546（69.60%） | rejected / rollback |
| 3（base 重试） | 399/551（72.41%） | pooled evidence |
| 4 | 398/552（72.10%） | rejected / rollback |

两个 proposal 都未优于 base；同 base pooled 为 **789/1094（72.12%）**。训练 filter-on 未达
75%，因此没有追加 filter-off deployment gate。下一步扩大有效 PPO batch，而不是继续对同一
noisy gradient 做更细的 reward 调参。

## 溯源

- source commit：`3666721`
- input checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected base `round_00.pt` SHA-256：`586d3b4eec3ea168cae63b6eb2508da23df9b00d80168b4801c2c3ebd681105e`
- selected actor SHA-256：`b0a717cef34d128e4175226b86780e5210cc5287558d78cb5a44e095e37fb600`
- final unaligned checkpoint SHA-256：`23fb290aba84b8920ddc16546b7de7b8f092104b2acd1013c9e50b4df0841965`
- 4080 原始目录：`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/full_filter_proximity_v86_3666721_s201352629`

完整原始 JSON/CSV 位于 `training/`。


# v85 增强 Eq. (23) action proximity

v85 从 v79 best 出发，恢复 `margin_weight=0.1`，只将 reduced-order action-proximity
`intervention_weight` 从 1 提到 2。实际 CBF penalty 约为 nominal reward 的 22%，与 v84
总强度相近，但学习信号集中在 nominal action 接近 filtered action。256 environments、
2 gradient chunks、4 rounds 耗时 **148.21 秒**。

| Rollout | Filter off | Filter on | 决策 |
|---:|---:|---:|---|
| 1 | 259/396（65.40%） | 92/137（67.15%） | anchor |
| 2 | 271/390（69.49%） | 92/141（65.25%） | accepted |
| 3 | 261/401（65.09%） | 102/135（75.56%） | rejected / rollback |
| 4（同 accepted actor） | 255/387（65.89%） | 98/142（69.01%） | pooled evidence |

同 accepted actor 的 filter-off 合并结果为 **526/777（67.70%）**。增强 action proximity
没有稳定突破混合执行平台；下一步不继续调同一总 reward 幅值，而是恢复论文的 100%
safety-filtered training execution，并继续保留 nominal action 作为 PPO 样本。

## 溯源

- source commit：`10f09be`
- input checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected `round_01.pt` SHA-256：`9728485b07dddc1779ca03f96f366a9daa9bc0b0ce42c1c45d735b48d86cc678`
- selected actor SHA-256：`dd6f88d8e5bad71c1b232a9bbf18d29a6d9b02be0971c8dfecf8c55c4ccdc2a8`
- final unaligned checkpoint SHA-256：`2d9ab9977e3076056fb549112e931cc26da595a56fc5799cae24492839a295cd`
- 4080 原始目录：`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/proximity_v85_10f09be_s201352628`

完整原始 JSON/CSV 位于 `training/`。


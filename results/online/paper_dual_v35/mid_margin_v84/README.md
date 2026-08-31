# v84 中间 Eq. (27) margin 强度

v84 从 v79 best 出发，只将 sloped unit-balanced CBF margin weight 从 `0.1` 提到 `0.25`。
实际 manager-scaled CBF penalty 约为 nominal reward 的 23%，位于 v78 的约 14% 与 v76 的
73--82% 之间。256 environments、2 gradient chunks、4 rounds 在 RTX 4080 SUPER 上耗时
**150.28 秒**。

| Rollout | Filter off | Filter on | 决策 |
|---:|---:|---:|---|
| 1 | 260/403（64.52%） | 94/141（66.67%） | anchor |
| 2 | 261/393（66.41%） | 98/139（70.50%） | accepted |
| 3 | 267/395（67.59%） | 88/136（64.71%） | accepted / selected |
| 4 | 260/401（64.84%） | 88/135（65.19%） | rejected / rollback |

最佳对齐结果为 67.59%，低于 v79 的 72.02% 和 75% gate。提高状态 margin 没有带来更强
filter-off transfer，因此下一候选恢复 `margin_weight=0.1`，转而适度提高 Eq. (23) 的
action-proximity 权重。

## 溯源

- source commit：`b1756e8`
- input checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected `round_02.pt` SHA-256：`69b8463a07cd6a13ec50b1a5a7ba991022bf22310f33e65af6736bb94df44883`
- selected actor SHA-256：`9f037fe041ff89220b1118a9ad9b53f0c22a7cf2df40a49c4eabd5cea8d64f53`
- final unaligned checkpoint SHA-256：`c7764f66fd2e74fff3ba58172591500cb2f0757c2ea2462bf106ff8cd1833c71`
- 4080 原始目录：`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/mid_margin_v84_b1756e8_s201352627`

完整原始 JSON/CSV 位于 `training/`。


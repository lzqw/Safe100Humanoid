# v89 成功干预 deterministic safe-mean 模仿

v89 修复 v88 的 stochastic sampled-action 噪声：只选择完整到顶 episode 中 frozen
deterministic policy mean 实际触发 CBF 的 transition，并以同状态 deterministic safe
mean 为目标。训练继续使用 100% filtered execution、nominal PPO storage、256 environments、
1024 steps、2 个 gradient chunks 和每轮一次 SGD step；4 轮耗时 **113.21 秒**。

| Rollout | Filter on | 决策 |
|---:|---:|---|
| 1（base） | 390/546（71.43%） | anchor |
| 2 | 378/542（69.74%） | rejected / rollback |
| 3（base 重试） | 385/546（70.51%） | pooled evidence |
| 4 | 377/543（69.43%） | rejected / rollback |

两个 proposal 均回落；同一 base actor 的 pooled evidence 为
**775/1092（70.97%）**，未达到 75% filtered prerequisite，因此没有追加 filter-off
deployment gate，全局最佳仍是 v79 的 72.02%。

## 关键发现

v89 将 teacher eligibility 从 v88 的约 61%–63% 正确缩到全部 transitions 的
**5.26%–5.55%**，彻底去掉未干预 sampled-action noise。但直接克隆完整 deterministic
safe mean 的平均 correction norm 达到 **0.512–0.524**，目标过于激进；下一版应使用论文
A2 的 bounded residual，即只应用 25% deterministic correction。

## 溯源

- implementation/source commit：`fb31dab`
- input checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected checkpoint SHA-256：`d187b02ccfbaa666a88f2e2ddc9d0db5e2f9e67e742450a5b06b29e44f3abcfe`
- selected actor SHA-256：`b0a717cef34d128e4175226b86780e5210cc5287558d78cb5a44e095e37fb600`
- 4080 原始目录：`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/success_intervention_v89_fb31dab_s201352635`

完整原始 JSON/CSV 位于 `training/`。checkpoint 未过 gate，因此模型二进制不提交 Git。

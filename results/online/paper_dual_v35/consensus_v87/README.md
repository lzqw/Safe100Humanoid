# v87 四 rollout actor-delta 共识

v87 从同一个 v79 actor 生成四个独立 256-environment、1024-step、unit-balanced mixed
execution PPO proposal。每个成员都只执行一次 clipped SGD step；随后计算
`base + mean(proposal_i - base)`。总有效样本数为 **1,048,576 transitions**。

| Seed | 生成 rollout 的 base filter-off | Proposal actor SHA-256 前缀 |
|---:|---:|---|
| 201352625（v82 round 1） | 251/397（63.22%） | `dd910a37` |
| 201352630 | 265/390（67.95%） | `ac9d6266` |
| 201352631 | 259/389（66.58%） | `611e75f2` |
| 201352632 | 253/386（65.54%） | `9f3386cc` |

成员 delta 的平均 pairwise cosine 为 **0.0170**，范围 -0.1030 到 0.0881；共识 delta
范数仅为单成员平均范数的 **51.26%**。这说明单 rollout PPO 方向几乎正交，确实存在很强
梯度噪声。但共识 actor 在独立 seed `201352633`、F2 18.4 cm、filter-off、256 个固定初始
episode 上只有 **173/256（67.58%）**，未达到 75% gate，因此没有第二次确认。

## 溯源

- consensus implementation commits：`b6783f6`、`910be4d`
- base checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- consensus checkpoint SHA-256：`e1f538b5d528a7f656e59866ccc2ebf0ff403d4ed6660afdaa0a01e8952b71ac`
- consensus actor SHA-256：`27e7bb145ba4aea1091e9bc106ed147c14b7095db845e5978013da1c8d6df364`
- 4080 checkpoint：`/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/consensus_v87_910be4d/consensus.pt`

`manifest.json` 保存每个 proposal 的完整哈希、配置签名和 delta 几何；`members/` 保存三个
新增成员的原始训练结果（第四个成员是已上传的 v82 round 1）；`gate/` 保存逐 episode
filter-off 结果。checkpoint 未过 gate，因此二进制不提交 Git。


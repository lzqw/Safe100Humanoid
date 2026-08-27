# v91 成功干预 residual-only actor（本机筛选）

v91 在 v90 的论文 A2 风格 `25%` bounded deterministic residual 基础上，将 actor 的
PPO 与 entropy transition mask 置零。critic 仍用 task+CBF reward 学习；actor 只接收
“成功 episode 且 deterministic mean 实际触发 CBF”上的 bounded residual teacher，外加
moving-reference KL。

由于 4080 被外部 CARLA supervisor 占用，本轮在本机 GTX 1660 SUPER 上用 32
environments 做方向筛选；它不是正式 256-env 证据，也不是 deployment filter-off gate。

| Rollout | 对齐 actor | Filter on | 决策 |
|---:|---|---:|---|
| 1 | base | 46/66（69.70%） | initial anchor |
| 2 | LR `1e-3` proposal | 44/68（64.71%） | rejected / rollback |
| 3 | base 重试 | 49/68（72.06%） | base pooled evidence |
| 4 | LR `1.25e-4` proposal | 47/66（71.21%） | noninferior / selected |

同一 base actor 的 pooled evidence 为 **95/134（70.90%）**。最终选中的保守 proposal
相对该 pooled base 高约 **0.32 pp**，但样本只有 66 episodes，且结果仍低于 75% 目标，
因此只能判定为“值得放大验证的方向”，不能宣称超过 v79 或通过 gate。正式全局最佳仍是
v79 的 139/193（72.02%）filter-off 结果。

## 关键发现

- actor PPO transition count 在四轮均为 **0**，确认 reward-noise actor 梯度已隔离。
- actor gradient norm 降到 **0.1180–0.1276**，四轮均未发生 gradient clipping；v90
  对照为 7.43–9.42 且每轮裁剪。
- `1e-3` / `5e-4` 的较大更新仍会退化；`1.25e-4` 更新的 moving KL 为
  **4.84e-6**，action mean shift 为 **1.41e-4**，得到非劣 proposal。
- 下一次有足够 GPU 时应直接固定 `1.25e-4`，先做更大 rollout，再决定是否运行正式
  256-env filter-off gate，不重复较大的学习率。

## 溯源与 checkpoint 索引

- implementation/source commit：`21444c87a9a79f1a27e1091f6e7251b7131f23c4`
- targeted tests：30 passed
- input checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- selected rollout：round 4；对应 checkpoint：`round_03.pt`
- selected actor SHA-256：`e6a27287bb73ed454fc0ba278adb750763a65cbfc1d5e699832c28ade11da6cd`
- selected checkpoint SHA-256：`c9a276f694ad5ab7035ceaccbf65d6a42d1fce0c6292dbdf7e89e4ed5314d2a1`
- checkpoint 本机路径：`/home/lzqw/PycharmProject/safe100/HUMANOID/artifacts/paper_dual_v35/success_residual_only_v91_21444c8_s201352637_n32_local1660/round_03.pt`
- hardware：本机 NVIDIA GeForce GTX 1660 SUPER，32 environments
- seed：`201352637`
- wall time：`173.43 s`

完整原始 JSON/CSV 位于 [`training/`](training/)；由于本轮未过 gate，checkpoint 二进制
不提交 Git，哈希与可复现路径已记录。

# v91 成功干预 residual-only actor（本机筛选）

v91 在 v90 的论文 A2 风格 `25%` bounded deterministic residual 基础上，将 actor 的
PPO 与 entropy transition mask 置零。critic 仍用 task+CBF reward 学习；actor 只接收
“成功 episode 且 deterministic mean 实际触发 CBF”上的 bounded residual teacher，外加
moving-reference KL。

由于 4080 被外部 CARLA supervisor 占用，先在本机 GTX 1660 SUPER 上用 32
environments 做方向筛选，随后用 64 environments 放大；两者都不是正式 256-env
证据或 deployment filter-off gate。

| Rollout | 对齐 actor | Filter on | 决策 |
|---:|---|---:|---|
| 1 | base | 46/66（69.70%） | initial anchor |
| 2 | LR `1e-3` proposal | 44/68（64.71%） | rejected / rollback |
| 3 | base 重试 | 49/68（72.06%） | base pooled evidence |
| 4 | LR `1.25e-4` proposal | 47/66（71.21%） | noninferior / selected |

同一 base actor 的 pooled evidence 为 **95/134（70.90%）**。32-env 中的保守 proposal
相对该 pooled base 高约 **0.32 pp**，因此进入一次 64-env 放大。

## 64-env 放大结果

| Rollout | 对齐 actor | Filter on | 决策 |
|---:|---|---:|---|
| 1 | base | 101/133（75.94%） | initial anchor |
| 2 | LR `1.25e-4` proposal | 97/136（71.32%） | rejected / rollback |
| 3 | base 重试 | 93/136（68.38%） | base pooled evidence |
| 4 | LR `6.25e-5` proposal | 94/132（71.21%） | rejected / rollback |

放大后的同一 base actor pooled 为 **194/269（72.12%）**；两个 proposal 分别比它低
0.80 pp 和 0.91 pp，均已回滚。32-env 的 +0.32 pp 因而被判定为 rollout 方差，v91
路线正式拒绝，不运行 filter-off gate。正式全局最佳仍是 v79 的 139/193（72.02%）
filter-off 结果。

## 关键发现

- actor PPO transition count 在四轮均为 **0**，确认 reward-noise actor 梯度已隔离。
- actor gradient norm 降到 **0.1180–0.1276**，四轮均未发生 gradient clipping；v90
  对照为 7.43–9.42 且每轮裁剪。
- 64-env 中 `1.25e-4` 与 `6.25e-5` 都退化；仅降低优化噪声不足以让 blind actor
  内化 runtime filter 使用的摆动脚/下一台阶几何。
- 下一步冻结原 405-D actor，只新增 5 个可部署 CBF geometry 输入，并仅从 matched
  rescue episode 学习，以补上 actor 与 runtime CBF 之间的信息缺口。

## 溯源与 checkpoint 索引

- implementation/source commit：`21444c87a9a79f1a27e1091f6e7251b7131f23c4`
- targeted tests：30 passed
- input checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- 32-env provisional actor SHA-256：`e6a27287bb73ed454fc0ba278adb750763a65cbfc1d5e699832c28ade11da6cd`
- 64-env 最终 selected actor：原始 base，SHA-256
  `b0a717cef34d128e4175226b86780e5210cc5287558d78cb5a44e095e37fb600`
- 64-env selected checkpoint SHA-256：`4164ce0597d4da0a22cdd6e7d106237477b1ac99031d61f96439ccd26934ea1c`
- 64-env checkpoint 本机路径：`/home/lzqw/PycharmProject/safe100/HUMANOID/artifacts/paper_dual_v35/success_residual_only_v91_fixed125e4_63851b7_s201352638_n64_local1660/round_00.pt`
- hardware：本机 NVIDIA GeForce GTX 1660 SUPER
- seeds：`201352637`（32 env）、`201352638`（64 env）
- wall time：`173.43 s` + `189.77 s`

32-env 原始 JSON/CSV 位于 [`training/`](training/)，64-env 放大数据位于
[`scale64_training/`](scale64_training/)；由于未过 gate，checkpoint 二进制不提交 Git。

# v107 conservative outcome + transactional rollback

v107 从 v106 最佳 `round_03.pt` 的完整 415-D actor/848-D critic 精确继续，保留
CBF-RL 的 25% filter-on / 75% filter-off dual-reward PPO、persistent geometry 和
geometry/legacy 首层梯度 1:1 平衡。相对 v106 只做两个稳定性改动：episode outcome
advantage 权重从 1.0 降到 **0.5**；每个 proposal 在下一轮对齐 filter-off rollout
中评估，回落时原子恢复 actor、critic 和两个 optimizer，并把 LR 减半。

正式运行使用新 seed `201353361`、F2 18.4 cm、128 environments、1024 steps、
8 轮，初始 actor LR `5e-5`。共处理 1,048,576 transitions，在 RTX 4080 SUPER
上耗时 **208.76 秒（3 分 29 秒）**。

## 对齐结果与事务决策

奇数轮是保留 anchor 的基线/回滚后重测，并从该数据产生下一个 proposal；偶数轮
才是 proposal 的对齐评估。

| Rollout | actor | Filter off | LR | 决策 |
|---:|---|---:|---:|---|
| 1 | v106 anchor | 139/199 (69.85%) | `5.00e-5` | 建立 anchor |
| 2 | proposal 1 | **133/191 (69.63%)** | `4.89e-5` | rejected / rollback |
| 3 | anchor retry | 124/200 (62.00%) | `2.45e-5` | pooled anchor；产生 proposal 2 |
| 4 | proposal 2 | 116/201 (57.71%) | `2.45e-5` | rejected / rollback |
| 5 | anchor retry | **140/192 (72.92%)** | `1.22e-5` | pooled anchor；产生 proposal 3 |
| 6 | proposal 3 | 128/196 (65.31%) | `1.22e-5` | rejected / rollback |
| 7 | anchor retry | 131/191 (68.59%) | `6.12e-6` | pooled anchor；产生 proposal 4 |
| 8 | proposal 4 | 122/197 (61.93%) | `6.12e-6` | rejected / rollback |

四个 proposal 全部回落并被完整恢复；最高 proposal 只有 69.63%。训练结束时选择的
仍是输入 v106 actor，hash 与输入完全相同。相同 anchor 的四次 filter-off rollout
从 62.00% 到 72.92% 波动，pool 后为 **534/782 = 68.29%**。最高 72.92% 是同一
anchor 重测，不是新模型，也低于 75%，因此没有运行独立 gate；全局选择仍为 v79。

## 诊断与结论

- same-interface continuation 明确保留了 v106 的 geometry columns；没有重新零初始化。
- outcome credit 平均覆盖 81.30% transitions，权重确认为 0.5；GAE/credit cosine
  平均 0.254。
- geometry 自适应倍率为 8.49×–12.92×，每轮缩放后 geometry/legacy ratio 为 1.0。
- proposal 1 的 moving KL 为 `3.64e-6`，后续随 LR 缩小最低到 `8.67e-8`；即使
  更新已经极小，四个 outcome proposal 仍无一改善 filter-off。

因此 v106 的 71.58% 峰值不能解释为可稳定累积的 outcome-gradient 提升。问题不再是
outcome 权重过大或 SGD 步长过大，而是高噪声 stochastic episode outcome 对局部动作的
credit 不够因果。下一步回到论文原生 GAE + CBF dual reward，降低近收敛 policy 的
exploration std，使训练 rollout 更接近 filter-off deployment，再保留事务回滚。

## 文件与溯源

- source commit：`e6ec3fdad1586fa95bba79713f212d9cca54df8e`
- base v106 checkpoint：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/outcome_geometry_v106_58ad05d_64e_s201353360/round_03.pt`
- base checkpoint SHA-256：
  `4b1e558fb21eea162d0585263f7af018b34d6996877b9788ebcf1983d52c274c`
- retained actor SHA-256：
  `9053bbf42079f397a83d73c6d6a9f5b508021ebcf74f4f441ce5b964939ff935`
- v107 retained `round_00.pt` SHA-256：
  `1aa0e63be195296065d89719e1238120bf0bfef0896a816a2b979ee0e3092057`
- final restored `round_08.pt` SHA-256：
  `6e703844068a154c9e6889c8ae96dab341a345f2dbd79b92102e74e675cc0c26`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/outcome_transactional_v107_e6ec3fd_128e_s201353361`

未通过 gate 的模型二进制未上传 GitHub；完整训练 JSON/CSV、checkpoint 哈希和选择
依据均保存在本目录。

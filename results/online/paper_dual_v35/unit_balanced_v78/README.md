# v78 Eq. (27) velocity/displacement reward 单位平衡

v78 根据 v76 实测的约 9:1 raw margin/proximity 比例，将 margin weight 从 1 降到
0.1，使 velocity-scale Eq. (22) margin 与 one-step-displacement Eq. (27)
proximity 对 PPO 的贡献接近。其余保持 100% safety-filtered execution、nominal
action storage、next-riser clearance、单次 full-batch SGD 和事务回滚。训练使用
128 environments、4×1024 steps，RTX 4080 SUPER 耗时 **134.87 秒**。

## 结果

| Rollout | Filter on | LR | 结论 |
|---:|---:|---:|---|
| 1，base | 183/274 (66.79%) | `5e-5` | anchor |
| 2，proposal 1 | 184/277 (66.43%) | `4.81e-5` | rejected / rollback |
| 3，同一 base 重试 | 184/269 (68.40%) | `2.40e-5` | 产生 proposal 2 |
| 4，proposal 2 | **194/276 (70.29%)** | `2.40e-5` | accepted / selected |

proposal 2 相对紧邻的同一 base 重试提高 1.89 pp，但仍未达到 75% filter-on gate，
所以不追加 filter-off 评估或上传 checkpoint。跨独立训练进程的 base rollout 与 v76
相差数个百分点，再次说明不能用不同 rollout 的 raw rate 做精细超参数排序。

按 RewardManager `step_dt=0.02` 修正后，四轮 CBF contribution 为
`-0.00316`–`-0.00293/step`，nominal reward 为 `+0.01945`–`+0.02087/step`；
margin 与 proximity contribution 各约 `-0.0014` 和 `-0.0016`。单位桥接准确实现，
CBF penalty 从 v76 的 nominal reward 73%–82% 降到约 14%–16%，但 fully-filtered
成功率没有超过 v76。下一步用 v72 完全相同的 base、seed、25/75 execution 和 LR
做 controlled reward comparison，检验较弱且平衡的 CBF reward 是否更适合
filter-free deployment distribution。

## 文件与溯源

- training source commit：`a7b2d9e0da02b2ba30bd3e24e80531bfd17d39b5`
- telemetry `dt` 命名修复 commit：`824d6e9`
- base checkpoint SHA-256：
  `3285223174b01c97009db54361042c4c3d2d87054ca2156c84769f6d13ceccbc`
- selected aligned round-3 checkpoint SHA-256（未上传模型）：
  `b6f7b03deb24faa9b83759e759f920ca4350f4c9c1b5c5bcbc39420a12ec1f85`
- selected actor SHA-256：
  `d0eb24d33775ceaf0cebfe98b3ed0421c2aa33e84fd6ef84de4fd3d985f9ef13`
- final unaligned checkpoint SHA-256：
  `0669c9fa63312be3c2aeab0e8b5c1539a10ec07aa42bf3d4497dc64eb42e1111`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/unit_balanced_v78_a7b2d9e_s201352623`
- `training/` 保存完整 JSON/CSV；`decision_summary.json` 保存机器可读结论。

原始训练 JSON 由 `a7b2d9e` 生成，其中 `rollout_cbf_*_reward_mean_per_transition`
字段是 reward-term raw 值；本页及 decision 文件均已明确乘 `0.02` 后再与 PPO total
比较。`824d6e9` 之后的训练同时记录 raw 与 manager-weighted 字段。

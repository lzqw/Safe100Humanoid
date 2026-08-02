# 人形机器人长楼梯纠偏与安全在线更新：实现和测试报告

更新日期：2026-08-01
执行平台：RTX 4080 SUPER，MJLab/MuJoCo-Warp，CUDA GPU physics
范围：纯仿真；不包含 Unitree 真机、网络接口或 sim2real

## 本轮结论

本轮完成了三个原先缺失的在线训练环节，并在 4080 上实际运行：

1. 只从**真实 CBF 投影事件前 10 步**保存完整仿真状态，再用于 hard-case
   rollout 起点；不把普通跌倒状态或伪造事件混入 bank。
2. 候选策略由同 seed、同初始状态的 D0/目标域/邻域配对评估决定是否事务性
   接纳，置信区间改为确定性 paired bootstrap。
3. 根据实际 `CBF interventions / crossed riser` 调节下一轮探索标准差；rollout
   一旦出现跌倒，本轮接纳后强制采用下界因子 `0.8`。

使用保守 actor 学习率 `2e-6` 的单次 GPU 更新通过了完整门控。它是框架集成
smoke，不是论文级长期训练结果。随后使用接纳 checkpoint 做了同一初始状态
签名下 CBF 开/关各 64 episode 的独立评估。

## 整体算法

每轮在线更新执行：

```text
基础爬楼 actor + 人类式闭环遥控命令
        │
        ├─ actor 看：5 帧本体状态 + [vx, vy, wz] 遥控历史
        ├─ privileged critic 看：actor 状态 + 地形/接触 + CBF/命令延迟状态
        │
        ├─ a_policy ~ πθ(.|s)              写入 PPO storage 和 old log-prob
        ├─ a_nominal = wrapper_clip(a_policy)
        ├─ a_safe = CBF_projection(a_nominal)
        └─ a_executed = a_safe（训练有 shield）
                    │
          Dual-CBF reward + 投影前 10 步信用回传
                    │
          保守 full-policy PPO（actor/critic，冻结 obs normalizer）
                    │
          cheap precheck：finite / KL / clip / saturation
                    │
          paired bootstrap gate：D0 / DQH / DQNH
                    │
             接纳并调 std，或完整回滚模型和优化器
```

PPO 的概率比始终使用 actor 实际采样的 `a_policy`，不会把 CBF 投影后的
`a_safe` 冒充成策略样本。执行动作与策略动作分别记录，因而能够对投影前状态
分配信用，同时保持 on-policy log-prob 语义。

遥控纠偏仍是高层闭环：仿真操作者根据楼梯中心线横向误差与航向误差产生有界
`vy/wz`，actor 只看到与真机手柄相同形式的 `[vx, vy, wz]`，看不到世界坐标、
中心线标签或 privileged CBF 状态。

## Hard-case 状态包含什么

仅恢复机器人 `q/qd` 不足以复现事件前的 MDP 状态。本实现同时保存和恢复：

- root pose（相对各并行环境原点）、root velocity、joint state/targets；
- 当前、前一、前前一动作；CBF 名义/安全/执行动作和 barrier 内部量；
- 遥控 raw/delivered command、导数、延迟步数、完整 delay queue 和脉冲状态；
- contact air-time/history；
- actor、critic 和 privileged observation 的 circular history 与 push count；
- episode length、踏步索引，以及进度/踏步奖励的内部基线。

bank 有界为 256 条（本 smoke 中得到 57 条），按实际投影幅度做 priority
sampling。每次 rollout 先正常 reset，再把指定比例环境替换成 bank 状态；本次
训练使用 50% hard case / 50% bottom start。正式运行的默认比例仍为 25%，避免
过拟合事故附近状态。

## GPU 状态恢复验收

任务 `Unitree-G1-Stairs-Online-DQH`，4 个并行环境：

| 检查 | 结果 |
|---|---:|
| 跨环境完整状态恢复最大绝对误差 | `1.7881e-7` |
| 真实投影前状态数 | 4 |
| hard-case 起点 | 2/4 |
| 恢复后继续一步 | finite，通过 |
| 伪造 CBF 事件 | 0 |

原始证据：`artifacts/online_framework_v2/hard_case_restore_smoke.json`；完整日志：
`logs/online_framework_v2/hard_case_restore_smoke.log`。

## PPO 更新和事务门控

第一次仍用 `1e-5` actor 学习率，clip fraction 为 `0.37598 > 0.30`，在便宜的
预检查阶段被拒绝，没有启动候选仿真评估。保持阈值不变，将学习率降到 `2e-6`
后得到：

| 项目 | 数值 |
|---|---:|
| PPO clip fraction | 0.02832 |
| update KL | 0.00022765 |
| total KL from base | 0.00022766 |
| action saturation fraction | 0.31543 |
| policy storage action error | 0 |
| executed action routing error | 0 |
| old log-prob 重算最大误差 | `2.57e-5` |
| hard-case start | 2/4 |
| bank 新增 / 最终 | 28 / 57 |
| 实际 CBF intervention/riser | 1.275 |
| rollout 跌倒事件 | 3 |

smoke 配对门控为每域 2 episode，只用于验证链路。目标域成功率和跌倒率均未
退化，CBF 需求差值为 `-1.0/riser`，paired bootstrap 区间在该单 replicate
下退化为同一点；候选被接纳。因为训练 rollout 有跌倒，接纳后的探索标准差
因子为 `0.8`。

checkpoint：
`artifacts/online_framework_v2/online_dqh_hardcase_smoke_v6/accepted_final.pt`

SHA256：
`e5f66ba50107cde74d310ea6211923a02052cc24a87cd241854fd42c8eed1b9f`

## 64-episode CBF 开/关结果

两组的初始状态签名相同：
`1c7921aa5c1462b5c065cffc96e10744a3bb2df6bbf992fbacaecf856a43751e`。

| 指标 | CBF on | CBF off |
|---|---:|---:|
| 成功率 | 92.19% | **95.31%** |
| 跌倒率 | 7.81% | **4.69%** |
| 侧向跌倒率 | 7.81% | **3.13%** |
| 侧边越界率 | 10.94% | **6.25%** |
| 平均中心线误差 | 0.241 m | **0.217 m** |
| 平均 episode 最大中心线误差 | 0.621 m | **0.530 m** |
| 平均最大航向误差 | 1.481 rad | **1.406 rad** |
| 平均到达踏步 | **8.953/9** | 8.891/9 |
| 实际 CBF intervention/riser | 0.726 | 0 |
| CBF-off 反事实需求/riser | — | 0.678 |

当前样本下，toe--riser CBF 动作投影没有改善整体成功率，反而与更高的侧向
跌落相关。这不证明 CBF 普遍有害：当前 CBF 只约束脚尖撞台阶立面，不约束
楼梯侧边或整体动态稳定性；它的投影也可能在纠偏转向时扰动原有 gait。

关闭 CBF 的策略表现已经有效，但尚未通过严格的“CBF 独立”门槛。原因是其
反事实投影需求 `0.678/riser` 仍高于预设 `0.10/riser`。因此当前正确表述是：

> CBF-off 在本次 64 episode 测试中表现更好，但策略尚未被证明在全部目标域
> 上不再依赖 CBF；需要继续降低反事实 barrier 需求，并做更大规模配对门控。

## 视频结果

CBF-off、seed 42 的确定性 rollout：597 步成功到达第 9 级，未跌倒、未侧边
越界；平均中心线误差 0.070 m，最大 0.281 m。

视频：
`videos/online_dqh_hardcase_v6/g1-stairs-online_dqh-filter-off-seed42-step-0.mp4`

SHA256：
`09a199337c733e9dd7b21549882e091294e9a9316803a8733ff07999b4d578bf`

## 测试与代码位置

- 纯算法测试：`24 passed in 11.10s`；
  `logs/online_framework_v2/pytest_hardcase_bootstrap_final.log`
- 完整状态捕获/恢复：
  `third_party/unitree_rl_mjlab/src/tasks/stairs_cbf/hard_cases.py`
- paired bootstrap、CBF/riser 和 adaptive std：
  `third_party/unitree_rl_mjlab/src/tasks/stairs_cbf/online.py`
- 在线收集、事件前 10 步入 bank、事务门控：
  `third_party/unitree_rl_mjlab/experiments/scripts/online_refine_stairs.py`
- GPU 状态恢复 smoke：
  `third_party/unitree_rl_mjlab/experiments/scripts/smoke_hard_case_restore.py`
- 64-episode JSON/CSV：
  `artifacts/online_framework_v2/online_dqh_hardcase_smoke_v6/`

## 尚未完成

- 本轮只有一次小步在线更新；未进行数千轮持续更新。
- bootstrap 集成门控只有 2 episode/域、1 replicate，不能作为统计结论；正式
  接纳应恢复至少 32 episode × 3 paired replicates。
- hard-case curriculum 已包含 bottom 和 pre-intervention 两类起点，但附件建议
  的 15% 邻近楼梯几何混合尚未接入同一个训练 rollout；邻域目前只用于 gate。
- 18 级 D4H 仍是困难域，先前 32 episode 成功率只有 43.75%；本轮 quick DQH
  的结果不能外推到 D4H。
- 未运行真机，未验证真实手柄延迟、人类间歇纠偏或执行器不确定性。

## 追加：5 次 CBF-off 门控候选（v7）

用户要求增加轮次后，从 v6 accepted checkpoint 继续运行 5 次候选更新。训练
rollout 使用 32 环境、512 步、25% hard-case start、CBF-on；接纳评估使用
CBF-off 的 D0/DQH/DQNH，每域 16 episode × 3 个配对重复。actor 学习率保持
`2e-6`，没有放宽 KL、clip、D0 retention 或邻域安全阈值。

| 候选 | 结果 | KL | clip | rollout CBF/riser | 原因 |
|---|---|---:|---:|---:|---|
| 1 | 接纳 | 0.000233 | 0.0252 | 1.206 | DQH 反事实需求显著下降 |
| 2 | 回滚 | 0.000300 | 0.0560 | 1.196 | 无严格改善 |
| 3 | 回滚 | 0.000092 | 0.0026 | 1.134 | 无严格改善；D0 保留失败 |
| 4 | 回滚 | 0.000049 | 0.0000 | 1.329 | 无严格改善；D0 保留失败 |
| 5 | 回滚 | 0.000324 | 0.0551 | 1.720 | 无严格改善 |

第 1 个候选的 CBF-off 配对门控中，DQH 反事实需求从 `0.747` 降至
`0.660/riser`，paired-bootstrap 95% 区间 `[-0.162,-0.0208]`。成功率均值
`91.67% → 87.50%`、跌倒率 `8.33% → 12.50%`，但各自区间均接触 0，未达到
“显著退化”条件；其余安全条件通过，因此接纳。训练 rollout 有 6 次跌倒，
接纳后的探索 std 自动乘 `0.8`。后四次更新均未通过事务门控，最终 deterministic
MLP 保持在第 1 个接纳点。

### GPU 数值一致性审计

32×512 rollout 暴露出，同一网络按 32 样本逐步前向与 16384 样本 flatten
前向时，CUDA float32 GEMM 会产生数个 ULP 的差异。当前不再只看聚合 log-prob：

- Gaussian mean/std 参数逐元素误差必须 ≤`1e-5`；实际最大 `3.81e-6`；
- 12 维 log-prob reduction 误差必须 ≤`2e-4`；实际 smoke 为 `5.98e-5`；
- action storage 和 executed routing 误差仍必须分别 ≤`1e-6`/`1e-5`；实际均 0。

真正的分布参数变化 `2e-5` 会被测试拒绝。最终纯算法测试为
`26 passed in 9.27s`，日志：
`logs/online_framework_v2/pytest_online_refinement_26_param1e5.log`。真实 GPU
更新证据：`artifacts/online_framework_v2/logprob_distribution_audit_32x512_v2.json`。

### 相同 128 初始 episode 的 v6/v7 对比

初始状态签名均为
`0977ab0d66e2c2eb5b04fa3c027e66f5b862a8dceb825e8da3fd98f9532de388`。

| 指标 | v6 | final v7 | 变化 |
|---|---:|---:|---:|
| CBF-off 成功率 | 89.84% | **91.41%** | +1.56 pp |
| 跌倒率 | 10.16% | **8.59%** | -1.56 pp |
| 侧向跌倒率 | **7.03%** | 7.81% | +0.78 pp |
| 侧边越界率 | **7.03%** | 8.59% | +1.56 pp |
| 平均中心线误差 | 0.217 m | **0.205 m** | -0.012 m |
| 平均到达踏步 | 8.836/9 | **8.906/9** | +0.070 |
| 反事实 CBF demand/riser | 0.984 | **0.721** | -26.8% |
| 反事实 correction mean | 0.00150 | **0.00110** | -26.3% |

因此，多轮事务筛选改善了总体成功率、跌倒率、中心线误差和 CBF 依赖，但没有
改善侧边越界指标。`0.721/riser` 仍远高于 CBF-independence 门槛 `0.10`，不能
宣布已完全摆脱 CBF。

最终 checkpoint：
`artifacts/online_framework_v2/online_dqh_offgate_round5_v7c/accepted_final.pt`

SHA256：
`cc27f228809a4a5b9862119eded31e81588c95ad066168b77a77232d035620a5`

### 最终视频（成功与失败均保留）

- seed42：第 8 级跌倒，失败证据；SHA256
  `c23b8f672e3c738a8acd153b0503eb72ea2bcb6d34b04623bfb6b2c346de8759`；
- seed43：458 步到达第 9 级，未跌倒，平均中心线误差 0.129 m；SHA256
  `3c031c46441799cb88d23204cc95660157de9f94874c5d93ef3daf1196c7d6f6`。

两段视频都在 `videos/online_dqh_final_v7/`，没有删除失败 episode。

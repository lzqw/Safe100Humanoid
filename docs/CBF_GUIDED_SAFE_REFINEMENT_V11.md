# CBF-Guided Safe Online Refinement v11

更新日期：2026-08-02

执行范围：Unitree G1 爬楼梯纯仿真，MJLab/MuJoCo-Warp，RTX 4080 SUPER。
不包含真机执行、Unitree 网络接口或 sim2real 完成声明。

## 1. 研究问题

基础策略在 6 阶 waypoint 楼梯中训练，部署域改为持续 joystick 命令和更长
楼梯。长时横向/航向漂移、接触误差累积和后半程 CBF 介入形成 residual rare
failures。v11 的目标不是关闭 CBF，而是在 CBF 持续保护下，通过少量 on-policy
数据逐步降低 nominal policy 对安全层的依赖，同时保留 D0 能力。

## 2. 完整数据流

每个时刻行为策略采样原始动作：

\[
a_t^{\mathrm{policy}}\sim\pi_b(\cdot\mid o_t).
\]

CBF 只在 environment action path 上产生安全动作：

\[
a_t^{\mathrm{safe}}=\mathcal F_{\mathrm{CBF}}
(s_t,a_t^{\mathrm{policy}}).
\]

环境执行安全动作，但 PPO buffer 保存 `a_policy` 及其 behavior log probability。
ratio 始终为：

\[
\rho_t(\theta)=
\frac{\pi_\theta(a_t^{\mathrm{policy}}\mid o_t)}
{\pi_b(a_t^{\mathrm{policy}}\mid o_t)}.
\]

GPU rollout 会同时审计 policy/storage、nominal/safe/executed action routing 和
behavior Gaussian 参数；safe action 进入 PPO storage 会立即报错。

## 3. v11 策略损失

v11 保留 single clipped PPO surrogate，并加入原始基础策略 anchor：

\[
\mathcal L=
-J_{\mathrm{clip}}
+c_v\mathcal L_V
-c_H\mathcal H
+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_0).
\]

当前参数：

| 参数 | 数值 |
|---|---:|
| PPO clip | 0.03 |
| actor learning rate | 5e-6 |
| critic learning rate | 1e-4 |
| epochs | 1 |
| minibatches | 4 |
| target KL | 0.003 |
| base anchor beta | 0.01 |
| max grad norm | 0.5 |

`pi_0` 的 MLP 来自原始 CBF-RL checkpoint；observation normalizer 和 bounded
online standard deviation 使用部署时相同表示。所有 anchor 参数冻结。Actor 仍为
full-policy update，但使用 0.10/0.25/0.50/1.0 的逐层学习率倍率，log-std 冻结。
拒绝候选后仅将 actor 学习率减半，禁止缩放 exploration std；否则会在没有 actor
均值更新时制造与冻结基础分布之间的虚假大 KL。

## 4. CBF learning signals

保留 Dual CBF reward 和 10 步前置信用：

\[
\widetilde r_t=r_t^{\mathrm{task}}
+\lambda_{dual}r_t^{dual}
-\lambda_{pre}c_t^{pre}.
\]

在 normalized GAE 之后增加 policy-only intervention shaping：

\[
\widehat A'_t=\widehat A_t-\lambda_I
\operatorname{clip}\left(
\frac{\lVert a_t^{safe}-a_t^{policy}\rVert}{s_I},0,1
\right)\mathbf 1[I_t].
\]

当前 `lambda_I=0.075`、`s_I=0.05`。它不修改 critic return，因此 critic 仍学习
任务、Dual reward 和时间信用的 state value；直接 shaping 仅影响 policy surrogate。

## 5. Full privileged critic

Actor 接收 405 维、5 帧历史，不接收 privileged CBF 或楼梯几何。v11 critic 为：

- 405 维 actor history；
- 283 维原始 privileged critic；
- 150 维 online privileged deployment state；
- 合计 838 维。

新增 failure anticipation 特征：5 步 barrier history、5 步 correction history、
barrier derivative 和 5-step extrapolated margin。它们均对应上一动作执行后的当前
状态，不包含当前待采样动作产生的 projection，因此可合法进入 `V(s_t)`。

critic burn-in 冻结 actor，至少运行一轮；若 explained variance 小于 0.5，自动
继续到最多 4 轮。仍不达标则停止 actor PPO。

## 6. Hard-case curriculum

当前 rollout 初始分布：

- 65% 固定目标楼梯底部；
- 20% 从真实 CBF projection 前 10 步的完整 Markov state 恢复；
- 15% 从底部开始，但使用邻近 joystick speed/delay 扰动。

后两类仍由冻结的当前 behavior actor 重新采样动作，所以不是 off-policy replay。
hard-case state 包含 robot、action manager、command delay queue、contact/history、
observation circular history、reward baseline 和新增 CBF history。跨 DQ→D4 时不会
恢复 hard-case bank，避免把 9 阶状态写入 18 阶几何。

## 7. Safe Improvement Operator

候选首先通过 KL、clip fraction、action saturation、梯度有限性和总 drift 预检查。
随后在完全配对的 D0/target/neighbor 初始条件上评估。综合分数为：

\[
S=Success+0.02R-2Fall-0.05I_{CBF/riser}
-D_{KL}(\pi\Vert\pi_0).
\]

只有以下条件同时成立才接纳：

1. target success/fall/CBF demand 未发生统计退化；
2. D0 retention 不低于 base 2 个百分点；
3. neighbor success/fall 未超过容忍区间；
4. 至少一个 target 指标严格改善；
5. 综合安全分数严格提高；
6. 所有 paired initial-state signatures 一致。

失败时原子恢复 actor、critic 和 optimizer，然后 actor LR 乘 0.5，探索 std 保持
与冻结基础分布一致。CBF-on/off independence 仅在仿真最终审计中运行；真实机器人设计始终保留
runtime CBF 和 emergency protection。

## 8. Staged deployment curriculum

正式单 seed 流程为：

1. Stage 1：DQ（9 阶、训练几何、open-loop joystick），5 个候选轮次；
2. 仅当至少一轮接纳、DQ success >= 60% 且最终安全分数高于 baseline，进入
   Stage 2；
3. Stage 2：D4（18 阶、目标 geometry profile），3 个候选轮次；
4. DQN/D5 分别作为邻域泛化 gate，不参与对应 target rollout 更新。

人工中心线纠偏 DQH/D4H 被保留为独立可选模式，不混入默认 open-loop OOD
实验。

## 9. 当前验证证据

- 纯测试：39 passed；
- 三路 hard-case GPU restore：最大误差 5.96e-8；
- 8 env x 128 step 真实 PPO update：actor 405、critic 838、参数有限；
- behavior log-prob 最大误差 1.35e-5；
- behavior distribution 参数最大误差 1.43e-6；
- policy storage error 和 executed routing error 均为 0；
- anchor KL：update 前 0，update 后 8.22e-5；
- intervention advantage penalty mean：7.99e-4；
- critic explained variance：0.589。

端到端 v9 gate smoke 中，候选在 DQH 的安全分数提高 0.0182、CBF/riser 下降，
但 DQNH success 从 50% 降到 25%、fall 从 50% 升到 75%，因此被正确回滚。
这证明 Safe Improvement Operator 没有用 target 的局部改善掩盖邻域退化。

## 10. v10 诊断与 v11 正式实验

首次 v10 长运行在第一轮正确回滚后暴露了 distribution-level anchor 问题：旧逻辑
把 exploration std 乘 0.8，第二轮 `base_anchor_kl_before_update` 从约 `2.1e-4`
跳到 `0.518`。确定性 actor mean drift 仍只有 `8.8e-5`，但该轮不满足同一 KL
定义，因此 v10 在第二轮完成后被主动停止，完整目录仅作为失败诊断保留，不作为
正式结果。

RTX 4080 tmux：`staged_v11`

输出：`artifacts/online_framework_v3/staged_dq_to_d4_v11`

日志：`logs/online_framework_v3/staged_dq_to_d4_v11.log`

正式配置为单训练 seed 42、32 env、256 step/round（8192 transitions）、5 个 DQ
候选轮次；每次 gate 使用 16 env x 3 paired repeats，即 48 episodes/策略/域。
critic burn-in 第一轮 explained variance 为 0.736，超过 0.5 门槛。

| 轮次 | actor LR | KL from base | DQ success old→new | DQ fall old→new | CBF/riser old→new | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 5.0e-6 | 2.33e-4 | 41.67%→58.33% | 58.33%→41.67% | 1.130→0.979 | 95% 区间触及 0，回滚 |
| 2 | 2.5e-6 | 9.25e-5 | 41.67%→50.00% | 58.33%→50.00% | 1.119→1.060 | 无严格改善，回滚 |
| 3 | 1.25e-6 | 2.27e-5 | 50.00%→54.17% | 50.00%→45.83% | 1.140→1.146 | DQN 退化，回滚 |
| 4 | 6.25e-7 | 9.28e-6 | 50.00%→39.58% | 50.00%→60.42% | 0.988→1.117 | DQ 退化，回滚 |
| 5 | 3.125e-7 | 1.53e-6 | 43.75%→47.92% | 56.25%→52.08% | 1.026→1.248 | 无严格改善，回滚 |

五轮均未通过，事务回滚后的 `accepted_total_kl_from_base=0`。Stage 1 gate
未通过，因此 Stage 2 D4 **没有启动**；这正是 staged safe-improvement 条件的预期
行为，不能称为 D4 训练完成。

最终回滚策略独立评估：

| 域 | success | fall | mean return | CBF/riser | correction mean |
|---|---:|---:|---:|---:|---:|
| D0 | 87.50% | 6.25% | 8.170 | 0.810 | 6.72e-4 |
| DQ | 45.83% | 54.17% | 3.572 | 1.274 | 4.46e-3 |
| DQN | 62.50% | 37.50% | 5.016 | 1.122 | 3.45e-3 |

CBF independence audit 未通过。DQ filter-on 为 45.83% success、1.274
interventions/riser；filter-off 为 54.17% success，但反事实 correction mean 仍为
2.07e-3。一次 filter-off 成功率较高不等于不依赖 CBF；判定依据是 nominal action
仍持续违反 barrier，需要非零安全修正。

## 11. 第一轮候选大样本复核

第一轮在 48 episodes/策略/域时改善明显但置信区间触及 0。为排除小样本边界，
使用 `reevaluate_online_candidate.py` 增加到 32 env x 3 repeats，即 96 个完全配对
episodes/策略/域。结果仍被同一个 gate 拒绝：

| 域 | 指标 | base | candidate |
|---|---|---:|---:|
| D0 | success / fall | 86.46% / 2.08% | 91.67% / 4.17% |
| DQ | success / fall | 43.75% / 56.25% | 42.71% / 57.29% |
| DQ | CBF/riser | 0.972 | 0.923 |
| DQN | success / fall | 50.00% / 50.00% | 51.04% / 48.96% |
| DQN | CBF/riser | 1.029 | 0.949 |

DQ success delta 为 -1.04 个百分点，fall delta 为 +1.04 个百分点；综合安全分数
从 -0.668 变为 -0.697。候选虽降低 CBF demand，却没有提高目标域安全，因此保持
rollback。当前结论是：v11 信号和安全门实现有效，但这 5 个小步 PPO 候选没有产生
可稳健接收的 open-loop DQ 改善；不能声称策略已经内化 CBF，也不能进入 D4。

正式回滚 checkpoint：
`artifacts/online_framework_v3/staged_dq_to_d4_v11/stage1_dq/accepted_final.pt`。

第一轮未接收候选仅供诊断：
`artifacts/online_framework_v3/staged_dq_to_d4_v11/stage1_dq/candidate_round_001.pt`。

## 12. 视频检查

使用 DQ、CBF-on、deterministic actor、seed42 录制两段完整 episode：

| checkpoint | success | steps | max riser | CBF intervention integral | min foot edge clearance | side breach |
|---|---:|---:|---:|---:|---:|---:|
| rollback accepted_final | yes | 541 | 9 | 23.22 | +0.123 m | no |
| unaccepted round-1 candidate | yes | 552 | 9 | 19.16 | -0.0205 m | yes |

候选的单 episode 登顶且介入积分较低，但脚部进入楼梯侧边界外，因此视频并不能
推翻 96-episode gate 的拒绝结论。

本地视频：

- `videos/online_framework_v3/v11_rollback/g1-stairs-online_dq-filter-on-seed42-step-0.mp4`
- `videos/online_framework_v3/v11_candidate1/g1-stairs-online_dq-filter-on-seed42-step-0.mp4`

## 13. 提交范围

GitHub 只提交源程序、测试、可复现实验脚本和本报告。checkpoint、原始 JSON/CSV、
日志和视频保留在实验目录，不提交仓库。

# Task-First Constrained CBF-PPO v12

更新日期：2026-08-03

## 结论

v12 完成了 task-first constrained PPO、独立 task/fall/intervention critics、
自适应安全 multiplier、短期 risk head、success-gated CBF correction、1024-step
rollout、低比例 hard-case、候选族和事务式 rollback。实现通过 41 项测试、GPU
smoke、checkpoint roundtrip 和两组正式 matched experiment。

当前没有新的 accepted policy。最强候选在 DQH 上的点估计同时提高任务表现并
降低 CBF demand，但 128 episode/domain 的 paired audit 仍未得到严格 95% 改善，
且 D0 retention 不通过，所以正确 rollback。运行时 CBF 没有被关闭。

## 为什么正式域改为 DQH

同一 checkpoint、seed=42、CBF-off、每组 512 episode 的 command-interface
诊断为：

| 域 | joystick | success | fall | side fall | center error |
|---|---|---:|---:|---:|---:|
| DQ | 开环随机脉冲 | 46.88% | 53.12% | 50.78% | 0.395 m |
| DQH | 闭环中心纠偏 | **88.28%** | **11.72%** | **9.77%** | **0.209 m** |
| DQN | 开环随机脉冲 | 46.88% | 52.93% | 51.17% | 0.402 m |
| DQNH | 闭环中心纠偏 | **91.60%** | **8.40%** | **5.66%** | **0.200 m** |

Actor 仍只接收标准 `[vx, vy, wz]`，不读取中心线或 world-state privileged
变量。该对照确认开环命令缺少纠偏是原 DQ 失败的主要原因之一，因此 v12 在
DQH 收集数据，并用 D0、DQNH 做 retention/neighbor gate。

## 算法

环境 task reward 中固定 fall/CBF shaping 权重均为 0。算法分别训练
`V_task`、`V_fall`、`V_intervention`，并使用：

```text
A_con = A_task - lambda_fall * A_fall - lambda_intervention * A_intervention
```

预算从 formal baseline 得到：fall budget 等于 baseline fall rate，intervention
budget 等于 `1.05 × baseline intervention/riser`。两个 multiplier 根据观测 cost
自适应更新，candidate rollback 时与 actor、critics、risk head 和 optimizer 一起
恢复。

正式配置：

- 64 env × 1024 steps/round；每轮 65,536 transition；
- 每轮 100–110 条 normal-start 完整 episode，超过最低 32 条要求；
- hard-case reset 8%，`hard_case_policy_weight=0`；
- neighbor-command reset 8%；
- risk horizon 50 steps，正式 AUC 0.642–0.706，Brier <0.245；
- correction success horizon 100 steps；
- actor LR 每轮固定 `2e-6`，拒绝后不永久减半；
- 每轮筛选 `{0.25,0.5,1.0,1.5}` 候选族。

验收使用 D0/DQH/DQNH、相同 initial-state signature 和 paired bootstrap 区间。
DQH 必须有严格 task improvement，fall/CBF demand 不能恶化；D0 与 DQNH 的
success/fall 使用 2% 容差；任何 precheck、retention 或 neighbor 条件失败都
rollback。

## 正式 correction ablation

两个 arm 除 correction weight 外相同，均运行 2 次 critic burn-in、3 个在线
round，每个 formal gate 为 48 episode/domain/policy。

| arm/round | fraction | DQH success | DQH fall | return Δ | CBF/riser Δ | 决策 |
|---|---:|---:|---:|---:|---:|---|
| correction r1 | 1.5 | 93.75%→93.75% | 6.25%→6.25% | +0.192 | +0.138 | rollback |
| correction r2 | 1.0 | 91.67%→93.75% | 8.33%→6.25% | +0.872 | -0.073 | rollback |
| correction r3 | 1.5 | 87.50%→89.58% | 12.50%→10.42% | +0.282 | -0.016 | rollback |
| no-correction r1 | 1.5 | 87.50%→95.83% | 12.50%→4.17% | +1.045 | -0.004 | rollback |
| no-correction r2 | 0.5 | **83.33%→93.75%** | **16.67%→6.25%** | **+1.342** | **-0.074** | rollback |
| no-correction r3 | 1.0 | 87.50%→93.75% | 12.50%→6.25% | +1.049 | -0.052 | rollback |

success-gated correction 每轮实际使用 1261–1624 个 eligible sample，但当前
交互预算下没有优于 no-correction。默认应保持 correction weight=0，直到 teacher
gate 或数据量改变后重新做 matched comparison。

## 128-episode/domain paired audit

审计最强的 no-correction round-2 candidate，16 env × 8 repeats，paired seeds
42–49，CBF-on。每个 policy 完成 D0/DQH/DQNH 各 128 episode，总计 768 episode。

| 域 | old success/fall | candidate success/fall | return old→new | CBF/riser old→new |
|---|---:|---:|---:|---:|
| D0 | 96.09% / 0.78% | 92.19% / 2.34% | 8.692→8.669 | 0.624→0.674 |
| DQH | 84.38% / 15.63% | **89.84% / 10.16%** | 7.822→8.769 | 0.864→0.643 |
| DQNH | 92.97% / 7.03% | 90.63% / 9.38% | 8.863→8.699 | 0.820→0.754 |

DQH paired deltas：

| metric | mean | 95% lower | 95% upper |
|---|---:|---:|---:|
| success | +0.0547 | -0.0156 | +0.1250 |
| fall | -0.0547 | -0.1250 | +0.0156 |
| return | +0.9469 | -0.0415 | +1.8698 |
| CBF/riser | -0.2202 | -0.6412 | +0.0596 |
| safe score | +0.1941 | -0.0449 | +0.4249 |

点估计方向正确，但所有区间触及 0。D0 success 92.19% 也低于 formal baseline
95.83% 减 2% 后的 93.83% retention floor。最终决策为 `rollback`，原因：

```text
target metrics show no strict improvement
target task metrics show no strict improvement
D0 retention bound violated
```

## CBF 独立性

两个 arm 的最终 accepted actor 都因 rollback 与输入 actor bitwise 相同。最终
CBF-on/off audit 仍有 0.673–0.687 intervention/riser，CBF-off counterfactual
demand 为 0.737–0.799/riser，未达到 near-zero demand。当前策略不能声明为
filter-free。

## 验证

- `41 passed`；
- 修改文件全部通过 `py_compile`；
- launcher 通过 `bash -n`；
- `git diff --check` 通过；
- 4×768 GPU smoke finite，action storage/routing error 为 0；
- candidate-family smoke 覆盖所有 fraction 和完整 rollback；
- 8×768 end-to-end 拒绝无 task improvement 的候选；
- 6 个 formal candidate 和大审计 candidate 均未污染 accepted actor。

## 下一步

当前瓶颈已从 v11 的“只降低 CBF demand”变为“DQH 点估计改善，但 D0/DQNH
泛化和统计功效不足”。下一轮应保持学习率和 gate 不变，在 on-policy collection
中增加 D0/DQNH command mixture 或使用多域 constrained gradient，并增加 paired
repeat 数。不得通过降低 retention 或置信区间要求来提升本次 rejected candidate。

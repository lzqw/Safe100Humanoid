# CBF-Proximal Online Policy Refinement v23

## Scope / 范围

v23 is an independent main path on top of the preserved v17–v22 history. It
tests whether a small on-policy update can improve one already calibrated pure
lateral context while the runtime CBF remains the executed safety mechanism.
No v17–v22 artifact, result, or interpretation is replaced.

v23 是在完整保留 v17–v22 历史之上的独立主线。它只检验一个问题：运行时 CBF
继续负责实际安全动作时，小步 on-policy 更新能否改善一个已经校准的纯 lateral
场景。既有证据、结果与解释均不修改。

## Policy, observations, reward / 策略、观测与奖励

- one actor with the original 405-D observation;
- one privileged critic with the corresponding 838-D observation;
- no appended five-dimensional failure observation;
- the environment executes the CBF-filtered action, while PPO stores the raw
  sampled policy action and its behavior log probability;
- reward is the existing base task reward plus fall event and dual-CBF reward;
- the `specialist_failure_signal` term is absent from the v23 runtime;
- ordinary CBF intervention is safety telemetry, never a failure label.

即：一个原始 405-D actor、一个 838-D privileged critic；不增加 failure
观测；环境执行 CBF 过滤后的动作，但 PPO 始终保存原始采样动作与 behavior log
probability；奖励只保留基础任务、fall 与 dual-CBF 语义。普通 CBF 介入只作为
安全遥测，不被解释成 failure。

## Update / 更新

At the start of round `k`, the current actor is frozen as `pi_k`. The actor
objective is

```text
L_actor = L_PPO-clip + 0.5 * KL(pi_theta || pi_k) - 0.0 * entropy.
```

The forward KL is the analytic diagonal-Gaussian expression. Reference means
and standard deviations are the round's stored behavior-distribution
parameters and are stop-gradient. The reference moves after every round; it is
never fixed to the original `pi_0`.

每轮开始时将当前 actor 冻结为 `pi_k`。KL 使用解析的对角高斯 forward
`KL(pi_theta || pi_k)`，参考均值与标准差来自该轮 behavior buffer，并完全
stop-gradient。下一轮参考随当前策略移动，绝不固定锚定 `pi_0`。

The critic uses one ordinary value loss on every transition. Advantages are
normalized once over the whole rollout. Actor and critic use separate Adam
optimizers.

## Frozen training constants / 冻结训练常量

| item | value |
|---|---:|
| rounds | 8 |
| environments | 64 |
| steps per environment | 1024 |
| actor / critic LR | `5e-6` / `1e-4` |
| PPO clip | `0.05` |
| maximum actor / critic epochs | 2 / 2 |
| minibatches | 4 |
| moving KL beta | `0.5` |
| target / hard KL | `0.003` / `0.01` |
| max gradient norm | `0.5` |
| base-std scale | `0.35` |
| std clamp | `[0.05, 0.25]` |
| log std / entropy | frozen / `0.0` |
| gamma / GAE lambda | `0.99` / `0.95` |

Target-KL is checked only after a complete actor epoch. If the first epoch is
above `0.003`, the second actor epoch is skipped. A hard rollback is permitted
only for a non-finite state, forward KL above `0.01`, raw-action/behavior
routing corruption, or optimizer-state corruption. Rollback restores actor,
critic, and both optimizers. Performance never causes rollback.

## Data and final policy / 数据与最终策略

Each round has exactly one on-policy rollout from ordinary physical resets.
There is no state restart, replay bank, dual rollout, candidate fraction,
screen, confirmation, D0 gate, validation selector, or best-so-far choice.
Round-start and round-end checkpoints are recovery/curve artifacts only. The
final policy is always the round-8 actor.

每轮只采集一个由正常物理 reset 开始的 on-policy batch。不使用 state restart、
failure/matched-success bank、dual rollout、候选比例、screen、confirmation、D0
性能门或 best-so-far。最终策略固定为第 8 轮 actor。

## Context and prospective sequence / 场景与前瞻顺序

The previously base-only calibrated `L_effect` context is reused without
reselection:

- base success: 68.75%;
- 160 failures among 512 episodes;
- lateral-failure purity: 92.5%;
- context file SHA-256:
  `650a97519168382bf4f7fc45580fa179cb3c51a1f18195f4850c5667d6f0d6a7`;
- parameter SHA-256:
  `f3d4470ec01c7f55982d93b4be53dcafb13c0d2d82f4c03f34835f84c99cf4ae`.

This satisfies the prospective 65–75% base-success and at least 85% purity
requirements. One fresh adaptation is run only after the implementation and
protocol are committed. A contact context is not run unless this lateral test
passes its point-estimate development gate.

## Final paired test / 最终配对测试

- target: 512 conditions in four batches of 128;
- D0: 256 conditions in two batches of 128;
- the base and round-8 actors receive identical initial conditions;
- report success, fall, return, intervention per riser, repairs, regressions,
  and paired bootstrap 95% intervals;
- confidence intervals are report-only;
- point gate: target success `>= +3 pp`, target fall `<= +1 pp`, D0 success
  `>= -5 pp`.

The gate is reported only after the fixed actor exists and never influences
training, rollback, checkpoint selection, or stopping.

## Explicit exclusions / 明确排除

No multi-context training, off-diagonal audit, macro aggregation, candidate
ablation, specialist shaping, failure precursor bank, matched-success bank,
grouped advantage, large defensive validation, or performance-selected
checkpoint is part of v23.

## Method references

- PPO: <https://arxiv.org/abs/1707.06347>
- PROTO (moving-reference policy regularization context):
  <https://arxiv.org/abs/2305.15669>
- Spinning Up PPO: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>

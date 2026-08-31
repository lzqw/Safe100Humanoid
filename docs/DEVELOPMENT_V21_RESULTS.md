# Safe100 specialist v21 development result / 开发阶段结果

Status: development complete; formal revision 2 frozen before any formal
adaptation or audit outcome.

状态：开发阶段完成；在任何正式 adaptation 或 audit outcome 出现之前，已冻结
formal revision 2。

## Result / 结论

The prospectively frozen selector chose **beta = 0**. The local
matched-success preservation variants did not outperform the v20-style control
on the primary development metric, the mean across `L_dev` and `C_dev` of
repair rate minus regression rate (`RR - RG`). No grid, metric, tie-break, seed,
or episode count was changed after observing this result.

前瞻冻结的选择器选中 **beta = 0**。在主要开发指标（`L_dev` 与 `C_dev` 上
`RR - RG` 的平均值）上，带 local matched-success preservation 的非零 beta
没有超过 v20-style control。观察结果后没有修改网格、指标、tie-break、seed
或 episode 数量。

| beta | mean `RR-RG` | worst-context `RR-RG` | mean regression rate | selected |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 0.616562 | 0.466446 | 0.170565 | yes |
| 1 | 0.593932 | 0.534242 | 0.179474 | no |
| 4 | 0.567241 | 0.440205 | 0.183903 | no |
| 16 | 0.540796 | 0.443560 | 0.186000 | no |

Beta 1 had the best worst-context score, but the frozen primary ordering is
mean score first. Beta 0 therefore wins without invoking a tie-break.

beta=1 的 worst-context 分数最好，但冻结规则首先比较 mean score，因此
beta=0 直接胜出，并未用到 tie-break。

## Fresh paired selection evidence / 全新配对选择证据

Each policy used 512 fresh paired episodes per excluded development context.
The base actor was evaluated once per context and paired against all four
trained candidates.

每个策略在每个排除于正式实验之外的开发 context 中使用 512 个全新配对
episode。base actor 在每个 context 评价一次，并与四个训练后候选逐一配对。

| context | beta | base SR | candidate SR | repairs / base failures | regressions / base successes | `RR-RG` |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| L_dev | 0 | 76.953% | 76.758% | 79 / 118 | 80 / 394 | 0.466446 |
| L_dev | 1 | 76.953% | 78.320% | 87 / 118 | 80 / 394 | 0.534242 |
| L_dev | 4 | 76.953% | 75.195% | 78 / 118 | 87 / 394 | 0.440205 |
| L_dev | 16 | 76.953% | 76.367% | 76 / 118 | 79 / 394 | 0.443560 |
| C_dev | 0 | 87.695% | 86.719% | 57 / 63 | 62 / 449 | 0.766677 |
| C_dev | 1 | 87.695% | 83.984% | 51 / 63 | 70 / 449 | 0.653622 |
| C_dev | 4 | 87.695% | 85.156% | 53 / 63 | 66 / 449 | 0.694277 |
| C_dev | 16 | 87.695% | 82.617% | 51 / 63 | 77 / 449 | 0.638032 |

The selector intentionally uses conditional repair and regression rates, not
raw success delta. Because their denominators differ, a positive `RR-RG` does
not imply a positive raw success delta.

选择器按预注册定义使用条件 repair/regression rate，而不是直接使用 success
delta。两者分母不同，因此正的 `RR-RG` 不等价于正的原始 success delta。

## Training completion / 训练完成情况

All eight runs completed the fixed budget of eight rounds. Accepted-update
counts were: `L_dev` beta 0/1/4/16 = 2/3/2/1, and `C_dev` = 4/1/2/5.
Training diagnostic evaluations participated in candidate selection and are not
used as the fresh beta-selection evidence above.

8 个训练均完成固定的 8 轮预算。接受更新次数为：`L_dev` beta 0/1/4/16 =
2/3/2/1，`C_dev` = 4/1/2/5。训练诊断参与了 candidate selection，不能替代
上表的全新 beta-selection 证据。

## Execution amendment / 执行修订

The first selection invocation stopped while constructing its read-only runner,
before any actor evaluation or metric write. Commit `51c2fb4` supplied only the
already frozen algorithm invariants required by that constructor. The retry
reused the original training artifacts without mutation and passed 24 binding
checks. The retry then completed 40/40 batches and 5,120 fresh actor rollouts.

首次选择调用在只读 runner 构造时退出，尚未评价 actor 或写入指标。提交
`51c2fb4` 仅补齐构造器要求、且已在训练协议中冻结的算法参数。重试未修改或
重跑训练产物，通过 24 项绑定检查，随后完成 40/40 批和 5,120 个全新 actor
rollout。

## Formal consequence / 对正式实验的影响

Formal protocol revision 2 freezes beta 0 for all ten deployment contexts.
Thus the registered v21 and control configurations collapse to the same actor
objective; both branches will still be executed independently as registered,
from the same base policy and deployment seed, before paired formal audits.
This preserves the negative development result instead of substituting a
post-hoc nonzero beta.

formal protocol revision 2 已为十个 deployment context 冻结 beta=0。因此，
登记的 v21 与 control 配置在 Actor objective 上退化为相同设置；仍会按照登记
方案从同一 base policy 与 deployment seed 独立执行两个分支，再进行配对正式
audit。这样保留开发阶段的负结果，而不会事后换用某个非零 beta。

Machine-readable evidence is in
`results/online/specialist_v21/development/`; the frozen formal protocol is
`results/online/specialist_v21/protocol_formal.json`.

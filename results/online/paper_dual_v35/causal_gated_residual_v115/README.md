# v115 causal discordant-treatment gate

v115 将 v113 的 memoryless gate 改为 64-D causal GRU，并只在真正 discordant 的
paired episode 上区分 `off失败/on成功`（rescued）与 `off成功/on失败`（harmed）。
前三个 seed 用于训练，第四个 seed 只作离线验证；部署阈值提高到 `0.6`，低置信度时
residual 精确关闭并退回冻结的 v79 actor。

## Paired 数据

| Split | Filter off | Filter on | Rescued | Harmed |
|---|---:|---:|---:|---:|
| Train，3×64 | 131/192 | 138/192 | 44 | 37 |
| Validation，1×64 | 46/64 | 45/64 | 13 | 14 |
| Total | 177/256 | 183/256 | 57 | 51 |

唯一 filter-off screen（seed `201353680`）为 **47/64 = 73.4375%**，仅差一个
episode 达到 75%。gate 只在 7.37% transition 激活，平均 residual norm 为
`0.000616`；mean reached riser 为 `7.938`。因为未达到门槛，没有运行独立 gate，
正式最佳仍保留 v79 的训练内 139/193 = 72.02%。

## 汇总写入故障

训练、候选保存和 screen 都已完成，但脚本在最后组装 `training_summary.json` 时把
Python 布尔值写成了 `false`，触发 `NameError`。因此本轮没有完整训练指标 JSON；
原始控制台日志、execution record、screen summary 和逐 episode CSV 均已保留。
该拼写已在 commit `8fc8a3b` 修复。遵照“不做很多校验/不要久等”的要求，没有为
补写诊断而重复这次约三分钟的 paired rollout。

候选二进制未上传，精确路径与 SHA-256 见 `checkpoint_index.json`。实现 commit 为
`cef3b929d0970e84776894ef95f9b259ec138295`。

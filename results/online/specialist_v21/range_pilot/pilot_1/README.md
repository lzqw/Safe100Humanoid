# v21 base-only range pilot 1 / v21 基础策略范围试验 1

This non-formal pilot completed normally at 2026-08-08 15:30:48 UTC. It
evaluated 12 frozen candidates in each of 12 context families, with 128
base-policy episodes per candidate (18,432 episodes total). No adaptation,
development selection, or formal audit was started.

本轮非正式范围试验于 2026-08-08 15:30:48 UTC 正常完成。12 个 context family
各自按冻结顺序评估了 12 个候选，每个候选运行 128 个基础策略 episode，共
18,432 个 episode。没有启动适配训练、development 选择或正式审计。

Prospective protocol commit: `4f469cd4d33fd12f275d203bd2ba5909fd9afa11`

Prospective protocol SHA-256: `ddde1a7128fe416e6c2829cfe0d7f99a2e3050ff47deed69ec6305ef29df7366`

Base checkpoint SHA-256: `cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8`

## Result / 结果

The pilot gates were 70–85% success, at least 25 falls, and dominant failure
purity. Seven families produced at least one qualifier; five require a focused
second range pilot.

试验门槛为成功率 70–85%、至少 25 次跌倒，并要求声明的失败机制占主导。7 个
family 至少产生一个合格候选；另外 5 个需要定向进行第二轮范围试验。

| Context | Qualifying seeds | Result |
| --- | --- | --- |
| C_dev | 11211, 11213 | Keep range |
| L1 | 11308, 11310, 11313, 11315–11319 | Keep range |
| L3 | 11509, 11511, 11513 | Keep range |
| C1 | 11811–11813 | Keep range |
| C2 | 11908 | Keep range |
| C4 | 12113 | Keep range |
| C5 | 12216, 12218 | Keep range |
| L_dev | None | Near the gates at 11108; fix smoothing and isolate lateral load |
| L2 | None | Difficulty reached the window, but lateral purity was too low |
| L4 | None | Pure endpoint was too easy; later points lost purity |
| L5 | None | Mechanism-pure but the entire range was too hard |
| C3 | None | Sharp difficulty cliff between the easy and hard brackets |

The exact candidate rows are the 12 `calibration_progress.json` files beside
this summary. [`summary.json`](summary.json) binds those compact files and the
external raw artifacts by deterministic SHA-256 manifests. The raw artifacts
contain 588 files (580,642,516 bytes) and remain on the compute host rather than
being added to GitHub.

精确的候选结果保存在本目录下 12 份 `calibration_progress.json` 中；
[`summary.json`](summary.json) 使用确定性的 SHA-256 manifest 绑定这些紧凑证据和
算力机上的原始产物。原始产物共 588 个文件、580,642,516 字节，因此不直接提交
到 GitHub。

## Scientific boundary / 科学边界

Pilot 2 may use only base-policy evidence from this pilot and the earlier failed
calibration. It must use fresh candidate/evaluation randomness and be frozen in
a committed protocol before execution. It remains range finding, not formal
context selection, and cannot use any adaptation outcome.

第二轮只能使用本轮和此前失败校准中的基础策略证据；必须使用全新的候选与评估
随机数，并在执行前将协议冻结提交。它仍然只是范围查找，不是正式 context 选择，
也不得使用任何适配结果。

# v21 preflight status / v21 预实验状态

Status at 2026-08-08 14:07:44 UTC: **stopped cleanly before any online adaptation**. The corrected revision-0 protocol evaluated every frozen `L_dev` base-policy candidate in order, but none passed the prospectively declared calibration gates. The fail-fast queue therefore did not start `C_dev`, any formal context, development beta selection, or any v20/v21 adaptation.

截至 2026-08-08 14:07:44 UTC：**任务已在任何在线适配开始前按规则停止**。修正后的 revision-0 协议依次评估了全部冻结的 `L_dev` 基础策略候选，但没有候选通过预先声明的校准门槛。因此 fail-fast 队列没有启动 `C_dev`、任何正式 context、development beta 选择或任何 v20/v21 适配。

Frozen source commit: `6dfdd45b5ca734c4cd0ee0886a189e5b27d4655d`

Frozen protocol SHA-256: `dc181bb604b8cbb8bc19f6abd95f7063302ff98d7ed495f99e1624e208ad360b`

Base checkpoint SHA-256: `cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8`

## L_dev base-only sweep

Each row contains 512 episodes. The frozen gates were success rate 70–85%, at least 100 falls, lateral failure fraction at least 80%, and second failure fraction at most 30%.

| Seed | Severity | Success | Falls | Lateral fraction | Second fraction | Qualified |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 9108 | 0.000 | 86.72% | 68 | 70.59% | 29.41% | No |
| 9109 | 0.091 | 88.09% | 61 | 73.77% | 26.23% | No |
| 9110 | 0.182 | 88.28% | 60 | 80.00% | 20.00% | No |
| 9111 | 0.273 | 87.89% | 62 | 74.19% | 25.81% | No |
| 9112 | 0.364 | 88.09% | 61 | 63.93% | 36.07% | No |
| 9113 | 0.455 | 90.82% | 47 | 59.57% | 40.43% | No |
| 9114 | 0.545 | 91.21% | 45 | 53.33% | 46.67% | No |
| 9115 | 0.636 | 89.84% | 52 | 73.08% | 26.92% | No |
| 9116 | 0.727 | 91.21% | 45 | 55.56% | 44.44% | No |
| 9117 | 0.818 | 92.97% | 36 | 52.78% | 47.22% | No |
| 9118 | 0.909 | 92.58% | 38 | 60.53% | 39.47% | No |
| 9119 | 1.000 | 92.97% | 36 | 58.33% | 41.67% | No |

No candidate entered the success-rate window or reached 100 falls. The observed direction was also opposite the intended difficulty sweep: stronger command smoothing and weaker centering coincided with a higher success rate and lower lateral purity. This is a context-range calibration failure, not an algorithm-performance result.

没有任何候选进入目标成功率窗口或达到 100 次跌倒。实际趋势也与预期难度方向相反：更强的 command smoothing 和更弱的 centering 对应更高的成功率和更低的 lateral 纯度。这是 context 范围校准失败，不是算法性能结果。

## Evidence and next boundary / 证据与下一边界

- Exact compact progress: [`preflight/valid_revision0_L_dev_calibration_progress.json`](preflight/valid_revision0_L_dev_calibration_progress.json)
- Failure/provenance amendment: [`precalibration_calibration_failure_amendment.json`](precalibration_calibration_failure_amendment.json)
- External raw evidence remains sealed by file counts, byte counts, log hashes, and a deterministic manifest hash in the amendment.
- The replacement calibration may use only base-policy evidence, must use fresh candidate/evaluation randomness, and must be frozen and committed before execution. No adaptation result exists or may influence context design.

- 精确紧凑进度见上述 JSON；失败与 provenance 见 amendment。
- 外部原始证据由文件数、字节数、日志哈希和确定性 manifest 哈希封存。
- 后续范围修订只能使用基础策略证据，必须使用全新的 candidate/evaluation 随机数，并在执行前冻结和提交；目前不存在任何可影响 context 设计的适配结果。

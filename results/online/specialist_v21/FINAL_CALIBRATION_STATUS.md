# v21 final calibration status / v21 最终校准状态

The prospectively frozen replacement revision-0 calibration stopped cleanly at
`L2` on 2026-08-08 16:58:18 UTC, before any adaptation. It completed 9,216
base-policy episodes. `L_dev`, `C_dev`, and `L1` qualified and were frozen;
all 12 `L2` candidates failed the 100-fall gate, so `L3` through `C5` were not
started.

预先冻结的 replacement revision-0 校准于 2026-08-08 16:58:18 UTC 在 `L2`
按规则停止，且发生在任何适配之前。共完成 9,216 个基础策略 episode。`L_dev`、
`C_dev`、`L1` 已合格并冻结；`L2` 的 12 个候选全部未达到 100 次跌倒门槛，因此
`L3` 到 `C5` 均未启动。

Protocol commit: `da4c5a4b8655b743be951cec02eb77e2ea9f0414`

Protocol SHA-256: `177152643b52e4bc366f415cb40a854c3f6317644f836ca61884b875be376c39`

## Frozen contexts before the stop / 停止前已冻结的 context

| Context | Seed | Success | Falls | Target purity | Second mechanism |
| --- | ---: | ---: | ---: | ---: | ---: |
| L_dev | 17110 | 79.69% | 104 | 85.58% | 14.42% |
| C_dev | 17209 | 78.91% | 108 | 94.44% | 5.56% |
| L1 | 17308 | 79.49% | 105 | 87.62% | 12.38% |

## L2 negative result / L2 负结果

Each row contains 512 episodes. A candidate required 70–85% success, at least
100 falls, at least 80% lateral purity, and at most 30% for the second failure
mechanism.

| Seed | Success | Falls | Lateral purity | Second mechanism | Qualified |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 17408 | 85.55% | 74 | 81.08% | 18.92% | No |
| 17409 | 86.72% | 68 | 76.47% | 23.53% | No |
| 17410 | 85.55% | 74 | 70.27% | 29.73% | No |
| 17411 | 84.38% | 80 | 81.25% | 18.75% | No |
| 17412 | 85.35% | 75 | 78.67% | 21.33% | No |
| 17413 | 84.18% | 81 | 70.37% | 29.63% | No |
| 17414 | 87.50% | 64 | 81.25% | 18.75% | No |
| 17415 | 84.77% | 78 | 84.62% | 15.38% | No |
| 17416 | 83.79% | 83 | 73.49% | 26.51% | No |
| 17417 | 83.40% | 85 | 80.00% | 20.00% | No |
| 17418 | 83.98% | 82 | 64.63% | 35.37% | No |
| 17419 | 84.38% | 80 | 78.75% | 21.25% | No |

No candidate reached 100 falls; the observed range was 64–85. The formal
512-episode evidence therefore falsifies the pilot-3 small-sample inference:
isolated yaw-command disturbance is not a reliable enough `L2` failure
mechanism. This is a context-design failure, not an algorithm result.

没有候选达到 100 次跌倒，实际范围为 64–85。正式的 512-episode 证据因此推翻了
pilot-3 的小样本推断：隔离后的 yaw-command 扰动不能可靠地产生足够的 `L2` 失败。
这是 context 设计失败，不是算法结果。

Exact compact evidence is under [`calibration/`](calibration/), with the full
machine-readable provenance in
[`final_revision0_L2_failure_amendment.json`](calibration/final_revision0_L2_failure_amendment.json).
Raw simulator artifacts remain external and are bound by deterministic hashes.

# v79 25/75 mixed execution 的 Eq. (27) 单位平衡 reward

v79 复用 v72 的原始 base checkpoint、seed `201352619`、25% filter-on / 75%
filter-off execution、分组 advantage normalization、actor LR `1e-4` 和四轮
full-batch SGD。算法差异是采用 Eq. (27) foot-task proximity、0.1 velocity-margin
单位桥接及 next-riser foot-clearance reference。未启用事务回滚，以保留与 v72 相同
的多步优化轨迹。128 environments、4×1024 steps 在 RTX 4080 SUPER 上耗时
**134.51 秒**。

## 对齐结果

| Rollout | 实际 checkpoint | Filter off | Filter on | 总成功率 |
|---:|---:|---:|---:|---:|
| 1 | base | 121/198 (61.11%) | 47/69 (68.12%) | 168/267 (62.92%) |
| 2 | round 1 | 126/190 (66.32%) | **52/65 (80.00%)** | 178/255 (69.80%) |
| 3 | round 2 | 115/195 (58.97%) | 48/68 (70.59%) | 163/263 (61.98%) |
| 4 | round 3 | **139/193 (72.02%)** | 48/67 (71.64%) | **187/260 (71.92%)** |

best aligned round-3 actor 相对本次 base 的 filter-off 提高 **10.91 pp**，是 v79
明确的运行内改善，但仍低于 75% 门槛，故没有追加独立 deterministic gate 或上传
checkpoint。round-4 update 产生的 final actor 尚未对齐，不参与选择。

尽管 base、actor SHA 和 seed 与 v72 完全相同，v79 首轮 off 为 61.11%，而 v72
为 67.86%，说明当前 GPU simulator 跨进程并非 bitwise deterministic，不能把两个
run 的绝对 rate 差异完全归因于 reward。仍可比较优化形态：v72 最佳运行内提升
3.72 pp，v79 为 10.91 pp；单位平衡 reward 在 mixed deployment distribution 上
产生了更强的恢复，但四轮仍明显振荡。

manager-scaled CBF contribution 为 `-0.00288`–`-0.00270/step`，nominal reward
为 `+0.0181`–`+0.0205/step`；margin/proximity 分别约 `-0.0012`/`-0.0016`，
分量重构误差不超过 `1.67e-11`。这确认 reward 尺度和路由均按设计执行。

## 文件与溯源

- source commit：`d65f0b6a18d02b085338cf54af14079b8cd05851`
- base checkpoint SHA-256：
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`
- base actor SHA-256：
  `3964c0ef24707addbaa0dacfd4fd627882bd8b45a2c9799142710fa45bc29499`
- best aligned round-3 checkpoint SHA-256（未上传模型）：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- best aligned actor SHA-256：
  `b0a717cef34d128e4175226b86780e5210cc5287558d78cb5a44e095e37fb600`
- final unaligned checkpoint SHA-256：
  `5dac00ab9ae468ddc05f991a5794228a061ec9a1878f79792546127105fc542b`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/mixed_unit_balanced_v79_d65f0b6_s201352619`
- `training/` 保存完整 JSON/CSV；`decision_summary.json` 保存机器可读结论。

下一步从 best aligned round-3 checkpoint 用 `5e-5` full-batch step 和事务回滚做
一次短 continuation，目标是降低四轮 `~4e-5` KL 引起的振荡并跨过 75% gate。

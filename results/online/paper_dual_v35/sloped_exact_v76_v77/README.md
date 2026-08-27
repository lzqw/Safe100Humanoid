# v76–v77 Eq. (27) reduced-order reward 与同 seed continuation

v76 将 v75 的 12-D raw-action proximity（weight 10）改为论文 humanoid Eq. (27)
的 reduced-order swing-foot displacement（weight 1、`sigma=0.05`），并对 exact
candidate 启用论文描述的 next-riser foot-clearance reference。训练仍为 100%
safety-filtered execution、nominal-action PPO storage 和单次 full-batch SGD。
128 environments、4×1024 steps 在 RTX 4080 SUPER 上耗时 **91.00 秒**。

## v76 对齐结果

| Rollout | Filter on | moving KL | 决策 |
|---:|---:|---:|---|
| 1，base | 193/275 (70.18%) | `8.31e-6` | anchor |
| 2，round 1 | 193/273 (70.70%) | `9.91e-6` | accepted |
| 3，round 2 | 189/267 (70.79%) | `8.51e-6` | accepted |
| 4，round 3 | **196/265 (73.96%)** | `1.03e-5` | accepted / selected |

四轮表面上连续改善，未发生回滚，但 selected 仍低于 75% 门槛。round-4 update
产生的 actor 尚未对齐，因此 v77 从该 actor 出发、复用 seed `201352623` 做两轮短
continuation。v77 首轮为 **189/267（70.79%）**，相对 v76 同 seed 的起点
70.18% 仅 +0.60 pp；下一 proposal 为 192/279（68.82%）并被回滚。这证明 v76
末轮 73.96% 含有明显跨 rollout 方差，不能作为部署 gate 的依据。因此未运行独立
filter-off 评估，当前正式最佳不变。

## reward 数值分解

新增 telemetry 对每个 transition 精确分解 `r = r_nominal + r_margin +
r_proximity`。v76 四轮范围为：

| 分量 | Mean / transition |
|---|---:|
| nominal task + regularization | `+0.766` 到 `+0.854` |
| Eq. (22) barrier margin | **`-0.769` 到 `-0.680`** |
| Eq. (23)/(27) foot proximity | `-0.081` 到 `-0.080` |
| CBF reward total | `-0.850` 到 `-0.760` |

分量重构最大误差仅 `2.18e-9`。barrier active fraction 为 27.15%–27.40%，active
foot correction norm 为 0.0435–0.0494。结果推翻了“proximity 权重主导”的假设：
即使 margin weight 为 1，未归一化 margin 仍占 CBF penalty 的约 90%，几乎逐步
抵消整个 nominal reward。下一步必须检查论文 joint-velocity `v` 与当前 joint-position
setpoint/action Jacobian 的单位映射，而不是继续搜索 proximity 权重。

## 文件与溯源

- source commit：`93935b8cc4424c5099a66999028d7975aaf460f0`
- v76 base checkpoint SHA-256：
  `3285223174b01c97009db54361042c4c3d2d87054ca2156c84769f6d13ceccbc`
- v76 selected aligned round-3 checkpoint SHA-256（未上传模型）：
  `66374c1114901a1f97592750be004bb4f84dede6d43ca8101d2b0aedf903df28`
- v76 selected actor SHA-256：
  `02296e8b1d868d4dde5f93e0a82c4a90501c7add69d9fe7b979a697b7ee121e7`
- v76 final unaligned checkpoint SHA-256：
  `bd0c7e65ae0e13990b854f38e8b619105af82823c7c20405584451543cdfa8c1`
- v76 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/sloped_exact_v76_93935b8_s201352623`
- v77 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/sloped_exact_cont_v77_93935b8_s201352623`
- `v76_training/` 与 `v77_continuation/` 保存完整 JSON/CSV；
  `decision_summary.json` 保存机器可读结论。

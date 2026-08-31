# v63--v64 论文式全过滤 Eq. (27) continuation

这两次实验从 v60 的目标高度对齐 checkpoint 继续训练，检验 v62 的过滤退火
回落是否可以通过论文原始做法修复：训练全程执行 CBF，策略通过 CBF dual reward
内化安全动作，部署候选才关闭过滤。两次实验均固定在 18.4 cm F2，使用 256 个
并行环境、6×1024 steps（各 1,572,864 transitions），并在每轮刷新 25% 强度的
friction、base COM、encoder bias 和 actor observation noise。

## 对齐训练轨迹

| 轮次（评估前一 checkpoint） | v63 水平 Eq. (27) CBF | v64 斜坡 task-compatible CBF |
|---:|---:|---:|
| 1（base） | 290/551 (52.63%) | **370/554 (66.79%)** |
| 2（round 1） | **303/551 (54.99%)** | 353/566 (62.37%) |
| 3（round 2） | 289/559 (51.70%) | 360/563 (63.94%) |
| 4（round 3） | 293/547 (53.56%) | 356/565 (63.01%) |
| 5（round 4） | 300/552 (54.35%) | 365/560 (65.18%) |
| 6（round 5） | 303/572 (52.97%) | 355/550 (64.55%) |

v63 使用论文的水平下一台阶 hyperplane、persistent next-riser clearance 和
reduced-order swing-foot dual reward。它在新 DR 上的 filtered 成功率只有
51.70%--54.99%，表明该水平几何不足以保护当前 18.4 cm 高台阶 continuation。

v64 保留同一 CBF-RL dual reward 与脚部 reference，但恢复可提前抬脚的斜坡
x-z clearance。其起点 filtered 成功率为 66.79%，显著高于 v63 的起点；由于
两次实验还使用不同 seed，且 v64 恢复了 moving KL，这只作为方向性证据，不作
严格配对因果结论。更关键的是，v64 所有 post-update 对齐 rollout 都低于自己的
起点，说明强 Eq. (27) reward 的普通 PPO 更新仍损伤任务策略。

## 决定

v63 和 v64 均拒绝。没有新 checkpoint 超过各自训练起点，因此未运行额外
filter-off gate，也没有上传劣化 checkpoint。v60 对齐 checkpoint 继续保留。

下一步不再重复 reward 权重或几何组合，而是把任务 PPO 与安全内化梯度解耦：
先保护成功任务轨迹，再只在 CBF 介入状态上吸收安全动作，避免强稀疏 safety
advantage 改写整条 locomotion policy。

## 溯源

- v63 source commit：`4675ede990e7f7e58287ae3dfaa2cfac6ae2a418`
- v64 source commit：`61b98afa3ee56fd955a07c46fb0e9e94eeff220c`
- 共同 base checkpoint SHA-256：
  `f00e3a56276f629504234a20b40c124ee43a2f4d145cb143b3b2899acc024b27`
- v63 训练时间：164.57 s；v64：174.55 s。
- 每个子目录包含完整 `training_summary.json` 和
  `round_metrics.{json,csv}`。

实现入口：`experiments/scripts/refine_paper_dual_v35.py`；reward 配置：
`src/tasks/stairs_cbf/paper_dual_v35.py`。

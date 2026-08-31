# 长时爬楼横向/航向闭环纠偏实现与测试

更新日期：2026-07-17

## 结论

用户观察到的失败模式得到定量确认：在关闭运行时 CBF、使用开放环遥控指令时，DQN 场景的 271 次摔倒中有 262 次曾发生机器人根部或脚越过楼梯侧边界，占 96.7%。主要问题不是 CBF 被移除本身，而是长时累计的横向和航向漂移没有得到类似人类遥控者的闭环纠正。

本次增加了一个“合成闭环操作者”：它持续观察机器人相对楼梯中心线的横向偏差和前方 1 m 中心点的航向偏差，然后只通过原有遥控命令通道输出有界的 `vy` 和 `wz`。策略 observation 结构、动作维数、checkpoint 和底层关节策略均未修改，运行时 CBF 保持关闭。

在 RTX 4080 SUPER 上，使用相同 checkpoint、seed=42、64 个并行环境、每组 512 回合进行配对测试后：

| 场景 | 指令方式 | 成功率 | 摔倒率 | 侧向越界率 | 侧向摔倒率 | 平均绝对中心线误差 |
|---|---|---:|---:|---:|---:|---:|
| DQN | 开放环随机脉冲 | 46.88% | 52.93% | 56.45% | 51.17% | 0.402 m |
| DQNH | 闭环模拟遥控纠偏 | **91.60%** | **8.40%** | **6.45%** | **5.66%** | **0.200 m** |
| DQ | 开放环随机脉冲 | 46.88% | 53.12% | 55.08% | 50.78% | 0.395 m |
| DQH | 闭环模拟遥控纠偏 | **88.28%** | **11.72%** | **10.55%** | **9.77%** | **0.209 m** |

DQN→DQNH 的成功率提高 44.73 个百分点，摔倒率降低 44.53 个百分点，侧向摔倒率降低 45.51 个百分点，平均中心线误差降低 50.17%。DQ→DQH 的成功率提高 41.41 个百分点，侧向摔倒率降低 41.02 个百分点。

这是一组固定 seed 的大样本配对仿真结果，不是多 seed 统计，也不是实际人类遥控实验或真机结果。

## 闭环操作者算法

每个控制步从地形的 `stair_targets` flat patches 读取楼梯中心线。设机器人世界坐标横向位置为 `y_robot`，楼梯中心为 `y_center`：

```text
e_center = y_robot - y_center
lookahead_world = [1.0 m, y_center - y_robot, 0]
lookahead_body = R_world_to_body * lookahead_world
e_heading = atan2(lookahead_body.y, lookahead_body.x)
```

对误差施加连续的带死区比例控制：

```text
deadband(x, d) = sign(x) * max(|x| - d, 0)
vy = clip(0.80 * deadband(lookahead_body.y, 0.04), -0.16, 0.16)
wz = clip(1.40 * deadband(e_heading, 0.03), -0.45, 0.45)
```

`vx` 沿用原始前进遥控指令。整个命令仍经过原有 40–160 ms 指令延迟和 0.08 s 低通滤波，因此不是理想、无延迟的直接状态反馈。

关键设计边界：

- 中心线和航向误差不直接送入 actor；actor 只看到最终的 `[vx, vy, wz]`，与真机接收遥控器速度指令的接口一致。
- 未修改 405 维 actor observation、12 维策略动作、29-DoF 机器人状态、步态相位或 checkpoint。
- 未打开关节层 CBF action filter。视频与 512 回合评估中的 `runtime_filter=false`，CBF 干预积分为 0。
- DQH/DQNH/DQMH 分别继承 DQ/DQN/DQM 的完全相同地形几何，仅将指令源切换为闭环纠偏。

## 实现位置

- `src/tasks/stairs_cbf/teleop_math.py`：带死区、限幅的纯 Torch 纠偏函数。
- `src/tasks/stairs_cbf/command.py`：GPU 批量中心线/航向反馈、命令延迟、诊断指标。
- `src/tasks/stairs_cbf/config.py`：DQH、DQNH、DQMH 配置别名。
- `src/tasks/stairs_cbf/__init__.py`：新任务注册。
- `experiments/scripts/evaluate_online_stairs.py`：逐回合中心线误差、边缘余量、侧向越界和纠偏占比。
- `experiments/scripts/render_stairs_video.py`：视频同时输出上述安全指标。
- `experiments/tests/test_online_refinement.py`：反馈符号、死区、限幅、配置继承和任务注册测试。

任务名：

- `Unitree-G1-Stairs-Online-DQH`
- `Unitree-G1-Stairs-Online-DQNH`
- `Unitree-G1-Stairs-Online-DQMH`

## 测试与证据

服务器最终单元测试结果：`19 passed in 10.46s`。

日志：

- `/home/carla/LZQW/SAFE100/humanoid/logs/pytest_human_centering_final_all.log`
- `/home/carla/LZQW/SAFE100/humanoid/logs/eval_human_centering_DQN_off_512.log`
- `/home/carla/LZQW/SAFE100/humanoid/logs/eval_human_centering_DQNH_off_512.log`
- `/home/carla/LZQW/SAFE100/humanoid/logs/eval_human_centering_DQ_off_512.log`
- `/home/carla/LZQW/SAFE100/humanoid/logs/eval_human_centering_DQH_off_512.log`
- `/home/carla/LZQW/SAFE100/humanoid/logs/render_human_centering_DQNH_seed42.log`

机器可读结果位于 `results/online/evaluation/human_centering_v1/`。每组都包含 512 回合逐条 CSV 和汇总 JSON，未删除失败 episode。

视频回合使用 DQNH、seed=42、同一 checkpoint、运行时 CBF 关闭：554 步到顶，未摔倒，未发生侧向越界；平均中心线误差 0.133 m，最大中心线误差 0.440 m，最小脚部边缘余量 0.674 m。

视频：`results/online/videos/human_centering_v1/g1-stairs-online_dqnh-filter-off-seed42-step-0.mp4`

视频 SHA256：`3343e3ec48bac529c45dbe88fabc3f103990634aad8e15191e8c897fdc25bd80`

checkpoint：`results/online/checkpoints/accepted_no_runtime_cbf_round_002.pt`

checkpoint SHA256：`cce88b7403f7a8d1e979ad8c2b0eaa83e16a03282b218cd5cc03aee123490ac2`

## 仍然存在的限制

- 纠偏激活占比约 99%，说明 4 cm/0.03 rad 死区对于当前扰动较小；这是持续视觉辅助式操作者，不代表真实人的操作频率。
- DQNH 仍有 8.40% 摔倒，剩余摔倒中 29/43 伴随侧向越界。后续应把楼梯边缘余量或中心线误差加入策略训练分布，而不是无限提高遥控器反馈增益。
- 最大航向误差会包含 reset、瞬态和登顶附近转动，DQN→DQNH 仅从 1.791 rad 降到 1.344 rad；中心线误差、侧向越界率和完成率是本次更直接的指标。
- `minimum_foot_edge_clearance` 是 512 回合的单次最小值，容易被一个极端 episode 主导，不能代替侧向越界率。
- 本结果只证明“闭环高层遥控纠偏 + 无运行时 CBF”的仿真有效性，不证明真机部署安全。

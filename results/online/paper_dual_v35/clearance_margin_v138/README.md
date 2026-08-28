# v138 Paper Next-Riser Clearance Margin

论文在 stair CBF Eq. (27) 后明确说明，foot-clearance reference 应随机器人前方台阶高度
变化。现有 paper-dual 路径已经使用 persistent next-riser reference，但只保留 tread 上方
5 cm 的固定裕量。v132 两个部署 seed 的 128 个 episode 中，失败轨迹平均 unsafe overlap
为 20.97 steps，成功轨迹为 6.54 steps；v138 因此从最强 v132 selected actor 继续，将
唯一 reward 参数 `foot_clearance.height_above_tread` 从 0.05 m 提到 0.08 m。

CBF 几何、actor/critic、PPO、100% filtered execution、Eq. (27) dual reward、训练 std、
学习率控制和固定 F2 18.4 cm 环境均保持不变。论文来源：
[CBF-RL: Safety Filtering Reinforcement Learning in Training](https://arxiv.org/html/2510.14959v6)。

## 训练结果

- 128 environments × 1,024 steps × 4 rounds = 524,288 transitions。
- RTX 4080 SUPER 用时 137.99 秒（2 分 17.99 秒）。
- 最佳 aligned filter-on rollout 使用 round-2 checkpoint：
  **201/271 = 74.17%**，mean reached riser 8.0111。

| Rollout | Checkpoint | Filter on | Mean riser | Moving KL | Selected |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0 | 189/276 (68.48%) | 7.7790 | 0.000703 | yes |
| 2 | 1 | 196/275 (71.27%) | 8.0473 | 0.000402 | yes |
| 3 | 2 | **201/271 (74.17%)** | 8.0111 | 0.000436 | **yes** |
| 4 | 3 | 196/267 (73.41%) | 8.1348 | 0.000435 | no |

## 部署结果与判定

唯一 deterministic filter-off development screen（seed `201355980`）：

- **46/64 = 71.875%**，未达到预声明 `48/64` 门槛。
- 18/64 跌倒，mean reached riser 7.9219。
- unsafe overlap 为 0.8363 steps/riser；描述性低于 v132 screen 的 1.0849 和 gate 的
  1.3969，但三个 seed 不同，不能作为严格 paired 改善结论。
- 因 screen 未通过，没有运行独立 gate、其他 seed、其他 checkpoint 或补充评测。

结论：更高的论文式 next-riser clearance reference 让训练内 filtered success 连续升到
74.17%，并降低了本次 deployment 的 overlap 指标，但 deterministic filter-off 成功率
仍差 2 个 episode。v138 拒绝；v132 仍是最强 development candidate，正式最佳仍为
v79 aligned filter-off `139/193 = 72.02%`。

## 溯源

- 实现提交：`b1d2d9c7ab76cbce3ce85343068e234e568a1ad7`。
- 唯一针对性测试：
  `test_v138_changes_only_paper_next_riser_clearance_margin`，
  4080 上 `1 passed in 17.77s`；未运行完整测试套件。
- base v132 checkpoint SHA-256：
  `a7fdd4d07dc79f1f001b09ff3638bcc5de000c3804f369dcf13ed61ebd18bde3`。
- selected round-2 checkpoint SHA-256：
  `7a3899c515d5afd93f79f4db251feab4cd59f003e7150711e506ef5850604c63`。
- selected actor SHA-256：
  `0efe0cf603f958d6874efa924f62999fe7fb640999d89180d10c30cca8abfeb9`。
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/clearance_margin_v138_b1d2d9c_128x1024x4_s201355900`。
- 全部训练和 screen JSON/CSV 已提交；模型未通过 screen，所以二进制未提交。

实现入口：`src/tasks/stairs_cbf/paper_clearance_margin_v138.py` 与
`experiments/scripts/refine_paper_dual_v35.py`。

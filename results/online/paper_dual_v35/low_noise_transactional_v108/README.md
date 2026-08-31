# v108 low-noise native CBF-dual PPO

v107 证明直接 episode outcome credit 的 proposal 持续回落。v108 因此回到论文原生
GAE + Eq. (27) CBF dual reward，从全局最佳 v79 重新做 405→415 exact-zero geometry
expansion，并保留 v105 geometry gradient balance 与逐 proposal transactional rollback。
唯一训练假设是：近收敛 policy 的固定 action std `0.05` 造成同一 actor 约 10 pp 的
rollout 波动，降低 std 可让梯度更接近 deterministic filter-off deployment。

## std=0.02 数值诊断

首个尝试使用 std `0.02` / LR `2e-5`。前两轮 filter-off 为 127/196=64.80% 和
124/193=64.25%，首个 proposal 已回滚。第三轮在更新前的一致性证明中停止：低 std
放大 float32 Gaussian log-prob reduction 误差，最大差 `0.0010242462`，刚超过冻结的
`0.001` 阈值；distribution parameter 本身仍保持一致。这不是 OOM 或训练发散。

没有放宽数据流校验，而是改用仍比历史 std 低 40%、且有明确数值余量的 std `0.03`。

## std=0.03 正式结果

正式运行使用同一新 seed `201353362`、F2 18.4 cm、25% filter-on / 75%
filter-off、128 environments、1024 steps、8 轮，初始 LR `3e-5`。共处理
1,048,576 transitions，在 RTX 4080 SUPER 上耗时 **211.34 秒（3 分 31 秒）**。

| Rollout | actor | Filter off | Filter on | LR | 决策 |
|---:|---|---:|---:|---:|---|
| 1 | v79 expanded anchor | 134/194 (69.07%) | 46/70 (65.71%) | `3.00e-5` | 建立 anchor |
| 2 | proposal 1 | **130/192 (67.71%)** | 48/69 (69.57%) | `2.59e-5` | rejected / rollback |
| 3 | anchor retry | 131/201 (65.17%) | 43/69 (62.32%) | `1.29e-5` | pooled anchor；产生 proposal 2 |
| 4 | proposal 2 | 116/200 (58.00%) | 42/64 (65.62%) | `1.29e-5` | rejected / rollback |
| 5 | anchor retry | 128/194 (65.98%) | 48/69 (69.57%) | `6.46e-6` | pooled anchor；产生 proposal 3 |
| 6 | proposal 3 | 126/189 (66.67%) | 53/70 (75.71%) | `6.46e-6` | rejected / rollback |
| 7 | anchor retry | **137/196 (69.90%)** | 46/67 (68.66%) | `3.23e-6` | pooled anchor；产生 proposal 4 |
| 8 | proposal 4 | 128/191 (67.02%) | 45/67 (67.16%) | `3.23e-6` | rejected / rollback |

四个 proposal 全部回落，训练结束时 actor 与输入 anchor 相同。保留 anchor 四次
filter-off 合并为 **530/785 = 67.52%**；最高 69.90% 仍只是同一 actor 重测。
没有任何训练信号达到 75%，因此不运行独立 gate，全局选择继续为 v79。

## 诊断与结论

- std=0.03 的 behavior log-prob 最大重算误差为 `0.000555`，完整运行始终低于冻结阈值。
- moving KL 从 `1.35e-5` 随 LR 自动缩小到 `4.69e-8`；即使 proposal 已极小，仍无
  一次 filter-off 改善。
- geometry 梯度倍率为 8.93×–13.80×，每轮缩放后 geometry/legacy ratio 为 1.0。
- std=0.02 没有抬高起点；std=0.03 的四个 proposal 也全部回落，因此 rollout
  exploration 噪声不是现有 paper-dual PPO 无法累积改善的充分解释。

结合 v87 的跨 seed PPO gradient cosine 0.017、v107 的 outcome proposal 全回落，
当前证据已经同时排除了“大 batch 不够”“geometry 梯度太弱”“episode 权重太大”
和“action std 太高”。下一步不应继续调整同一 PPO 标量；需要利用 matched filter
rescue 的成对反事实轨迹，学习在安全 filter 真正改变最终 outcome 的状态依赖方向。

## 文件与溯源

- source commit：`f5093c4310551c66ea214798f6259eeec4aaa82a`
- base v79 checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- retained expanded actor SHA-256：
  `9d4efc6f67b4098b8cf9742aaa38d5744918781017c6f46266745881aa1d5b8e`
- selected `round_00.pt` SHA-256：
  `850b88116c191b094242cc6fd8f97acfdc68a34eb9947dbd28fd5ff534017df7`
- final restored `round_08.pt` SHA-256：
  `7994f59ca757452bb8b2bfde26fe74769c70256c21c1556a5ed8fa7b4e24972f`
- std=0.02 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/low_noise_transactional_v108_f5093c4_128e_s201353362`
- std=0.03 正式目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/low_noise_transactional_v108b_f5093c4_128e_s201353362`

未通过 gate 的模型二进制未上传 GitHub；正式与诊断逐轮 JSON/CSV、checkpoint 哈希和
选择依据均保存在本目录。

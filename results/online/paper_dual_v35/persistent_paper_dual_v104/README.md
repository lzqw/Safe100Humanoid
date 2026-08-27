# v104 persistent geometry 完整 paper-dual PPO

v104 将 v94/v95 已验证可部署的 bilateral next-riser persistent geometry 从
10-D adapter 输入升级为完整 paper-dual filtered-execution PPO 的 actor 输入。actor
由 v79 最佳 checkpoint 从 405-D 零列扩展到 415-D，critic 从 838-D 零列扩展到
848-D；旧列逐位复制、新列严格为零，因此初始 nominal policy 与 v79 完全一致。
与 v95 只更新 5,120 个新增首层权重不同，v104 的 378,764 个 actor MLP 参数全部
可训练。

训练使用 F2 18.4 cm、25% filter-on / 75% filter-off mixed execution、分组 advantage
normalization、Eq. (27) unit-balanced reward、128 environments、1024 steps、8 轮
full-batch SGD，actor LR 为 `1e-4`。共处理 1,048,576 transitions，在 RTX 4080
SUPER 上耗时 **193.66 秒**。

## 对齐结果

每个 rollout 使用更新前的 checkpoint；因此 rollout 6 对应 `round_05.pt`。

| Rollout | checkpoint | Filter off | Filter on | moving KL |
|---:|---:|---:|---:|---:|
| 1 | round 0 | 129/197 (65.48%) | 48/66 (72.73%) | 4.73e-5 |
| 2 | round 1 | 122/191 (63.87%) | 39/71 (54.93%) | 4.26e-5 |
| 3 | round 2 | 128/199 (64.32%) | 51/66 (77.27%) | 3.14e-5 |
| 4 | round 3 | 118/201 (58.71%) | 50/67 (74.63%) | 5.34e-5 |
| 5 | round 4 | 126/201 (62.69%) | 40/68 (58.82%) | 3.93e-5 |
| 6 | round 5 | **132/200 (66.00%)** | 50/69 (72.46%) | 3.87e-5 |
| 7 | round 6 | 125/197 (63.45%) | 48/70 (68.57%) | 4.44e-5 |
| 8 | round 7 | 128/204 (62.75%) | 50/67 (74.63%) | 3.22e-5 |

最佳更新后 actor 仅比本次零列基线高 **0.52 percentage points**，且低于 v79
历史最佳 72.02%，更未达到 75% gate。按既定协议没有运行独立 gate，v104 被拒绝，
全局选择继续保留 v79。

## 诊断与结论

warm-start 记录证明 actor 405→415 和 critic 838→848 都是 exact prefix expansion；
action routing 的逐轮最大误差均为 0，8/8 optimizer step 正常完成。对最佳
`round_05.pt` 与零列 `round_00.pt` 比较：

- 新增 geometry 首层权重最大绝对值：`2.2341e-7`
- legacy 首层权重最大变化：`5.6624e-7`
- 更深 MLP 层最大变化：`8.8811e-6`

因此完整 PPO 虽允许 geometry 学习，实际梯度仍主要流入共享深层网络，新增几何列
几乎未被利用。v95 已表明“只训练新增列”也不能泛化；下一步不能简单重复任一路径，
而应在保留完整 actor 更新的同时，对 persistent geometry 使用显式的相对梯度尺度或
与 episode outcome 对齐的 auxiliary objective，并继续用唯一 75% filter-off gate 判定。

## 文件与溯源

- source commit：`a09ffe4a6e9d118337021af8b9230179e95c6ba1`
- base checkpoint：v79 `round_03.pt`
- base checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- best aligned checkpoint：`round_05.pt`
- best aligned checkpoint SHA-256：
  `20d9ae29cd82c5ab59c57409d6fc6a0451660a61ddb3d074aba0b117edcfcf20`
- best aligned actor SHA-256：
  `ef1a7e3008f568eef6b02f88983febce6903dc38c928f68632de80566cd10ada`
- final unaligned checkpoint SHA-256：
  `309ca5830b35f511048e357dda197e287a342f1e49a111dd932582b870757290`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/persistent_paper_dual_v104_a09ffe4_s201353360`

模型二进制未上传 GitHub；`training/` 保存完整 JSON/CSV，checkpoint 哈希和选择依据
保存在本目录。

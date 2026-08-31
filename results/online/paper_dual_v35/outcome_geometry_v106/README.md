# v106 outcome-centered episode advantage

v106 将 v105 的 415-D persistent geometry、完整 actor PPO 和 geometry/legacy
首层梯度 1:1 平衡，与 episode outcome credit 结合。每个 filter group 内，完整成功
episode 平分 `+0.5` 总质量，失败 episode 平分 `-0.5` 总质量，再按 episode 长度分摊；
未结束 episode 不产生 outcome credit。该 credit 与组内标准化 GAE 以固定权重 1.0
相加并再次标准化，直接强化登顶轨迹、抑制跌倒轨迹，不做成功动作克隆。

正式运行复用 v79 最佳 checkpoint，配置为 F2 18.4 cm、25% filter-on / 75%
filter-off mixed execution、64 environments、1024 steps、8 轮 full-batch SGD，
actor LR `7.5e-5`。共处理 524,288 transitions，在 RTX 4080 SUPER 上耗时
**269.42 秒（4 分 29 秒）**。

最初的 128-environment 启动在首轮更新时因并发 GPU 显存占用而 OOM；没有从该次
不完整运行选择 checkpoint。为避免干扰其他任务，算法不变并立即降至 64 environments
完成上述正式运行。

## 对齐训练结果

| Rollout | checkpoint | Filter off | Filter on | outcome labeled | GAE/credit cosine | geometry scale |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | round 0 | 68/98 (69.39%) | 30/33 (90.91%) | 81.35% | 0.227 | 10.03× |
| 2 | round 1 | 63/92 (68.48%) | 27/33 (81.82%) | 80.31% | 0.249 | 9.49× |
| 3 | round 2 | 59/94 (62.77%) | 20/33 (60.61%) | 79.79% | 0.255 | 6.96× |
| 4 | round 3 | **68/95 (71.58%)** | 23/34 (67.65%) | 80.45% | 0.264 | 9.45× |
| 5 | round 4 | 67/100 (67.00%) | 25/34 (73.53%) | 81.28% | 0.306 | 10.05× |
| 6 | round 5 | 64/95 (67.37%) | 23/36 (63.89%) | 80.24% | 0.281 | 8.98× |
| 7 | round 6 | 63/93 (67.74%) | 25/32 (78.12%) | 78.34% | 0.243 | 11.22× |
| 8 | round 7 | 66/102 (64.71%) | 24/33 (72.73%) | 80.89% | 0.289 | 8.22× |

最佳更新后 checkpoint 为 `round_03.pt`，相对本次零列起点 69.39% 提高
**2.19 percentage points**。但它仍低于 v79 的 72.02%，也低于 75% gate；因此
没有运行独立 gate，v106 被拒绝，全局选择继续为 v79。

## 信号诊断与结论

- outcome credit 平均覆盖 80.33% transitions；其余为 rollout 末尾尚未结束的 episode。
- GAE/outcome-credit cosine 为 0.227–0.306，平均 0.264，说明 episode outcome 提供了
  与局部 GAE 部分一致、但并不相同的训练方向。
- geometry 自适应倍率为 6.96×–11.22×，缩放后的 geometry/legacy gradient ratio
  每轮均为 1.0；moving KL 保持在 `1.61e-5`–`2.61e-5`。
- 单轮峰值确实高于本次起点，但后续回落到 64.71%，固定 1.0 权重的 outcome credit
  没有形成稳定、可累积的 75% 以上改进。后续应对 episode signal 做保守的
  acceptance/weight control，而不是继续增加相同权重或盲目延长训练。

## 文件与溯源

- outcome 实现 commit：`4820362`
- 正式运行 source commit：`58ad05d181f46fdc3d33fb43fd5154ad394802d8`
- base checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- best aligned checkpoint：`round_03.pt`
- best aligned checkpoint SHA-256：
  `4b1e558fb21eea162d0585263f7af018b34d6996877b9788ebcf1983d52c274c`
- best aligned actor SHA-256：
  `9053bbf42079f397a83d73c6d6a9f5b508021ebcf74f4f441ce5b964939ff935`
- final unaligned checkpoint SHA-256：
  `46e102535cfa8c4b40d0aef2897f7f831a0f1166352a0e174b93b6b07e34e897`
- 4080 原始目录：
  `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/outcome_geometry_v106_58ad05d_64e_s201353360`

未通过 gate 的模型二进制未上传 GitHub；完整训练 JSON/CSV、checkpoint 哈希和选择
依据均保存在本目录。

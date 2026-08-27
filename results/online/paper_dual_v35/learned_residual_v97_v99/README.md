# v97–v99 learned CBF residual policy

这组实验冻结 v79 主策略，增加一个 28,364 参数的可部署 residual head。输入为
v79 的 128-D hidden state、10-D persistent next-riser geometry 和 12-D nominal
action。训练时执行 CBF-filtered action，residual head 直接学习论文 Eq. (23) 对应的
`safe - nominal` 方向；部署 gate 完全关闭 runtime filter。

## v97：全部 CBF intervention DAgger

四轮各使用 128 个 filter-on 首回合。teacher distance 持续下降，但 residual 越强，
filter-on 成功率越低：

| 轮次 | Filter-on | CBF teacher transitions | Teacher distance after |
|---:|---:|---:|---:|
| 1 | 95/128 (74.22%) | 5,115 | 0.16839 |
| 2 | 89/128 (69.53%) | 2,926 | 0.15134 |
| 3 | 85/128 (66.41%) | 3,061 | 0.13597 |
| 4 | 78/128 (60.94%) | 3,177 | 0.12764 |

训练与唯一 filter-off gate 总耗时 84.38 秒。最终 gate（seed `201353050`）为
**45/64 = 70.31%**，因此 v97 拒绝。

## v98：只校准 residual 幅度

不重新训练网络，在同一个 256-env screen 中并行比较四个 scale，每档 64 回合：

| Scale | Screen filter-off |
|---:|---:|
| 0 | 43/64 (67.19%) |
| 0.05 | **47/64 (73.44%)** |
| 0.10 | 43/64 (67.19%) |
| 0.20 | 46/64 (71.88%) |

选择 `0.05x` 后，唯一新 seed `201353080` gate 为
**46/64 = 71.88%**，仍低于 75%，因此 v98 拒绝。校准和 gate 共 38.96 秒。

## v99：只学习最终成功的 filtered episode

v99 排除所有失败 episode 的 teacher transition，共保留 9,270 个成功轨迹 CBF
transition。尽管离线 teacher distance 降至 0.10785，filter-on 成功率仍从第一轮
96/128 下降到第四轮 80/128。最终 scale screen 的结果为：

| Scale | Screen filter-off |
|---:|---:|
| 0 | **47/64 (73.44%)** |
| 0.025 | 46/64 (71.88%) |
| 0.05 | 39/64 (60.94%) |
| 0.10 | 43/64 (67.19%) |

选择规则最终选择 `0x`，即完全拒绝 learned residual。随后唯一 gate（seed
`201353180`）只是 v79 本体，为 43/64 = 67.19%。v99 训练 69.79 秒，screen+gate
54.03 秒。

## 结论

学习全部 intervention、只学习成功 episode、以及部署幅度校准都已完成。三组证据
一致表明：当前 instantaneous CBF correction 虽可被网络准确拟合，但它本身并非稳定的
task-success direction。问题已定位到现有 CBF 的 task compatibility / filter-on ceiling，
而不是 residual 网络容量不足。全局选择仍为 v79，不上传这些被拒绝的模型二进制。

## 溯源

- v97 source commit: `3631e4edc5d8823d706c84ea480bba1410d56848`
- v98 source commit: `ec51df1b7629a53a01c1c1e604d5b264b30c1572`
- v99 source commit: `e71728f2ec397e9cb7d834f7e0c9ff63fc8cb2f1`
- targeted tests: 2 passed
- input v79 SHA-256: `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`

精确候选哈希和 4080 路径在 [`checkpoint_index.json`](checkpoint_index.json)，
机器可读指标在 [`decision_summary.json`](decision_summary.json)。

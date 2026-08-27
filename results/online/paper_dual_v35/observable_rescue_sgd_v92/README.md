# v92 matched-rescue geometry adapter + full-batch SGD

v92 针对 405-D blind actor 看不到 runtime CBF 所用摆动脚/下一台阶几何的问题，追加
5 个可部署状态量，并把旧 actor 的新增输入列严格初始化为零。原 405-D 参数全部冻结，
只训练 2,560 个新输入权重；训练标签只来自同一初始状态下“filter-off 失败、filter-on
成功”的 matched-rescue episode，target 使用论文式 `25%` bounded correction。

4080 被外部任务占用，因此本轮在本机 GTX 1660 SUPER 上收集 4 seed × 64 个 paired
initial states。最终数据包含 **54** 个 matched rescue episode、**2,001** 个有效
teacher transition 和 101,556 个总 transition。

## Adam 诊断与 SGD 修复

同样的数据规模下，历史 Adam 路径把 action-correction cosine 推到 `-0.0271`，离线
gate 失败，因此未运行部署评估。v92 改成一次 full-batch SGD，使梯度幅值与方向都得到
保留，再通过参数线搜索把 active-state forward KL 限制到 `5e-5`。

| 指标 | 更新前 | v92 SGD 后 |
|---|---:|---:|
| teacher correction cosine | 0 | **0.5912** |
| teacher weighted distance | 0.136823 | **0.136479** |
| active-state forward KL | 0 | **4.9966e-5** |
| optimizer updates | — | **1** |

旧 405-D 权重 change max-abs 为严格 `0`，geometry inactive 时 candidate 与 base policy
保持完全一致。离线 gate 通过。

## 唯一 untouched filter-off gate

按预先约定只在全新 seed `201352750` 上运行一次 64-episode deterministic filter-off
gate。结果为 **47/64（73.44%）**，距离 75% 门槛只差 1 个 episode。由于未达到绝对
门槛，本轮立即停止，没有追加 filter-on、base 重测或更多 seed；不能宣称达到论文效果，
全局正式最佳仍是 v79。

## 溯源与 checkpoint 索引

- source commit：`5914b46decd3d1d93ffaa6bd049cc382ed7efe32`
- targeted tests：6 passed
- input v79 checkpoint SHA-256：`9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- expanded zero-adapter checkpoint SHA-256：`536d0a1db57a3a7067cd2ca24378717a53f21d0b2fba70a2cea40a941e8ce099`
- candidate actor SHA-256：`fc97811259ce869412371a77462183cb3663d1d84679b4d8e318c44574af27aa`
- candidate checkpoint SHA-256：`3c855c5b5e81cb221187d10db227595abbf7e765d7273d2a2b92515b697cfcce`
- candidate 本机路径：`/home/lzqw/PycharmProject/safe100/HUMANOID/artifacts/paper_dual_v35/observable_rescue_sgd_v92_5914b46_4x64/candidate.pt`
- training wall time：`228.67 s`

原始 Adam 方向失败记录位于 [`adam_direction_failure/`](adam_direction_failure/)，SGD
训练摘要位于 [`sgd_training/`](sgd_training/)，逐 episode gate 位于 [`gate/`](gate/)。
由于未过 75% gate，checkpoint 二进制不提交 Git，完整哈希和本机路径已保存。

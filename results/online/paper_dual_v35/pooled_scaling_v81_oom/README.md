# v81 pooled-scaling 显存诊断

v81 计划从 v79 best checkpoint 出发，沿用 Eq. (27) unit-balanced reward、25/75 mixed
execution、pooled transactional anchor 和 `5e-5` actor LR，将 environments 从 128 增加到
256，以降低每轮成功率估计和梯度的方差。训练机器为 RTX 4080 SUPER（16 GB）。

结果：**256 和 192 environments 均在第一轮 actor full-batch forward 失败**。两次均未执行
optimizer step，未产生 proposal checkpoint、`training_summary.json` 或可用于 gate 的训练结果。
输出目录内的 `round_00.pt` 只是启动时重存的输入 actor，不是新候选。

| 尝试 | 失败阶段 | 申请显存 | 报错时可用显存 | 训练结果 |
|---:|---|---:|---:|---|
| 256 env | round 1 actor full-batch forward | 406.00 MiB | 504.94 MiB | 无，0 optimizer step |
| 192 env | round 1 actor full-batch forward | 630.00 MiB | 944.75 MiB | 无，0 optimizer step |

失败位置为 `src/tasks/stairs_cbf/teacher_v30.py:485` 的
`self.actor(batch_observations, stochastic_output=True)`。当时 GPU 上还有 CARLA 和其他任务，
这些进程没有被停止。显存快照与 PyTorch 的额外临时张量需求不能简单相减，因此虽然报错前
free 数字略大于单项 allocation，请求仍因已有 allocation/reservation 和前向峰值而失败。

## 配置与溯源

- source commit：`8af2862f989fffed149f2ad87fd7044ddf8c29b9`
- seed：`201352625`
- rounds / rollout steps：`4 / 1024`
- actor：full-batch SGD，LR `5e-5`
- execution：25% filtered / 75% nominal，group-balanced advantages
- transaction：pooled rollout acceptance
- input checkpoint SHA-256：
  `9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317`
- 两个失败目录中 `round_00.pt` SHA-256：
  `cb555310a5dafff059a65aaeb4b189d115e066414b27eeaa048dbe5ee3ff59b9`
- 4080 原始目录：
  - `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/pooled_256_v81_8af2862_s201352625`
  - `/home/carla/LZQW/SAFE100/humanoid/artifacts/paper_dual_v35/pooled_192_v81b_8af2862_s201352625`

`decision_summary.json` 保存机器可读结论，`oom_excerpts.log` 保存两次关键报错。下一次不继续
盲目降低 environments；采用 microbatch 梯度累积，在一次 optimizer step 前累积完整的
256-environment 平均梯度。


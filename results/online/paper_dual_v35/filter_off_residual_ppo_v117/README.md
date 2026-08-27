# v117 filter-off episodic residual PPO

v117 不再模仿 CBF action。v79 actor 完全冻结，小型 residual head 在真实 filter-off
deployment states 中加入 `std=0.02` 探索；成功/失败 episode 分别平分正负 advantage，
随后做 clipped PPO，并投影到 reference KL `0.02`。

四个 64-env rollout 共 168/256 成功，收集 87,624 条 active-geometry transition。
训练的 unclipped surrogate 为正，但 minibatch Adam 的 clipped surrogate 从约 0
下降到 `-0.01666`；未投影 KL 为 `0.04591`，最终只保留 5.54% parameter delta，
KL 为 `0.01996`，离线 gate 明确失败。

唯一 deterministic filter-off screen（seed `201353880`）为
**44/64 = 68.75%**，mean reached riser `7.828`，没有运行独立 gate。候选拒绝，
正式最佳仍为 v79。

结果说明 deployment-state episode return 是正确坐标，但 44 次 Adam minibatch 把
正向 raw surrogate 扭成负向 clipped surrogate。下一步只做一次 full-batch
direction-preserving SGD，复用 v92 已验证的更新原则。

实现 commit：`64c7158de0caa7dcd09e2c52d132156132609cce`；训练总耗时
131.03 秒。模型二进制未上传，精确 SHA-256 见 `checkpoint_index.json`。

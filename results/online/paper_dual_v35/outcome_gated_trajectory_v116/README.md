# v116 outcome-gated successful trajectory residual

v116 保留 v115 的 causal GRU、rescued/harmed gate 和 `0.6` 高置信 fallback，只把
residual teacher 从 filter-off 状态上的瞬时 CBF 投影改为 matched-rescue 的真实
filter-on 成功轨迹动作。目标是让 residual 学习完整成功序列，而不是局部安全方向。

## Paired 数据

| Split | Filter off | Filter on | Rescued | Harmed |
|---|---:|---:|---:|---:|
| Train，3×64 | 140/192 | 135/192 | 33 | 38 |
| Validation，1×64 | 32/64 | 47/64 | 21 | 6 |
| Total | 172/256 | 182/256 | 54 | 44 |

唯一 filter-off screen（seed `201353780`）只有 **38/64 = 59.375%**；gate 在
3.73% transition 激活，平均 correction norm 为 `0.000333`，mean reached riser
为 `8.094`。结果远低于 v115 的 47/64，因此没有运行独立 gate。

这说明即使由因果 gate 限制，分叉后的 filter-on 成功轨迹 action 仍不是 filter-off
状态上的稳定 teacher；v109 暴露的 state/action mismatch 不能靠 outcome gating 消除。
后续不再使用该 teacher。

本轮同样在候选和 screen 已落盘后，由最终 summary 字典里遗留的第二处 Python
`false` 触发 `NameError`。commit `171ef70` 已修复所有剩余小写布尔值；没有重复
rollout。原始日志和完整 screen 证据已上传。实现 commit：
`931b88f22b2a68bcd0440e686779e8267802c468`。

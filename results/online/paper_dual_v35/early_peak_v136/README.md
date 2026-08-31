# v136 Earliest Exact-Peak Selection

v132 的 rollout 4（round-3 checkpoint）与 rollout 8（round-7 checkpoint）训练成功率
都精确为 `198/269 = 73.61%`。旧规则用 mean reached riser 与 later-checkpoint tie-break
选择 round 7；v133 又表明额外 continuation 会退化。v136 因此定义一个新的保守选择
协议：只读取 SHA 固定的 v132 training records，用整数分数比较 success rate；精确并列
时选择更新次数最少的最早 checkpoint。该规则选择 round 3，不读取 deployment 结果，
也不生成或训练新模型。

唯一 deterministic filter-off screen（seed `201355780`）结果为：

- **42/64 = 65.625%**
- 跌倒 22/64
- 平均到达 riser 8.2188

screen 未达到 `48/64`，因此没有运行独立 gate，也没有评估其他 v132 checkpoint。
保守 early tie-break 明显低于 v132 round-7 的 50/64 screen，说明 v132 的强结果不能
通过“同训练率时选择更少更新”稳定复制；原 v132 selected actor 继续保持最强候选。

## 溯源

- 代码提交：`4a53f1d8ae807960e909a3f8c741b02ef42e553f`。
- 唯一测试：
  `test_v136_selects_the_earliest_exact_peak_without_extra_evaluation`，
  1 passed in 16.47s。
- frozen round-metrics SHA-256：
  `7c0aa01acfaed0cc32f22a245384b20220234ada511c8a83b9d7300e7414ec9c`。
- selected checkpoint SHA-256：
  `f0c18b0965668fb8eab5e3fab6e8f2edc6555c35f9cb65df4b419c4c3df34b91`。
- 完整 selection manifest 与 screen JSON/CSV 已提交；模型是已有 v132 checkpoint，
  未重复提交二进制。

实现：`src/tasks/stairs_cbf/paper_early_peak_v136.py` 与
`experiments/scripts/select_paper_early_peak_v136.py`。

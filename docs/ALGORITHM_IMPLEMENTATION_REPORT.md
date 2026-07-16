# Unitree G1 楼梯 CBF-RL：整体算法与程序实现报告

## 1. 项目目标与当前结论

本项目在 `unitree_rl_mjlab`、MJLab、MuJoCo-Warp 和 `rsl_rl` PPO 上实现了
Unitree G1 的楼梯攀爬训练，并复现 CBF-RL 的训练期安全过滤思想。导航目标和
楼梯边沿表示借鉴 Hiking in the Wild 的 Flat Patch 目标与边沿感知思路。

服务器迁移后的项目根目录为：

`<repo-root>`

当前选择的有效控制屏障函数是“摆动脚相对下一阶立面的前向安全距离”。正式
CBF 策略使用 seed 42、1024 个并行环境训练 1500 次 PPO update。在固定
13 cm、六级楼梯、128 个确定性 episode 上：

| 方法 | 运行时 CBF 过滤 | 成功率 | 跌倒率 | 平均最大进度 | CBF 违反积分 |
|---|---:|---:|---:|---:|---:|
| CBF-RL | 开 | 95.31% | 4.69% | 3.309 m | 9.75 |
| CBF-RL | 关 | 95.31% | 4.69% | 3.315 m | 7.56 |
| 同预算 Nominal | 关 | 0% | 34.38% | 0.748 m | 1006.18 |

关闭在线过滤后成功率保持 95.31%，说明策略学习到了安全的爬楼动作，而不是
只依赖部署时 QP 修正。该结论是单随机种子方法级效果验证，不是多种子统计结论，
也不是论文数值完全复现或真机结果。

## 2. 软件栈与固定版本

| 组件 | 版本或提交 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER，16376 MiB |
| NVIDIA 驱动 | 550.144.03 |
| Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| MJLab | 1.2.0 |
| MuJoCo | 3.5.0 |
| MuJoCo-Warp | 3.5.0 |
| Warp | 1.12.0 |
| rsl-rl-lib | 5.0.1 |
| 上游 `unitree_rl_mjlab` | `1425b15f73bd4095f0df53709d7c389c3eb9e790` |
| 本地实现分支 | `repro/cbf-rl-g1-stairs` |
| 本地实现提交 | `1769c552fbbc86acb889e4215318108c9da34806` |

精确依赖记录位于：

- `artifacts/pip_freeze.txt`
- `artifacts/conda_list.txt`
- `artifacts/versions.txt`

## 3. 整体算法数据流

每个 policy step 的执行流程如下：

1. 程序化楼梯生成器建立六级前向楼梯和顶层平台，并为每个踏面生成 Flat Patch。
2. `StairTargetCommand` 从机器人前方踏面中选择目标点，将世界系目标转换到机体系。
3. 位置误差转换为 `vx / vy / wz` 命令并进行速度截断。
4. Actor 接收五帧本体感知历史，输出 12 个腿部关节位置目标。
5. 位置目标按控制周期转换为 nominal joint velocity。
6. 根据接触状态选出摆动脚，并从踏面 Flat Patch 恢复下一阶立面位置和高度。
7. MuJoCo-Warp 计算摆动脚实际位置 Jacobian。
8. 若 CBF 激活且 nominal action 违反约束，闭式 QP 将其投影到安全半空间。
9. 环境执行过滤后的关节目标；PPO 使用实际 rollout 更新 actor/critic。
10. Dual CBF reward 同时惩罚 nominal CBF 违反和策略动作与安全动作的距离，使
    actor 逐渐内化约束。
11. 楼梯高度课程从低台阶逐步提升到目标高度，评估时重新构造严格 13 cm 楼梯。

## 4. 状态、动作与网络接口

### 4.1 机器人与动作

- 仿真机器人：完整 29-DoF Unitree G1。
- Policy action：12 维腿部关节位置目标，包括双腿 hip、knee 和 ankle。
- 上身未由策略控制，保持默认关节目标。
- 控制周期：`4 × 0.005 s = 0.02 s`。

Nominal 关节速度由位置目标计算：

```text
qdot_nominal = (q_target_policy - q) / 0.02
```

### 4.2 Actor observation

Actor 不接收 privileged height scan。每个时刻包含：

- base angular velocity；
- projected gravity；
- 目标速度命令；
- gait phase；
- 29 维 joint position；
- 29 维 joint velocity；
- 12 维 previous action。

使用五帧历史，总 actor observation shape 为 `405`。

### 4.3 Critic observation

Critic 使用单帧 privileged observation，包括 actor 单帧信号、187 维高度扫描、
base linear velocity、脚高度、腾空时间、接触状态和接触力，总 shape 为 `283`。

### 4.4 PPO 网络

- Actor MLP：`405 -> 512 -> 256 -> 128 -> 12`，ELU；
- Critic MLP：`283 -> 512 -> 256 -> 128 -> 1`，ELU；
- Gaussian policy；
- backend：`MjlabOnPolicyRunner / rsl_rl PPO`；
- `num_steps_per_env = 24`。

## 5. 楼梯 CBF 的构造

### 5.1 安全集

设当前摆动脚前向坐标为 `x_foot`，相关下一阶立面为 `x_riser`，脚尖余量为
`d_toe = 0.08 m`：

```text
h(q) = x_riser - (x_foot(q) + d_toe)
```

`h >= 0` 表示摆动脚尚未侵入立面安全边界。脚抬高超过当前踏面后约束解除，
允许脚越过立面落到踏面上。

### 5.2 一阶指数 CBF 条件

因为 `dh/dq = -J_x_swing(q)`，使用：

```text
psi(q, qdot) = -J_x_swing(q) qdot + alpha h(q) >= 0
alpha = 10 s^-1
```

实际 Jacobian 由 `mujoco_warp.jac` 在 GPU 上批量计算，没有使用固定、伪造或
有限差分 Jacobian。

### 5.3 闭式安全过滤器

对单个线性半空间求 Euclidean projection：

```text
qdot_safe = argmin ||qdot - qdot_nominal||^2
            s.t. normal @ qdot >= rhs

normal = -J_x_swing
rhs    = -alpha * h
```

若 nominal margin 为负：

```text
lambda = -margin / max(||normal||^2, eps)
qdot_safe = qdot_nominal + lambda * normal
```

否则动作保持不变。安全速度重新转换为位置目标后交给 MJLab actuator。

### 5.4 CBF 激活逻辑

- 摆动脚：未接触地面且当前腾空时间最长的脚；
- 双脚支撑时约束不激活；
- 只考虑距离脚最近的相关立面；
- 立面前 `0.30 m` 开始激活；
- 已越过立面最多 `0.15 m` 时仍允许恢复；
- 足端高度超过踏面 `0.025 m` 后解除该立面约束。

### 5.5 Dual CBF reward

训练使用论文 CBF-RL 的 bounded dual reward 结构：

```text
r_cbf = min(psi_nominal, 0)
        + exp(-||qtarget_policy - qtarget_safe||^2 / sigma^2) - 1

sigma = 0.5
```

第一项惩罚 nominal policy 自身的安全违反，第二项促使 policy action 接近安全
filter 输出。因此训练结束后可以关闭运行时过滤进行部署评估。

## 6. 地形、边沿和导航实现

### 6.1 程序化楼梯

- 六级楼梯；
- tread run：`0.35 m`；
- 正式评估 rise：`0.13 m`；
- 顶层平台长度：`1.2 m`；
- 课程训练 rise 范围：`0.02--0.155 m`，五个 terrain level。

### 6.2 Flat Patch 导航点

每个踏面中心和顶层平台建立合法目标 patch。命令生成器选择机器人前方目标，
并使用 look-ahead 防止每一级频繁停顿：

```text
delta_body = rotate_world_target_to_base(target - root_position)
vx = clip(1.5 * delta_body.x, 0.0, 0.8)
vy = clip(1.5 * delta_body.y, -0.2, 0.2)
wz = clip(1.5 * heading_error, -0.8, 0.8)
```

### 6.3 边沿表示

Hiking in the Wild 对任意三角网格检测 sharp edge。本项目的楼梯是程序化且轴
对齐的，因此从有序踏面 Flat Patch 中精确恢复每个 riser：

```text
x_riser = x_tread_center - 0.5 * tread_width
z_top   = z_tread
```

对该楼梯任务，这比每次遍历 mesh 更直接且无几何近似；它不是通用非结构化地形
二面角检测器。

## 7. 公共奖励、终止与课程

CBF 与 Nominal 使用相同的公共奖励和随机化，唯一实验差异是 CBF filter 和
CBF dual reward。主要 reward 包括：

- linear/angular velocity tracking；
- body orientation、pose、angular momentum；
- joint acceleration、joint limits、action rate；
- gait、stair-relative foot clearance；
- foot slip、soft landing、swing-foot force；
- stair progress；
- `dont_wait`，避免机器人在第一阶前站立至 timeout；
- self collision 和 termination penalty。

终止包括 20 秒 timeout 和 fell-over。训练使用摩擦、编码器 bias、base COM 等
domain randomization。

## 8. 程序文件映射

核心实现目录：

`third_party/unitree_rl_mjlab/src/tasks/stairs_cbf/`

| 文件 | 作用 |
|---|---|
| `__init__.py` | 注册 CBF、Nominal 和冻结的 Engineering29 任务 |
| `config.py` | G1 observation/action、reward、课程、PPO 配置 |
| `terrain.py` | 六级程序化楼梯、顶层平台和 Flat Patch |
| `command.py` | Flat Patch 目标选择和位置式速度命令 |
| `edge_detection.py` | 从踏面 patch 恢复 riser 并选择活动边沿 |
| `cbf_math.py` | barrier、闭式 half-space projection、dual reward |
| `actions.py` | 实际 Jacobian、摆动脚选择和训练期 action filter |
| `mdp.py` | CBF、进度、clearance、dont-wait 等 reward |

项目级脚本：

| 文件 | 作用 |
|---|---|
| `scripts/smoke_stairs_cbf.py` | 环境、shape、finite、CBF margin smoke |
| `scripts/checkpoint_roundtrip.py` | checkpoint 严格保存/加载验证 |
| `scripts/capacity_sweep.py` | RTX 4080 环境数/显存/吞吐扫描 |
| `scripts/evaluate_stairs.py` | 批量 deterministic 统一评估 |
| `scripts/render_stairs_video.py` | EGL headless 有限步 MP4 录制 |
| `scripts/run_nominal_1500.sh` | 同预算 Nominal 对照训练 |

任务 ID：

- `Unitree-G1-Stairs-CBF`
- `Unitree-G1-Stairs-Nominal`
- `Unitree-G1-Stairs-Engineering29-CBF`
- `Unitree-G1-Stairs-Engineering29-Nominal`

## 9. 测试证据

- 纯张量测试：5 passed；
- G1 action shape：12；
- actor/critic observation：`[4,405] / [4,283]`；
- 所有 smoke observation/action/reward finite；
- 真实 MuJoCo-Warp 对抗测试：
  - nominal CBF margin：`-29.166748`；
  - filtered margin：`+1.0878e-6`；
- 5-update PPO integration smoke 通过；
- bounded dual reward 在 filter intervention 时非零；
- checkpoint roundtrip 严格加载通过并产生 finite 12-D action；
- RTX 4080 isolated capacity test 到 4096 env 无 OOM；
- 因共享 GPU 上可能运行 CARLA，正式训练保守选择 1024 env。

测试汇总：`artifacts/test_summary.json`。

## 10. 正式结果和产物

### 10.1 CBF 模型

```text
third_party/unitree_rl_mjlab/logs/rsl_rl/
g1_stairs_cbf/2026-07-16_00-00-11_paper_dual_curriculum_seed42_1024/
├── model_1500.pt
├── policy.onnx
└── events.out.tfevents.*
```

SHA256：

```text
model_1500.pt  c9ad30672f90dad7fc55360b056375658f31b5312f7f29868bbcb6d1086441a1
policy.onnx    4433b6b27da941e4e118d8a07775f5c085de8a2c75a52c1daa775c5401c9f72a
```

### 10.2 Nominal 对照

```text
third_party/unitree_rl_mjlab/logs/rsl_rl/
g1_stairs_nominal/2026-07-16_00-22-14_paper_nominal_curriculum_seed42_1024_1500/
├── model_1499.pt
├── policy.onnx
└── events.out.tfevents.*
```

`model_1499.pt` 是零基计数的第 1500 次 PPO update 后的最终模型。

### 10.3 评估和视频

- 汇总 CSV：`reports/evaluation_comparison.csv`；
- 逐 episode CSV：`reports/eval_curriculum_*.csv`；
- 评估报告：`reports/EVALUATION_SUMMARY.md`；
- 视频：`videos/g1-stairs-cbf-filter-off-seed42-step-0.mp4`；
- 视频指标：`artifacts/video_cbf_filter_off_seed42.json`。

视频为固定 13 cm 楼梯、关闭 runtime filter 的确定性 rollout：854×480、50 FPS、
20 秒、1000 帧；最大进度 3.537 m、未跌倒、CBF 违反事件 0，并通过完整 ffmpeg
解码检查。

## 11. 运行与复现实例

迁移后的独立 Conda 环境：

```text
<repo-root>/workspace/conda_env
```

训练示例：

```bash
cd <repo-root>/third_party/unitree_rl_mjlab
python scripts/train.py Unitree-G1-Stairs-CBF \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 1500 \
  --agent.save-interval 500 \
  --agent.seed 42 \
  --agent.logger tensorboard
```

TensorBoard：

```bash
tensorboard --logdir <repo-root>/third_party/unitree_rl_mjlab/logs/rsl_rl \
  --bind_all --port 6006
```

完整执行命令保存在 `COMMANDS.md`。

## 12. 工程假设与限制

论文没有公开完整 humanoid stair 代码和全部参数，因此下列参数是明确记录的工程
假设：toe margin 0.08 m、`alpha=10`、激活/恢复距离、`sigma=0.5`、clearance
系数、swing-foot force 系数、`dont_wait` 和课程范围。详见：

- `ASSUMPTIONS.md`
- `BLOCKERS.md`
- `reports/METHOD_PARITY.md`

本项目仅完成 MuJoCo/MJLab 仿真训练和评估，没有连接真实 Unitree G1，没有运行
真机控制或 sim2real。服务器迁移快照包含所有代码、Git 历史、日志、checkpoint、
ONNX、CSV、报告、视频，以及通过 Conda `--clone` 生成并重新绑定 editable 包的
完整 Python 环境。原项目和原环境仍完整保留在服务器旧路径，未删除。

# Unitree G1 楼梯 CBF-RL 训练框架总览

> 本文对应 4080 服务器当前实际实现，项目根目录：
> `<repo-root>`

## 1. 当前任务要解决什么

目标是在 MuJoCo/MJLab 中训练完整 29-DoF Unitree G1 爬上六级楼梯，并用
CBF-RL 的训练期安全过滤减少摆动脚踢到下一阶立面的行为。

当前正式任务：

- `Unitree-G1-Stairs-CBF`：训练 rollout 使用 CBF 安全过滤，同时启用 CBF
  dual reward；
- `Unitree-G1-Stairs-Nominal`：同样的机器人、地形、导航、公共奖励、课程和
  PPO 参数，但不执行 CBF 动作过滤，CBF reward 权重为 0；
- 两个 `Engineering29` 任务是早期 29 维动作实验的冻结版本，不是当前正式结果。

当前结果属于单随机种子、仿真方法级效果验证，不是多种子论文数值复现，也没有
进行真机控制或 sim2real。

## 2. 总体系统结构

训练链路如下：

```text
程序化楼梯 + Flat Patch
        │
        ├── 踏面目标选择 ──> 机体系速度命令 [vx, vy, wz]
        │                              │
        │                              v
        │                  Actor 五帧本体感知历史
        │                              │
        │                              v
        │                    12 维关节位置动作
        │                              │
        ├── 楼梯立面边沿 ──> CBF + 实际足端 Jacobian
        │                              │
        │                              v
        │                    安全动作闭式 QP 投影
        │                              │
        v                              v
  MuJoCo-Warp GPU 仿真 <──── 执行过滤后的关节目标
        │
        ├── actor observation / reward / termination
        ├── critic privileged observation
        └── PPO rollout 和参数更新
```

训练时策略不直接看到目标点坐标、台阶边缘或台阶高度。Flat Patch 导航器先将
目标点转换成三维速度指令，actor 只接收 `[vx, vy, wz]`。

## 3. 软件框架与版本

| 组件 | 当前版本或提交 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER，16376 MiB |
| Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| MJLab | 1.2.0 |
| MuJoCo | 3.5.0 |
| MuJoCo-Warp | 3.5.0 |
| Warp | 1.12.0 |
| rsl-rl-lib | 5.0.1 |
| unitree_rl_mjlab 上游基准 | `1425b15f73bd4095f0df53709d7c389c3eb9e790` |
| 当前实现分支 | `repro/cbf-rl-g1-stairs` |
| 当前实现提交 | `1769c552fbbc86acb889e4215318108c9da34806` |

训练环境：

```text
<repo-root>/workspace/conda_env
```

激活方式：

```bash
cd <repo-root>
source scripts/activate_migrated_env.sh
```

## 4. 仿真与机器人配置

### 4.1 时间尺度

| 参数 | 数值 |
|---|---:|
| Physics timestep | 0.005 s，200 Hz |
| Control decimation | 4 |
| Policy/control timestep | 0.02 s，50 Hz |
| Episode length | 20 s |
| 每个 episode 最大步数 | 1000 policy steps |

### 4.2 G1 模型

- 仿真加载完整 29-DoF G1；
- 关节、碰撞 mesh、内置 position actuator 和 PD 参数使用上游 G1 模型；
- 软关节位置限位系数为 0.9；
- 足部每侧使用 7 个碰撞几何；
- 足地接触传感器提供接触、接触力和 air time；
- self-collision sensor 使用 4 帧 force history。

虽然完整机器人是 29-DoF，正式 policy 只控制双腿 12 个 DoF。腰部和双臂在
当前 policy 中保持默认关节目标。

## 5. 地形和课程学习

### 5.1 楼梯几何

| 参数 | 数值 |
|---|---:|
| 台阶数 | 6 |
| 每级水平长度 | 0.35 m |
| 楼梯宽度 | 2.4 m |
| 第一阶世界 x | 1.35 m |
| 机器人出生 x | 0.75 m |
| 第一阶相对机器人 | 0.60 m |
| 顶层平台长度 | 1.2 m |
| 正式评估台阶高度 | 0.13 m |

每个踏面中心和顶层平台都生成一个 `stair_targets` Flat Patch。

### 5.2 训练课程

训练使用五个 terrain level，step height 配置范围为：

```text
0.02 m -- 0.155 m
```

seed 42 实际生成的五个 level 第一阶高度约为：

```text
0.04090, 0.05885, 0.09718, 0.11983, 0.13054 m
```

机器人走得足够远则提升 terrain level；走得明显不足则降低 level。正式评估不
直接使用最后一个随机 bin，而是重新生成严格 `0.13 m` 楼梯。

## 6. 导航命令如何生成和输入

### 6.1 Flat Patch 目标选择

导航器从当前机器人前方的踏面 patch 中选目标，并采用 `lookahead=2`，避免每上
一级都停下来重新瞄准。机器人进入目标点 `0.22 m` 半径后切换到下一目标；到达
顶层最后一个目标后命令置零。

### 6.2 从位置目标生成速度命令

先将世界系位置误差转换到机器人 base frame：

```text
delta_b = R_wb^T (target_w - root_w)
```

然后生成：

```text
vx = clip(1.5 * delta_b.x,  0.0,  0.8) m/s
vy = clip(1.5 * delta_b.y, -0.2,  0.2) m/s
wz = clip(1.5 * heading_error, -0.8, 0.8) rad/s
```

Actor 只接收最终 `[vx, vy,wz]`，不直接接收世界坐标、目标索引、距离、台阶高度
或边沿位置。

### 6.3 与真机遥控器的接口关系

仿真中：

```text
Flat Patch 自动导航器 -> [vx, vy, wz] -> actor
```

真机预期：

```text
人眼观察 + 遥控器 -> [vx, vy, wz] -> actor
```

两者对 actor 的三维接口相同，但真机需要操作人员负责对正楼梯、修正偏航和到顶
后松开摇杆。当前 actor 没有深度图或 height scan，因此不是自主视觉导航策略。

## 7. Actor 状态空间

Actor 使用五帧历史。每帧 81 维，总维度为 `81 × 5 = 405`。

| 顺序 | 状态项 | 单帧维度 | 五帧维度 | 来源/含义 |
|---:|---|---:|---:|---|
| 1 | base angular velocity | 3 | 15 | IMU，机体系角速度 |
| 2 | projected gravity | 3 | 15 | 机体系重力方向，反映姿态 |
| 3 | command | 3 | 15 | `[vx, vy, wz]` 导航/遥控命令 |
| 4 | gait phase | 2 | 10 | `[sin(2πφ), cos(2πφ)]` |
| 5 | joint position | 29 | 145 | 相对默认姿态的全部 G1 关节位置 |
| 6 | joint velocity | 29 | 145 | 全部 G1 关节速度 |
| 7 | previous action | 12 | 60 | 上一步腿部 policy action |
|  | **合计** | **81** | **405** |  |

### 7.1 训练 observation noise

- base angular velocity：uniform noise `[-0.2, 0.2]`；
- projected gravity：uniform noise `[-0.05, 0.05]`；
- joint position：uniform noise `[-0.01, 0.01]`；
- joint velocity：uniform noise `[-1.5, 1.5]`；
- play/evaluation 模式关闭 actor corruption。

Actor 明确不包含：

- base linear velocity；
- terrain height scan；
- 足端高度、接触力和 air time；
- Flat Patch 坐标；
- 台阶边沿、CBF `h/psi`；
- terrain level 或准确台阶高度。

## 8. Critic 特权状态空间

Critic 采用 asymmetric observation，只使用单帧，总维度 283：

| 状态项 | 维度 |
|---|---:|
| base angular velocity | 3 |
| projected gravity | 3 |
| command | 3 |
| gait phase | 2 |
| 29 joint positions | 29 |
| 29 joint velocities | 29 |
| 12 previous actions | 12 |
| terrain height scan | 187 |
| base linear velocity | 3 |
| 两脚高度 | 2 |
| 两脚 air time | 2 |
| 两脚接触状态 | 2 |
| 两脚接触力 | 6 |
| **合计** | **283** |

Height scan 是 pelvis 周围 `1.6 m × 1.0 m`、分辨率 `0.1 m` 的 187 点 ray scan。
这些 privileged 信息只帮助 critic 在训练时估值，不进入导出的 actor ONNX。

## 9. 固定 gait phase

当前步态先验：

```text
period = 0.6 s
frequency = 1 / 0.6 ≈ 1.67 Hz
left/right offset = [0.0, 0.5]
stance fraction = 0.56
```

即每条腿理论支撑约 `0.336 s`、摆动约 `0.264 s`，左右脚错开 `0.3 s`。

Gait phase 只作为 observation 和软 reward，不直接产生关节轨迹。命令范数小于
0.1 时，phase observation 置零，foot gait reward 关闭，避免静止时强迫踏步。

## 10. 动作空间

### 10.1 12 个动作及顺序

| index | 关节 | 默认 offset (rad) | action scale (rad) |
|---:|---|---:|---:|
| 0 | left_hip_pitch_joint | -0.1 | 0.54755 |
| 1 | left_hip_roll_joint | 0.0 | 0.35066 |
| 2 | left_hip_yaw_joint | 0.0 | 0.54755 |
| 3 | left_knee_joint | 0.3 | 0.35066 |
| 4 | left_ankle_pitch_joint | -0.2 | 0.43858 |
| 5 | left_ankle_roll_joint | 0.0 | 0.43858 |
| 6 | right_hip_pitch_joint | -0.1 | 0.54755 |
| 7 | right_hip_roll_joint | 0.0 | 0.35066 |
| 8 | right_hip_yaw_joint | 0.0 | 0.54755 |
| 9 | right_knee_joint | 0.3 | 0.35066 |
| 10 | right_ankle_pitch_joint | -0.2 | 0.43858 |
| 11 | right_ankle_roll_joint | 0.0 | 0.43858 |

Nominal 位置目标为：

```text
q_target_policy = q_default + action_scale * action_raw
```

当前 PPO Gaussian action 没有额外显式 clip。action scale 根据 actuator effort
limit 和 stiffness 计算：`0.25 * effort_limit / stiffness`。

### 10.2 CBF 前的速度形式

CBF 先把 policy 位置目标转换为关节速度：

```text
qdot_nominal = (q_target_policy - q_current) / 0.02
```

QP 投影后再转回位置目标：

```text
q_target_safe = q_current + 0.02 * qdot_safe
```

## 11. CBF-RL 算法设计

### 11.1 选择的安全问题

CBF 针对“摆动脚脚尖撞到下一阶竖直立面”，不是完整的防跌倒、关节限位或
全身碰撞 CBF。

下一阶边沿从程序化踏面 Flat Patch 精确恢复：

```text
x_riser = x_tread_center - 0.5 * tread_width
z_top   = z_tread
```

摆动脚通过真实接触状态判断：未接触且 air time 最长的脚。双支撑时 CBF 不激活。

### 11.2 Barrier function

```text
h(q) = x_riser - (x_swing_foot(q) + toe_margin)
toe_margin = 0.08 m
```

安全集合为 `h >= 0`。足端已经高于踏面 `0.025 m` 后，该立面约束解除，让脚能
越过边沿落到踏面上。

### 11.3 一阶指数 CBF

```text
psi(q, qdot) = -Jx_swing(q) qdot + alpha * h(q) >= 0
alpha = 10 s^-1
```

足端 Jacobian 使用 `mujoco_warp.jac` 在 GPU 上根据当前仿真状态实际计算。

CBF 在立面前 0.30 m 内激活，并保留最多 0.15 m 的越界恢复区。

### 11.4 闭式 QP filter

```text
qdot_safe = argmin ||qdot - qdot_nominal||²
            s.t. normal · qdot >= rhs

normal = -Jx_swing
rhs    = -alpha * h
```

单个 half-space 的闭式解：

```text
margin = normal · qdot_nominal - rhs

if margin < 0:
    lambda = -margin / max(||normal||², eps)
    qdot_safe = qdot_nominal + lambda * normal
else:
    qdot_safe = qdot_nominal
```

### 11.5 Dual CBF reward

正式 CBF 任务同时给 policy 一个训练信号：

```text
r_cbf = min(psi_nominal, 0)
        + exp(-||qtarget_policy - qtarget_safe||² / sigma²) - 1

sigma = 0.5
```

第一项惩罚 policy 自身的 nominal CBF 违反；第二项惩罚 policy 动作偏离 filter
输出。目标是让 policy 逐渐内化安全约束，最终可以在不运行 QP 的情况下保持安全。

## 12. Reward 设计

配置中的 reward weight 会由 MJLab 按 `dt=0.02` 再缩放，避免改变仿真步长时
累计 reward 尺度剧烈变化。下面是 reward manager 中实际启用的 20 项。

| Reward | 权重 | 符号 | 作用 |
|---|---:|---|---|
| `track_linear_velocity` | +1.0 | 奖励 | 跟踪 `[vx,vy]`，同时抑制 base z 速度 |
| `track_angular_velocity` | +1.0 | 奖励 | 跟踪 `wz`，小幅抑制 roll/pitch 角速度 |
| `body_orientation_l2` | -1.0 | 惩罚 | torso projected gravity XY 平方，保持直立 |
| `pose` | +1.0 | 奖励 | 速度相关的默认姿态 Gaussian reward |
| `body_ang_vel` | -0.05 | 惩罚 | torso roll/pitch 角速度平方 |
| `angular_momentum` | -0.025 | 惩罚 | 全身角动量平方 |
| `is_terminated` | -200.0 | 惩罚 | 非 timeout 异常终止 |
| `joint_acc_l2` | -2.5e-7 | 惩罚 | 关节加速度平方 |
| `joint_pos_limits` | -10.0 | 惩罚 | 超过 soft joint position limits |
| `action_rate_l2` | -0.05 | 惩罚 | 相邻 policy action 变化平方 |
| `foot_gait` | +0.5 | 奖励 | 实际接触与 0.6 s gait schedule 匹配 |
| `foot_clearance` | -1.0 | 惩罚 | 足高度偏离目标，按足端水平速度加权 |
| `foot_slip` | -0.25 | 惩罚 | 接触脚水平速度平方 |
| `soft_landing` | -0.001 | 惩罚 | 首次落地冲击力 |
| `stand_still` | -1.0 | 惩罚 | 零命令时关节偏离默认姿态 |
| `self_collisions` | -1.0 | 惩罚 | 超过 10 N 的自碰撞历史计数 |
| `stair_progress` | +0.25 | 奖励 | `max(root_x - origin_x, 0)` |
| `dont_wait` | -1.0 | 惩罚 | 有前进命令时速度低于 0.1 m/s 的缺口 |
| `swing_foot_force` | -0.001 | 惩罚 | gait 计划摆动阶段的接触力 |
| `cbf_dual` | +1.0 | CBF 信号 | nominal 违反 + safe-action imitation |

### 12.1 主要 reward 公式

线速度跟踪：

```text
r_lin = exp(-( ||v_cmd_xy-v_xy||² + 2*v_z² ) / 0.5²)
```

角速度跟踪：

```text
r_ang = exp(-( (wz_cmd-wz)² + 0.05*(wx²+wy²) ) / sqrt(0.5)²)
```

Foot gait：

```text
r_gait = mean(planned_stance == measured_contact)
```

楼梯 foot clearance：

```text
cost = sum( |z_foot-z_target| * ||v_foot_xy|| )
z_target = z_next_tread + 0.05 m   # CBF 相关摆动脚
z_target = origin_z + 0.10 m       # 其他情况
```

Dont-wait：

```text
cost = relu(0.1 - vx_actual) / 0.1,  when vx_command > 0.2
```

### 12.2 CBF 与 Nominal 的公平差异

两组都保留导航、课程、clearance、dont-wait、swing-force 和其他公共 reward。

| 项目 | CBF task | Nominal task |
|---|---|---|
| 训练 rollout QP filter | 开 | 关 |
| `cbf_dual` 权重 | 1.0 | 0.0 |
| 其他配置 | 相同 | 相同 |

## 13. Termination

正式任务只启用两项终止：

| Termination | 条件 |
|---|---|
| `time_out` | 20 秒，即 1000 policy steps |
| `fell_over` | bad orientation，倾角超过 70° |

评估的 success 是 episode 内最大前进距离达到 `2.65 m`。Success 本身不立即终止
episode，因此一个成功 episode 也可能继续运行至 timeout；success 和 timeout
不是互斥统计。

## 14. Domain Randomization 和扰动

### 14.1 Reset 随机化

```text
root x:   [-0.05, 0.05] m
root y:   [-0.08, 0.08] m
root yaw: [-0.08, 0.08] rad
```

### 14.2 Startup randomization

- 足底摩擦系数：`[0.3, 1.6]`，同一机器人足部共享；
- encoder bias：`[-0.015, 0.015]`；
- torso COM offset：x/y/z 各 `[-0.05, 0.05] m`。

### 14.3 训练中 push

每 5--6 秒通过设置 base velocity 施加一次扰动，范围：

```text
linear x/y: [-0.5, 0.5] m/s
linear z:   [-0.4, 0.4] m/s
roll/pitch: [-0.52, 0.52] rad/s
yaw:        [-0.78, 0.78] rad/s
```

Play/evaluation 模式关闭 observation corruption、push 和 curriculum。

## 15. PPO 网络与超参数

### 15.1 网络

Actor：

```text
405 -> 512 -> 256 -> 128 -> 12
activation: ELU
distribution: Gaussian, scalar initial std = 1.0
observation normalization: enabled
```

Critic：

```text
283 -> 512 -> 256 -> 128 -> 1
activation: ELU
observation normalization: enabled
```

Actor 和 critic 是独立 MLP，不共享 encoder。

### 15.2 PPO 参数

| 参数 | 数值 |
|---|---:|
| value loss coefficient | 1.0 |
| clipped value loss | True |
| PPO clip | 0.2 |
| entropy coefficient | 0.01 |
| learning epochs/update | 5 |
| mini-batches/epoch | 4 |
| learning rate | 1e-3，adaptive |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| desired KL | 0.01 |
| max gradient norm | 1.0 |
| rollout steps/environment | 24 |

## 16. 正式训练是怎么跑的

用户要求单 seed 有效结果，因此正式比较使用：

```text
seed = 42
num_envs = 1024
num_steps_per_env = 24
rollout batch = 1024 × 24 = 24576 samples/update
mini-batch = 24576 / 4 = 6144 samples
iterations = 1500
累计采样 = 36,864,000 environment transitions
```

RTX 4080 capacity scan 在独占条件下测试到 4096 environments 无 OOM；服务器会
间歇运行无关 CARLA 进程，因此正式训练保守选择 1024 environments。

训练命令等价于：

```bash
cd <repo-root>/third_party/unitree_rl_mjlab
source <repo-root>/scripts/activate_migrated_env.sh

python scripts/train.py Unitree-G1-Stairs-CBF \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 1500 \
  --agent.save-interval 500 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --agent.run-name paper_dual_curriculum_seed42_1024
```

所有长训练在 tmux 中运行，stdout/stderr 同时 `tee` 到项目 `logs/`。

## 17. 训练、评估和最终结果

固定 13 cm 六级楼梯、128 deterministic episodes、seed 42：

| 方法 | checkpoint | runtime filter | 成功率 | 跌倒率 | 最大进度均值 | CBF 违反事件 | CBF 违反积分 |
|---|---|---:|---:|---:|---:|---:|---:|
| CBF | model_1500.pt | 开 | 95.31% | 4.69% | 3.309 m | 3.64 | 9.75 |
| CBF | model_1500.pt | 关 | 95.31% | 4.69% | 3.315 m | 3.23 | 7.56 |
| Nominal | model_1499.pt | 关 | 0% | 34.38% | 0.748 m | 242.37 | 1006.18 |

关闭 runtime filter 后 CBF-trained policy 仍保持 95.31% success，说明训练得到的
actor 已在这个单 seed 实验中内化了相当一部分安全行为。

## 18. 代码模块对应关系

核心目录：`third_party/unitree_rl_mjlab/src/tasks/stairs_cbf/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | 注册 CBF/Nominal/Engineering29 任务 |
| `config.py` | 正式状态、动作、reward、课程和 runner 配置 |
| `terrain.py` | 六级楼梯、顶层平台和 Flat Patch |
| `command.py` | 位置目标到 `[vx,vy,wz]` 导航命令 |
| `edge_detection.py` | 从踏面 patch 恢复立面边沿并选择活动边沿 |
| `cbf_math.py` | barrier、闭式 QP 和 dual reward |
| `actions.py` | 摆动脚、实际 Jacobian、CBF action filter |
| `mdp.py` | CBF、进度、clearance、dont-wait 等 reward |

辅助脚本：

| 文件 | 作用 |
|---|---|
| `scripts/smoke_stairs_cbf.py` | 环境、shape、finite、真实 CBF margin smoke |
| `scripts/checkpoint_roundtrip.py` | checkpoint 严格加载与 inference |
| `scripts/capacity_sweep.py` | 显存和吞吐扫描 |
| `scripts/evaluate_stairs.py` | 固定台阶高度、批量 deterministic evaluation |
| `scripts/render_stairs_video.py` | EGL headless MP4 录制 |
| `scripts/activate_migrated_env.sh` | 激活迁移后的独立环境 |

## 19. 产物在哪里

CBF checkpoint：

```text
third_party/unitree_rl_mjlab/logs/rsl_rl/g1_stairs_cbf/
2026-07-16_00-00-11_paper_dual_curriculum_seed42_1024/model_1500.pt
```

CBF ONNX：同目录 `policy.onnx`。

Nominal checkpoint：

```text
third_party/unitree_rl_mjlab/logs/rsl_rl/g1_stairs_nominal/
2026-07-16_00-22-14_paper_nominal_curriculum_seed42_1024_1500/model_1499.pt
```

评估和视频：

- `reports/evaluation_comparison.csv`；
- `reports/EVALUATION_SUMMARY.md`；
- `artifacts/test_summary.json`；
- `videos/g1-stairs-cbf-filter-off-seed42-step-0.mp4`；
- `artifacts/video_cbf_filter_off_seed42.json`。

## 20. 当前限制和下一步部署注意事项

1. CBF 只处理摆动脚与楼梯立面的前向碰撞，不保证全身 forward invariance。
2. Toe margin、alpha、激活区间、sigma、clearance 和 dont-wait 参数是记录过的
   工程假设，因为论文没有发布完整 humanoid stair 参数。
3. Actor 没有视觉或 terrain scan，真机高层导航需要人类遥控或另接感知导航器。
4. 当前 ONNX 是 405 维五帧输入、12 维动作；上游现成 G1 deploy YAML 是单帧、
   29 维动作，不能直接互换。
5. 真机部署前必须补齐五帧 ring buffer、12->29 action mapping、未控制关节默认
   姿态、训练一致的 observation normalization、遥控命令限幅、状态机和急停。
6. 当前仅完成 MuJoCo/MJLab 仿真；没有连接真实 G1，也没有声称 sim2real 完成。

更细的论文方法差异、工程假设和迁移验证分别参见：

- `ALGORITHM_IMPLEMENTATION_REPORT.md`
- `reports/METHOD_PARITY.md`
- `ASSUMPTIONS.md`
- `BLOCKERS.md`
- `MIGRATION_MANIFEST.md`

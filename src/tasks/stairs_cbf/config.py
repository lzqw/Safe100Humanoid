"""G1 stair tasks with and without the CBF-RL dual mechanism."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from src.assets.robots import G1_ACTION_SCALE
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from . import mdp
from .actions import StairCbfJointPositionActionCfg
from .command import StairTargetCommandCfg
from .terrain import ForwardStairsTerrainCfg

NUM_STEPS = 6
STEP_WIDTH = 0.35
FIRST_RISER_X = 1.35
SPAWN_X = 0.75
LOWER_BODY_JOINT_PATTERNS = (
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_hip_yaw_joint",
  ".*_knee_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
)


def _terrain_cfg() -> TerrainGeneratorCfg:
  stairs = ForwardStairsTerrainCfg(
    proportion=1.0,
    # Keep the geometric height exact for the analytic CBF. Height curriculum
    # is intentionally deferred until per-patch riser metadata is available.
    step_height_range=(0.13, 0.13),
    step_width=STEP_WIDTH,
    num_steps=NUM_STEPS,
    first_riser_x=FIRST_RISER_X,
    spawn_x=SPAWN_X,
    flat_patch_sampling={
      "stair_targets": FlatPatchSamplingCfg(
        num_patches=NUM_STEPS + 1,
        patch_radius=0.12,
        max_height_diff=0.02,
      )
    },
  )
  return TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(5.2, 4.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    difficulty_range=(0.0, 1.0),
    sub_terrains={"forward_stairs": stairs},
    add_lights=True,
  )


def g1_stairs_env_cfg(
  *,
  use_filter: bool,
  use_cbf_reward: bool,
  play: bool = False,
  paper_spec: bool = True,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_rough_env_cfg(play=False)
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=_terrain_cfg(),
    max_init_terrain_level=0,
  )
  cfg.scene.extent = 3.0
  cfg.episode_length_s = 20.0 if not play else 60.0
  cfg.curriculum.pop("command_vel", None)
  if play:
    cfg.curriculum = {}
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  # CBF-RL humanoid setup: five-frame proprioceptive actor history, with the
  # one-frame height scan reserved for the asymmetric critic.
  if paper_spec:
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["actor"].history_length = 5
    cfg.observations["critic"].history_length = 1

  cfg.commands["twist"] = StairTargetCommandCfg(
    entity_name="robot",
    resampling_time_range=(20.0, 20.0),
    debug_vis=False,
    patch_name="stair_targets",
  )
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.08, 0.08),
    "z": (0.0, 0.0),
    "yaw": (-0.08, 0.08),
  }

  nominal = cfg.actions["joint_pos"]
  action_targets = LOWER_BODY_JOINT_PATTERNS if paper_spec else (".*",)
  action_scale = (
    {name: G1_ACTION_SCALE[name] for name in LOWER_BODY_JOINT_PATTERNS}
    if paper_spec
    else G1_ACTION_SCALE
  )
  cfg.actions["joint_pos"] = StairCbfJointPositionActionCfg(
    entity_name="robot",
    actuator_names=action_targets,
    scale=action_scale,
    use_default_offset=True,
    enabled=use_filter,
    contact_sensor_name="feet_ground_contact",
    foot_site_names=("left_foot", "right_foot"),
    alpha=10.0,
    first_riser_offset=FIRST_RISER_X - SPAWN_X,
    step_width=STEP_WIDTH,
    step_height=0.13,
    num_steps=NUM_STEPS,
  )
  del nominal

  cfg.rewards["stair_progress"] = RewardTermCfg(
    func=mdp.stair_progress,
    weight=0.25,
    params={"asset_name": "robot"},
  )
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=mdp.stair_feet_clearance,
    weight=-1.0,
    params={
      "action_name": "joint_pos",
      "asset_name": "robot",
      "default_height": 0.10,
      "height_above_tread": 0.05,
    },
  )
  cfg.rewards["swing_foot_force"] = RewardTermCfg(
    func=mdp.swing_foot_force,
    weight=-0.001,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["cbf_dual"] = RewardTermCfg(
    func=mdp.cbf_dual_reward,
    weight=1.0 if use_cbf_reward else 0.0,
    params={"action_name": "joint_pos", "sigma": 0.5},
  )
  return cfg


def g1_stairs_runner_cfg(experiment_name: str):
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.seed = 42
  cfg.save_interval = 50
  return cfg

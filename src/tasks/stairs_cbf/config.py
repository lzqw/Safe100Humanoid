"""G1 stair tasks with and without the CBF-RL dual mechanism."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from src.assets.robots import G1_ACTION_SCALE
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from . import mdp
from .actions import StairCbfJointPositionActionCfg
from .command import JoystickVelocityCommandCfg, StairTargetCommandCfg
from .online import OnlineSafePpoAlgorithmCfg
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
    # Start near-flat and promote toward the target 13 cm stairs. The CBF reads
    # exact per-row riser metadata recovered from the generated flat patches.
    step_height_range=(0.02, 0.155),
    step_width=STEP_WIDTH,
    num_steps=NUM_STEPS,
    first_riser_x=FIRST_RISER_X,
    spawn_x=SPAWN_X,
    flat_patch_sampling={
      "stair_targets": FlatPatchSamplingCfg(
        num_patches=NUM_STEPS + 1,
        patch_radius=0.12,
        max_height_diff=0.02,
      ),
      "stair_risers": FlatPatchSamplingCfg(
        num_patches=NUM_STEPS,
        patch_radius=0.01,
        max_height_diff=1.0,
      ),
    },
  )
  return TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(5.2, 4.0),
    border_width=2.0,
    num_rows=5,
    num_cols=1,
    difficulty_range=(0.0, 1.0),
    sub_terrains={"forward_stairs": stairs},
    add_lights=True,
  )


def _online_terrain_cfg(
  *,
  num_steps: int,
  step_height_range: tuple[float, float],
  step_width: float,
  step_height_profile: tuple[float, ...] | None = None,
  step_width_profile: tuple[float, ...] | None = None,
  curriculum: bool = False,
) -> TerrainGeneratorCfg:
  total_run = (
    sum(step_width_profile)
    if step_width_profile is not None
    else num_steps * step_width
  )
  stairs = ForwardStairsTerrainCfg(
    proportion=1.0,
    step_height_range=step_height_range,
    step_width=step_width,
    num_steps=num_steps,
    first_riser_x=FIRST_RISER_X,
    spawn_x=SPAWN_X,
    top_platform_length=1.2,
    step_height_profile=step_height_profile,
    step_width_profile=step_width_profile,
    flat_patch_sampling={
      "stair_targets": FlatPatchSamplingCfg(
        num_patches=num_steps + 1,
        patch_radius=0.10,
        max_height_diff=0.02,
      ),
      "stair_risers": FlatPatchSamplingCfg(
        num_patches=num_steps,
        patch_radius=0.01,
        max_height_diff=1.0,
      ),
    },
  )
  return TerrainGeneratorCfg(
    seed=42,
    curriculum=curriculum,
    size=(FIRST_RISER_X + total_run + 1.5, 4.0),
    border_width=2.0,
    num_rows=5 if curriculum else 1,
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
  cfg.rewards["dont_wait"] = RewardTermCfg(
    func=mdp.dont_wait,
    weight=-1.0,
    params={
      "command_name": "twist",
      "asset_name": "robot",
      "command_threshold": 0.2,
      "minimum_forward_speed": 0.1,
    },
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


TARGET_RISE_ERRORS = (
  0.000, 0.003, -0.004, 0.002, 0.005, -0.003,
  0.001, -0.005, 0.004, -0.002, 0.003, 0.000,
  -0.004, 0.002, 0.005, -0.001, 0.003, -0.003,
)
TARGET_TREAD_ERRORS = (
  0.000, 0.007, -0.005, 0.010, -0.008, 0.004,
  -0.006, 0.009, -0.003, 0.005, -0.010, 0.002,
  0.008, -0.004, 0.006, -0.007, 0.003, -0.005,
)
TARGET_RISE_PROFILE = tuple(0.145 + error for error in TARGET_RISE_ERRORS)
TARGET_TREAD_PROFILE = tuple(0.330 + error for error in TARGET_TREAD_ERRORS)
NEIGHBOR_RISE_PROFILE = tuple(value + 0.002 for value in TARGET_RISE_PROFILE)
NEIGHBOR_TREAD_PROFILE = tuple(value + 0.005 for value in TARGET_TREAD_PROFILE)


def g1_online_stairs_env_cfg(
  domain: str,
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build one controlled member of the D0--D5 deployment-shift matrix."""
  human_domain = {
    "D2H": "D2",
    "D3H": "D3",
    "D4H": "D4",
    "D5H": "D5",
    "DQH": "DQ",
    "DQNH": "DQN",
    "DQMH": "DQM",
  }
  closed_loop_centering = domain in human_domain
  base_domain = human_domain.get(domain, domain)
  if base_domain not in {
    "D0", "D1", "D2", "D3", "D4", "D5", "DQ", "DQN", "DQM"
  }:
    raise ValueError(f"unknown online stair domain {domain!r}")
  long_stairs = base_domain in {
    "D1", "D3", "D4", "D5", "DQ", "DQN", "DQM"
  }
  joystick = base_domain in {"D2", "D3", "D4", "D5", "DQ", "DQN", "DQM"}
  num_steps = (
    9
    if base_domain in {"DQ", "DQN", "DQM"}
    else (18 if long_stairs else NUM_STEPS)
  )
  step_width = STEP_WIDTH
  height_range = (0.13, 0.13)
  rise_profile = None
  tread_profile = None
  if base_domain == "D4":
    step_width = 0.33
    height_range = (0.145, 0.145)
    rise_profile = TARGET_RISE_PROFILE
    tread_profile = TARGET_TREAD_PROFILE
  elif base_domain == "D5":
    step_width = 0.335
    height_range = (0.147, 0.147)
    rise_profile = NEIGHBOR_RISE_PROFILE
    tread_profile = NEIGHBOR_TREAD_PROFILE
  elif base_domain == "DQN":
    step_width = 0.355
    height_range = (0.132, 0.132)

  cfg = unitree_g1_rough_env_cfg(play=False)
  terrain_generator = _online_terrain_cfg(
    num_steps=num_steps,
    step_height_range=height_range,
    step_width=step_width,
    step_height_profile=rise_profile,
    step_width_profile=tread_profile,
  )
  if base_domain == "DQM":
    # One rollout contains both calibrated quick domains.  All per-riser
    # consumers read generated metadata, so no nominal width is leaked into
    # the actor, reward, CBF, or success condition.
    dq_stairs = terrain_generator.sub_terrains["forward_stairs"]
    terrain_generator.sub_terrains = {
      "dq_stairs": replace(dq_stairs, proportion=0.5),
      "dqn_stairs": replace(
        dq_stairs,
        proportion=0.5,
        step_height_range=(0.132, 0.132),
        step_width=0.355,
      ),
    }
    terrain_generator.num_cols = 2
    terrain_generator.size = (
      FIRST_RISER_X + num_steps * 0.355 + 1.5,
      terrain_generator.size[1],
    )
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=terrain_generator,
    max_init_terrain_level=0,
  )
  cfg.scene.extent = 5.0 if long_stairs else 3.0
  cfg.episode_length_s = (
    35.0
    if base_domain in {"DQ", "DQN", "DQM"}
    else (50.0 if long_stairs else 20.0)
  )
  if long_stairs:
    # Eighteen independently represented stair boxes create more initial
    # contacts than the six-step baseline; size buffers explicitly instead of
    # hiding a MuJoCo-Warp nconmax overflow.
    cfg.sim.nconmax = 160
    cfg.sim.contact_sensor_maxmatch = 1200
    cfg.sim.njmax = max(cfg.sim.njmax, 3000)
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)
  # The target deployment has fixed robot/ground parameters.  Encoder bias is
  # retained as observation noise, while physical DR is deliberately removed.
  cfg.events.pop("foot_friction", None)
  cfg.events.pop("base_com", None)
  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("encoder_bias", None)

  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["actor"].history_length = 5
  cfg.observations["critic"].history_length = 1
  cfg.observations["online_privileged"] = ObservationGroupCfg(
    terms={
      "deployment_state": ObservationTermCfg(
        func=mdp.online_privileged_state,
        params={
          "action_name": "joint_pos",
          "command_name": "twist",
          "asset_name": "robot",
          "delay_queue_length": 9,
        },
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )

  if joystick:
    cfg.commands["twist"] = JoystickVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(60.0, 60.0),
      debug_vis=False,
      patch_name="stair_targets",
      closed_loop_centering=closed_loop_centering,
      stair_half_width=1.20,
    )
  else:
    cfg.commands["twist"] = StairTargetCommandCfg(
      entity_name="robot",
      resampling_time_range=(60.0, 60.0),
      debug_vis=False,
      patch_name="stair_targets",
    )
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.08, 0.08),
    "z": (0.0, 0.0),
    "yaw": (-0.08, 0.08),
  }

  cfg.actions["joint_pos"] = StairCbfJointPositionActionCfg(
    entity_name="robot",
    actuator_names=LOWER_BODY_JOINT_PATTERNS,
    scale={name: G1_ACTION_SCALE[name] for name in LOWER_BODY_JOINT_PATTERNS},
    use_default_offset=True,
    enabled=True,
    contact_sensor_name="feet_ground_contact",
    foot_site_names=("left_foot", "right_foot"),
    alpha=10.0,
    first_riser_offset=FIRST_RISER_X - SPAWN_X,
    step_width=step_width,
    step_height=height_range[0],
    num_steps=num_steps,
    patch_name="stair_targets",
    riser_patch_name="stair_risers",
  )

  cfg.rewards["stair_progress"] = RewardTermCfg(
    func=mdp.IncrementalStairProgress,
    weight=0.25,
    params={"asset_name": "robot", "maximum_forward_velocity": 0.8},
  )
  cfg.rewards["riser_crossing"] = RewardTermCfg(
    func=mdp.RiserCrossingReward,
    weight=0.5,
  )
  cfg.rewards["top_completion"] = RewardTermCfg(
    func=mdp.TopCompletionReward,
    weight=5.0,
    params={"position_tolerance": 0.10},
  )
  cfg.rewards["dont_wait"] = RewardTermCfg(
    func=mdp.dont_wait,
    weight=-1.0,
    params={
      "command_name": "twist",
      "asset_name": "robot",
      "command_threshold": 0.2,
      "minimum_forward_speed": 0.1,
    },
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
    weight=1.0,
    params={"action_name": "joint_pos", "sigma": 0.5},
  )
  # The generic termination penalty would incorrectly punish successful top
  # completion.  Replace it with an explicit fall-only event instead of
  # silently removing the principal rare-failure learning signal.
  cfg.rewards["is_terminated"].weight = 0.0
  cfg.rewards["fall_termination"] = RewardTermCfg(
    func=mdp.fall_termination,
    weight=-200.0,
    params={"termination_name": "fell_over"},
  )
  cfg.terminations["reached_top"] = TerminationTermCfg(
    func=mdp.reached_stair_top,
    params={
      "asset_name": "robot",
      "target_patch_name": "stair_targets",
      "position_tolerance": 0.10,
    },
  )
  return cfg


def g1_online_stairs_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_stairs_online_refinement"
  cfg.run_name = "D4_target_stair_01"
  cfg.seed = 42
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  cfg.save_interval = 1
  cfg.num_steps_per_env = 256
  cfg.max_iterations = 10
  cfg.obs_groups = {
    "actor": ("actor",),
    "critic": ("actor", "critic", "online_privileged"),
  }
  cfg.algorithm = OnlineSafePpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.05,
    entropy_coef=0.0,
    num_learning_epochs=2,
    num_mini_batches=4,
    learning_rate=1.0e-5,
    schedule="fixed",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.003,
    max_grad_norm=0.5,
  )
  return cfg

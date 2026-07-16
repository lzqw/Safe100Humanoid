"""Task registration for the CBF-RL G1 stair reproduction."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .config import (
  g1_online_stairs_env_cfg,
  g1_online_stairs_runner_cfg,
  g1_stairs_env_cfg,
  g1_stairs_runner_cfg,
)
from .online import OnlineSafeRefinementRunner


register_mjlab_task(
  task_id="Unitree-G1-Stairs-Nominal",
  env_cfg=g1_stairs_env_cfg(use_filter=False, use_cbf_reward=False),
  play_env_cfg=g1_stairs_env_cfg(use_filter=False, use_cbf_reward=False, play=True),
  rl_cfg=g1_stairs_runner_cfg("g1_stairs_nominal"),
  runner_cls=VelocityOnPolicyRunner,
)


for _domain in ("D0", "D1", "D2", "D3", "D4", "D5", "DQ", "DQN"):
  register_mjlab_task(
    task_id=f"Unitree-G1-Stairs-Online-{_domain}",
    env_cfg=g1_online_stairs_env_cfg(_domain),
    play_env_cfg=g1_online_stairs_env_cfg(_domain, play=True),
    rl_cfg=g1_online_stairs_runner_cfg(),
    runner_cls=OnlineSafeRefinementRunner,
  )

# Frozen task shapes for evaluating the initial 29-DoF engineering pre-run.
register_mjlab_task(
  task_id="Unitree-G1-Stairs-Engineering29-Nominal",
  env_cfg=g1_stairs_env_cfg(
    use_filter=False, use_cbf_reward=False, paper_spec=False
  ),
  play_env_cfg=g1_stairs_env_cfg(
    use_filter=False, use_cbf_reward=False, play=True, paper_spec=False
  ),
  rl_cfg=g1_stairs_runner_cfg("g1_stairs_nominal"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Stairs-Engineering29-CBF",
  env_cfg=g1_stairs_env_cfg(
    use_filter=True, use_cbf_reward=False, paper_spec=False
  ),
  play_env_cfg=g1_stairs_env_cfg(
    use_filter=True, use_cbf_reward=False, play=True, paper_spec=False
  ),
  rl_cfg=g1_stairs_runner_cfg("g1_stairs_cbf"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Stairs-CBF",
  env_cfg=g1_stairs_env_cfg(use_filter=True, use_cbf_reward=True),
  play_env_cfg=g1_stairs_env_cfg(use_filter=True, use_cbf_reward=False, play=True),
  rl_cfg=g1_stairs_runner_cfg("g1_stairs_cbf"),
  runner_cls=VelocityOnPolicyRunner,
)

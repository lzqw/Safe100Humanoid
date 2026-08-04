"""Geometry-aware deterministic evaluation for online stair refinement."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def evaluate_policy(
  policy,
  *,
  task: str,
  num_envs: int,
  num_episodes: int,
  seed: int,
  device: str,
  runtime_filter: bool = True,
  one_episode_per_env: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  # Reset every RNG used by reset events and command generation so old and
  # candidate policies receive paired initial states and joystick traces.
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  if one_episode_per_env and num_episodes != num_envs:
    raise ValueError(
      "strict paired evaluation requires num_episodes == num_envs so every "
      "initial environment contributes exactly one episode"
    )
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cfg.actions["joint_pos"].enabled = runtime_filter
  base_env = ManagerBasedRlEnv(cfg, device=device)
  agent_cfg = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  # The wrapper does not forward reset kwargs; cfg.seed already fixes the
  # environment/randomization sequence for paired old/candidate evaluation.
  obs, _ = env.reset()
  policy.eval()
  action_term = base_env.action_manager.get_term("joint_pos")
  command_term = base_env.command_manager.get_term("twist")
  n_risers = action_term._edge_x.shape[-1]
  stair_half_width = float(getattr(command_term.cfg, "stair_half_width", 1.20))
  step_dt = float(base_env.step_dt)
  max_episode_steps = int(base_env.max_episode_length)
  batch = torch.arange(num_envs, device=device)
  signature = hashlib.sha256()
  initial_tensors = [
    obs["actor"],
    base_env.scene["robot"].data.root_link_pos_w,
    base_env.scene["robot"].data.root_link_quat_w,
    base_env.scene["robot"].data.joint_pos,
    base_env.command_manager.get_command("twist"),
    getattr(command_term, "raw_command", base_env.command_manager.get_command("twist")),
    getattr(
      command_term,
      "delay_steps",
      torch.zeros(num_envs, dtype=torch.long, device=device),
    ),
    getattr(
      command_term,
      "_delay_queue",
      torch.zeros(num_envs, 1, 3, device=device),
    ),
  ]
  for tensor in initial_tensors:
    signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  initial_state_signature = signature.hexdigest()

  returns = torch.zeros(num_envs, device=device)
  steps = torch.zeros(num_envs, device=device)
  max_riser = torch.zeros(num_envs, dtype=torch.long, device=device)
  min_h = torch.full((num_envs,), torch.inf, device=device)
  interventions = torch.zeros(num_envs, device=device)
  corrections = torch.zeros(num_envs, device=device)
  correction_max = torch.zeros(num_envs, device=device)
  counterfactual_interventions = torch.zeros(num_envs, device=device)
  counterfactual_corrections = torch.zeros(num_envs, device=device)
  counterfactual_correction_max = torch.zeros(num_envs, device=device)
  centerline_error_integral = torch.zeros(num_envs, device=device)
  max_abs_centerline_error = torch.zeros(num_envs, device=device)
  max_abs_heading_error = torch.zeros(num_envs, device=device)
  min_root_edge_clearance = torch.full(
    (num_envs,), torch.inf, device=device
  )
  min_foot_edge_clearance = torch.full(
    (num_envs,), torch.inf, device=device
  )
  operator_correction_steps = torch.zeros(num_envs, device=device)
  geometric_active_steps = torch.zeros(num_envs, device=device)
  nominal_violation_steps = torch.zeros(num_envs, device=device)
  filtered_violation_steps = torch.zeros(num_envs, device=device)
  min_nominal_margin = torch.full((num_envs,), torch.inf, device=device)
  min_filtered_margin = torch.full((num_envs,), torch.inf, device=device)
  correction_history = torch.zeros(
    num_envs, max_episode_steps, device=device
  )
  counterfactual_correction_history = torch.zeros_like(correction_history)
  steps_by_riser = torch.zeros(num_envs, n_risers, device=device)
  interventions_by_riser = torch.zeros_like(steps_by_riser)
  counterfactual_interventions_by_riser = torch.zeros_like(steps_by_riser)
  completed: list[dict[str, Any]] = []
  initial_episode_recorded = torch.zeros(
    num_envs, dtype=torch.bool, device=device
  )

  try:
    with torch.inference_mode():
      while len(completed) < num_episodes:
        actions = policy(obs)
        obs, reward, done, _ = env.step(actions)
        returns += reward
        steps += 1
        root_x = base_env.scene["robot"].data.root_link_pos_w[:, 0:1]
        risers = action_term._edge_x[
          base_env.scene.terrain.terrain_levels,
          base_env.scene.terrain.terrain_types,
        ]
        current_riser = torch.sum(root_x >= risers, dim=1)
        max_riser = torch.maximum(max_riser, current_riser)
        active_riser = current_riser.clamp_max(n_risers - 1)
        before_top = current_riser < n_risers
        steps_by_riser[batch, active_riser] += before_top.float()
        interventions_by_riser[batch, active_riser] += (
          action_term.intervened.float() * before_top.float()
        )
        counterfactual_interventions_by_riser[batch, active_riser] += (
          action_term.would_intervene.float() * before_top.float()
        )
        finite_h = torch.where(
          torch.isfinite(action_term.h),
          action_term.h,
          torch.full_like(action_term.h, torch.inf),
        )
        min_h = torch.minimum(min_h, finite_h)
        interventions += action_term.intervened.float()
        corrections += action_term.target_intervention_norm
        correction_max = torch.maximum(
          correction_max, action_term.target_intervention_norm
        )
        counterfactual_interventions += action_term.would_intervene.float()
        counterfactual_corrections += (
          action_term.counterfactual_target_intervention_norm
        )
        counterfactual_correction_max = torch.maximum(
          counterfactual_correction_max,
          action_term.counterfactual_target_intervention_norm,
        )
        history_index = (steps.long() - 1).clamp(0, max_episode_steps - 1)
        correction_history[batch, history_index] = (
          action_term.target_intervention_norm
        )
        counterfactual_correction_history[batch, history_index] = (
          action_term.counterfactual_target_intervention_norm
        )
        active = action_term.geometric_active
        geometric_active_steps += active.float()
        nominal_violation_steps += (
          active & (action_term.psi_nominal < -action_term.cfg.intervention_epsilon)
        ).float()
        filtered_violation_steps += (
          active & (action_term.psi_filtered < -action_term.cfg.intervention_epsilon)
        ).float()
        nominal_margin = torch.where(
          active, action_term.psi_nominal, torch.full_like(action_term.psi_nominal, torch.inf)
        )
        filtered_margin = torch.where(
          active, action_term.psi_filtered, torch.full_like(action_term.psi_filtered, torch.inf)
        )
        min_nominal_margin = torch.minimum(min_nominal_margin, nominal_margin)
        min_filtered_margin = torch.minimum(min_filtered_margin, filtered_margin)
        centerline_error = getattr(
          command_term,
          "centerline_error",
          torch.zeros(num_envs, device=device),
        )
        heading_error = getattr(
          command_term,
          "heading_error",
          torch.zeros(num_envs, device=device),
        )
        abs_centerline_error = torch.abs(centerline_error)
        centerline_error_integral += abs_centerline_error
        max_abs_centerline_error = torch.maximum(
          max_abs_centerline_error, abs_centerline_error
        )
        max_abs_heading_error = torch.maximum(
          max_abs_heading_error, torch.abs(heading_error)
        )
        min_root_edge_clearance = torch.minimum(
          min_root_edge_clearance,
          stair_half_width - abs_centerline_error,
        )
        patches = base_env.scene.terrain.flat_patches["stair_targets"][
          base_env.scene.terrain.terrain_levels,
          base_env.scene.terrain.terrain_types,
        ]
        center_y = patches[:, 0, 1]
        foot_y = base_env.scene["robot"].data.site_pos_w[
          :, action_term._site_local_ids, 1
        ]
        foot_edge_clearance = stair_half_width - torch.max(
          torch.abs(foot_y - center_y.unsqueeze(1)), dim=1
        ).values
        min_foot_edge_clearance = torch.minimum(
          min_foot_edge_clearance, foot_edge_clearance
        )
        correction_active = getattr(
          command_term,
          "correction_active",
          torch.zeros(num_envs, dtype=torch.bool, device=device),
        )
        operator_correction_steps += correction_active.float()

        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) == 0:
          continue
        record_ids = done_ids
        if one_episode_per_env:
          record_ids = done_ids[~initial_episode_recorded[done_ids]]
        fell_all = base_env.termination_manager.get_term("fell_over")
        timeout_all = base_env.termination_manager.get_term("time_out")
        success_all = base_env.termination_manager.get_term("reached_top")
        for env_id in record_ids.tolist():
          reached = int(max_riser[env_id])
          episode_steps = max(1, int(steps[env_id]))
          correction_p95 = torch.quantile(
            correction_history[env_id, :episode_steps], 0.95
          )
          counterfactual_correction_p95 = torch.quantile(
            counterfactual_correction_history[env_id, :episode_steps], 0.95
          )
          completed.append(
            {
              "episode": len(completed),
              "success": bool(success_all[env_id]),
              "fell": bool(fell_all[env_id]),
              "timed_out": bool(timeout_all[env_id]),
              "return": float(returns[env_id]),
              "steps": int(steps[env_id]),
              "episode_time_s": float(steps[env_id]) * step_dt,
              "max_riser": reached,
              "completion_fraction": reached / n_risers,
              "minimum_cbf_h": (
                None if torch.isinf(min_h[env_id]) else float(min_h[env_id])
              ),
              "intervention_count": int(interventions[env_id]),
              "intervention_per_riser": float(
                interventions[env_id] / max(1, reached)
              ),
              "correction_mean": float(
                corrections[env_id] / max(1.0, float(steps[env_id]))
              ),
              "correction_max": float(correction_max[env_id]),
              "correction_p95": float(correction_p95),
              "would_intervene_count": int(
                counterfactual_interventions[env_id]
              ),
              "would_intervene_per_riser": float(
                counterfactual_interventions[env_id] / max(1, reached)
              ),
              "counterfactual_correction_mean": float(
                counterfactual_corrections[env_id]
                / max(1.0, float(steps[env_id]))
              ),
              "counterfactual_correction_max": float(
                counterfactual_correction_max[env_id]
              ),
              "counterfactual_correction_p95": float(
                counterfactual_correction_p95
              ),
              "geometric_active_fraction": float(
                geometric_active_steps[env_id] / episode_steps
              ),
              "intervention_fraction": float(
                interventions[env_id] / episode_steps
              ),
              "would_intervene_fraction": float(
                counterfactual_interventions[env_id] / episode_steps
              ),
              "nominal_violation_fraction": float(
                nominal_violation_steps[env_id] / episode_steps
              ),
              "filtered_violation_fraction": float(
                filtered_violation_steps[env_id] / episode_steps
              ),
              "minimum_nominal_margin": (
                None
                if torch.isinf(min_nominal_margin[env_id])
                else float(min_nominal_margin[env_id])
              ),
              "minimum_filtered_margin": (
                None
                if torch.isinf(min_filtered_margin[env_id])
                else float(min_filtered_margin[env_id])
              ),
              "steps_by_riser": [
                int(value) for value in steps_by_riser[env_id].tolist()
              ],
              "interventions_by_riser": [
                int(value) for value in interventions_by_riser[env_id].tolist()
              ],
              "would_interventions_by_riser": [
                int(value)
                for value in counterfactual_interventions_by_riser[env_id].tolist()
              ],
              "mean_abs_centerline_error": float(
                centerline_error_integral[env_id]
                / max(1.0, float(steps[env_id]))
              ),
              "max_abs_centerline_error": float(
                max_abs_centerline_error[env_id]
              ),
              "max_abs_heading_error": float(max_abs_heading_error[env_id]),
              "minimum_root_edge_clearance": float(
                min_root_edge_clearance[env_id]
              ),
              "minimum_foot_edge_clearance": float(
                min_foot_edge_clearance[env_id]
              ),
              "operator_correction_fraction": float(
                operator_correction_steps[env_id]
                / max(1.0, float(steps[env_id]))
              ),
              "side_edge_breach": bool(
                (min_root_edge_clearance[env_id] < 0.0)
                | (min_foot_edge_clearance[env_id] < 0.0)
              ),
            }
          )
          initial_episode_recorded[env_id] = True
          if len(completed) >= num_episodes:
            break
        returns[done_ids] = 0.0
        steps[done_ids] = 0.0
        max_riser[done_ids] = 0
        min_h[done_ids] = torch.inf
        interventions[done_ids] = 0.0
        corrections[done_ids] = 0.0
        correction_max[done_ids] = 0.0
        counterfactual_interventions[done_ids] = 0.0
        counterfactual_corrections[done_ids] = 0.0
        counterfactual_correction_max[done_ids] = 0.0
        centerline_error_integral[done_ids] = 0.0
        max_abs_centerline_error[done_ids] = 0.0
        max_abs_heading_error[done_ids] = 0.0
        min_root_edge_clearance[done_ids] = torch.inf
        min_foot_edge_clearance[done_ids] = torch.inf
        operator_correction_steps[done_ids] = 0.0
        geometric_active_steps[done_ids] = 0.0
        nominal_violation_steps[done_ids] = 0.0
        filtered_violation_steps[done_ids] = 0.0
        min_nominal_margin[done_ids] = torch.inf
        min_filtered_margin[done_ids] = torch.inf
        correction_history[done_ids] = 0.0
        counterfactual_correction_history[done_ids] = 0.0
        steps_by_riser[done_ids] = 0.0
        interventions_by_riser[done_ids] = 0.0
        counterfactual_interventions_by_riser[done_ids] = 0.0
  finally:
    env.close()

  completed = completed[:num_episodes]
  survival = {}
  hazard = {}
  for k in range(1, n_risers + 1):
    reached_k = sum(int(row["max_riser"]) >= k for row in completed)
    reached_previous = sum(int(row["max_riser"]) >= k - 1 for row in completed)
    failed_at_k = sum(
      (not bool(row["success"])) and int(row["max_riser"]) == k - 1
      for row in completed
    )
    survival[str(k)] = reached_k / len(completed)
    hazard[str(k)] = failed_at_k / max(1, reached_previous)

  finite_h_values = [
    float(row["minimum_cbf_h"])
    for row in completed
    if row["minimum_cbf_h"] is not None
  ]
  successful_times = [
    float(row["episode_time_s"]) for row in completed if bool(row["success"])
  ]
  finite_nominal_margins = [
    float(row["minimum_nominal_margin"])
    for row in completed
    if row["minimum_nominal_margin"] is not None
  ]
  finite_filtered_margins = [
    float(row["minimum_filtered_margin"])
    for row in completed
    if row["minimum_filtered_margin"] is not None
  ]
  per_riser = {}
  for riser in range(n_risers):
    step_count = sum(int(row["steps_by_riser"][riser]) for row in completed)
    actual_count = sum(
      int(row["interventions_by_riser"][riser]) for row in completed
    )
    would_count = sum(
      int(row["would_interventions_by_riser"][riser]) for row in completed
    )
    reached_count = sum(int(row["max_riser"]) >= riser for row in completed)
    per_riser[str(riser + 1)] = {
      "step_count": step_count,
      "reached_episode_count": reached_count,
      "intervention_count": actual_count,
      "would_intervene_count": would_count,
      "intervention_fraction": actual_count / max(1, step_count),
      "would_intervene_fraction": would_count / max(1, step_count),
      "intervention_per_reached_episode": actual_count / max(1, reached_count),
      "would_intervene_per_reached_episode": would_count / max(1, reached_count),
    }
  summary = {
    "task": task,
    "seed": seed,
    "num_envs": num_envs,
    "num_episodes": len(completed),
    "num_risers": n_risers,
    "runtime_filter": runtime_filter,
    "deterministic_policy_mean": True,
    "one_initial_episode_per_env": one_episode_per_env,
    "initial_state_signature": initial_state_signature,
    "success_rate": sum(bool(row["success"]) for row in completed) / len(completed),
    "fall_rate": sum(bool(row["fell"]) for row in completed) / len(completed),
    "timeout_rate": sum(bool(row["timed_out"]) for row in completed) / len(completed),
    "mean_reached_riser": sum(int(row["max_riser"]) for row in completed) / len(completed),
    "mean_return": sum(float(row["return"]) for row in completed) / len(completed),
    "mean_episode_time_s": sum(float(row["episode_time_s"]) for row in completed)
    / len(completed),
    "mean_success_time_s": (
      sum(successful_times) / len(successful_times) if successful_times else None
    ),
    "intervention_per_riser": sum(
      float(row["intervention_per_riser"]) for row in completed
    ) / len(completed),
    "correction_mean": sum(float(row["correction_mean"]) for row in completed) / len(completed),
    "mean_correction_p95": sum(
      float(row["correction_p95"]) for row in completed
    ) / len(completed),
    "would_intervene_per_riser": sum(
      float(row["would_intervene_per_riser"]) for row in completed
    ) / len(completed),
    "counterfactual_correction_mean": sum(
      float(row["counterfactual_correction_mean"]) for row in completed
    ) / len(completed),
    "mean_counterfactual_correction_p95": sum(
      float(row["counterfactual_correction_p95"]) for row in completed
    ) / len(completed),
    "geometric_active_fraction": sum(
      float(row["geometric_active_fraction"]) for row in completed
    ) / len(completed),
    "intervention_fraction": sum(
      float(row["intervention_fraction"]) for row in completed
    ) / len(completed),
    "would_intervene_fraction": sum(
      float(row["would_intervene_fraction"]) for row in completed
    ) / len(completed),
    "nominal_violation_fraction": sum(
      float(row["nominal_violation_fraction"]) for row in completed
    ) / len(completed),
    "filtered_violation_fraction": sum(
      float(row["filtered_violation_fraction"]) for row in completed
    ) / len(completed),
    "minimum_nominal_margin": (
      min(finite_nominal_margins) if finite_nominal_margins else None
    ),
    "minimum_filtered_margin": (
      min(finite_filtered_margins) if finite_filtered_margins else None
    ),
    "mean_abs_centerline_error": sum(
      float(row["mean_abs_centerline_error"]) for row in completed
    ) / len(completed),
    "mean_max_abs_centerline_error": sum(
      float(row["max_abs_centerline_error"]) for row in completed
    ) / len(completed),
    "mean_max_abs_heading_error": sum(
      float(row["max_abs_heading_error"]) for row in completed
    ) / len(completed),
    "minimum_root_edge_clearance": min(
      float(row["minimum_root_edge_clearance"]) for row in completed
    ),
    "minimum_foot_edge_clearance": min(
      float(row["minimum_foot_edge_clearance"]) for row in completed
    ),
    "operator_correction_fraction": sum(
      float(row["operator_correction_fraction"]) for row in completed
    ) / len(completed),
    "side_edge_breach_rate": sum(
      bool(row["side_edge_breach"]) for row in completed
    ) / len(completed),
    "side_fall_rate": sum(
      bool(row["side_edge_breach"]) and bool(row["fell"])
      for row in completed
    ) / len(completed),
    "minimum_cbf_h": min(finite_h_values) if finite_h_values else None,
    "survival_curve": survival,
    "conditional_failure_hazard": hazard,
    "per_riser_cbf": per_riser,
  }
  return summary, completed


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--num-episodes", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--runtime-filter", choices=("on", "off"), default="on")
  parser.add_argument(
    "--one-episode-per-env",
    action="store_true",
    help="Record exactly the initial episode of every environment for strict pairing.",
  )
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--output-csv", type=Path, required=True)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo.resolve()))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(args.device)
  env.close()

  summary, episodes = evaluate_policy(
    policy,
    task=args.task,
    num_envs=args.num_envs,
    num_episodes=args.num_episodes,
    seed=args.seed,
    device=args.device,
    runtime_filter=args.runtime_filter == "on",
    one_episode_per_env=args.one_episode_per_env,
  )
  summary["checkpoint"] = str(args.checkpoint)
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  with args.output_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
    writer.writeheader()
    writer.writerows(episodes)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()

"""Transactional CBF-protected PPO refinement on a fixed deployment stair."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch


def _actor_state(actor) -> dict[str, torch.Tensor]:
  return {key: value.detach().clone() for key, value in actor.state_dict().items()}


def _actor_state_sha256(state: dict[str, torch.Tensor]) -> str:
  """Hash exactly the inference state, excluding critic/optimizer payloads."""
  digest = hashlib.sha256()
  for name in sorted(state):
    value = state[name].detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
  return digest.hexdigest()


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _evaluate_state(
  runner,
  actor_state: dict[str, torch.Tensor],
  *,
  domains: tuple[str, ...],
  num_envs: int,
  num_episodes: int,
  seed: int,
  device: str,
  repeats: int = 1,
  runtime_filter: bool = True,
  artifact_dir: Path | None = None,
  resume: bool = False,
  deployment_context: Path | None = None,
  v19_context: Path | None = None,
  telemetry_env_id: int | None = None,
) -> dict[str, dict[str, Any]]:
  """Evaluate one actor in isolated CUDA processes.

  MuJoCo-Warp CUDA graphs are not reliably reusable after dozens of
  create/close cycles in one Python process (observed as capture error 901).
  The training environment remains in this parent process, while each paired
  replicate gets a fresh subprocess and therefore a fresh Warp context.
  """
  if num_envs != num_episodes:
    raise ValueError(
      "transactional paired evaluation requires --eval-num-envs equal to "
      "--eval-num-episodes; recycled environments are not independent pairs"
    )
  output: dict[str, dict[str, Any]] = {}
  repo = Path(__file__).resolve().parents[2]
  context_hash = None
  v19_mode = None
  actor_state_sha256 = _actor_state_sha256(actor_state)
  if deployment_context is not None:
    from src.tasks.stairs_cbf.deployment_context import (
      OBSERVABLE_SPECIALIST_CONTEXT_KINDS,
      load_frozen_deployment_context,
    )

    deployment_context = deployment_context.resolve()
    loaded_context = load_frozen_deployment_context(deployment_context)
    context_hash = loaded_context["parameters_sha256"]
    if loaded_context.get("kind") in OBSERVABLE_SPECIALIST_CONTEXT_KINDS:
      v19_context = deployment_context
      v19_mode = loaded_context["specialist_mode"]
  if v19_context is not None:
    from src.tasks.stairs_cbf.deployment_context import (
      OBSERVABLE_SPECIALIST_CONTEXT_KINDS,
      load_frozen_deployment_context,
    )

    v19_context = v19_context.resolve()
    loaded_v19 = load_frozen_deployment_context(v19_context)
    if loaded_v19.get("kind") not in OBSERVABLE_SPECIALIST_CONTEXT_KINDS:
      raise ValueError(
        "--v19-context must identify an observable v19/v21/v22 context"
      )
    v19_mode = loaded_v19["specialist_mode"]
  checkpoint_payload = runner.alg.save()
  checkpoint_payload["actor_state_dict"] = {
    key: value.detach().cpu() for key, value in actor_state.items()
  }
  # MjlabOnPolicyRunner.load() always returns checkpoint metadata even when
  # load_cfg requests the actor only.
  checkpoint_payload.setdefault("iter", 0)
  checkpoint_payload.setdefault("infos", {})
  temp_context = (
    tempfile.TemporaryDirectory(prefix="stairs-paired-eval-")
    if artifact_dir is None
    else nullcontext(str(artifact_dir.resolve()))
  )
  with temp_context as temp_dir:
    temp_root = Path(temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)
    checkpoint = temp_root / "actor.pt"
    torch.save(checkpoint_payload, checkpoint)
    for domain in domains:
      from src.tasks.stairs_cbf.deployment_context import (
        deployment_context_role_for_task,
      )

      context_role = deployment_context_role_for_task(domain)
      if context_role is not None and deployment_context is None:
        raise ValueError(f"medium domain {domain} requires a deployment context")
      replicate_summaries = []
      telemetry_files: list[str] = []
      for repeat in range(repeats):
        stem = f"{domain}-seed{seed + repeat}"
        output_json = temp_root / f"{stem}.json"
        output_csv = temp_root / f"{stem}.csv"
        telemetry_csv = temp_root / f"{stem}-inline-telemetry.csv"
        if resume and output_json.is_file() and output_csv.is_file():
          try:
            summary = json.loads(output_json.read_text())
          except (json.JSONDecodeError, OSError):
            summary = None
          if (
            isinstance(summary, dict)
            and summary.get("task") == f"Unitree-G1-Stairs-Online-{domain}"
            and int(summary.get("seed", -1)) == seed + repeat
            and int(summary.get("num_episodes", -1)) == num_episodes
            and summary.get("runtime_filter") is runtime_filter
            and summary.get("actor_state_sha256") == actor_state_sha256
            and (
              telemetry_env_id is None
              or (
                telemetry_csv.is_file()
                and summary.get(
                  "mechanism_telemetry_same_rollout_outcome_bound"
                )
                is True
              )
            )
            and (
              context_role is None
              or summary.get("deployment_context", {}).get(
                "parameters_sha256"
              )
              == context_hash
            )
          ):
            replicate_summaries.append(summary)
            if telemetry_env_id is not None:
              telemetry_files.append(str(telemetry_csv))
            continue
        command = [
          sys.executable,
          str(repo / "experiments/scripts/evaluate_online_stairs.py"),
          "--repo",
          str(repo),
          "--task",
          f"Unitree-G1-Stairs-Online-{domain}",
          "--checkpoint",
          str(checkpoint),
          "--num-envs",
          str(num_envs),
          "--num-episodes",
          str(num_episodes),
          "--seed",
          str(seed + repeat),
          "--device",
          device,
          "--runtime-filter",
          "on" if runtime_filter else "off",
          "--one-episode-per-env",
          "--output-json",
          str(output_json),
          "--output-csv",
          str(output_csv),
        ]
        if v19_context is not None:
          command.extend(("--v19-context", str(v19_context)))
        if context_role is not None:
          assert deployment_context is not None
          command.extend(
            ("--deployment-context", str(deployment_context))
          )
        if telemetry_env_id is not None:
          command.extend(
            (
              "--telemetry-env-id",
              str(telemetry_env_id),
              "--telemetry-output-csv",
              str(telemetry_csv),
            )
          )
        completed = subprocess.run(
          command,
          cwd=repo,
          check=False,
          capture_output=True,
          text=True,
        )
        if completed.returncode != 0:
          diagnostic = "\n".join(
            (completed.stdout + "\n" + completed.stderr).splitlines()[-120:]
          )
          raise RuntimeError(
            f"isolated paired evaluation failed for {stem}:\n{diagnostic}"
          )
        summary = json.loads(output_json.read_text())
        if summary.get("actor_state_sha256") != actor_state_sha256:
          raise RuntimeError(
            f"isolated evaluation loaded a different actor for {stem}"
          )
        replicate_summaries.append(summary)
        if telemetry_env_id is not None:
          telemetry_files.append(str(telemetry_csv))
      aggregate: dict[str, Any] = {
        "task": f"Unitree-G1-Stairs-Online-{domain}",
        "num_episodes": num_episodes * repeats,
        "repeats": repeats,
        "seeds": [seed + repeat for repeat in range(repeats)],
        "replicates": replicate_summaries,
        "runtime_filter": runtime_filter,
        "paired_one_initial_episode_per_env": True,
        "initial_state_signatures": [
          summary["initial_state_signature"] for summary in replicate_summaries
        ],
        "actor_state_sha256": actor_state_sha256,
        "v19_specialist_mode": v19_mode,
        "inline_telemetry_environment_id_per_batch": telemetry_env_id,
        "inline_telemetry_files": telemetry_files,
        "inline_telemetry_same_rollout_outcome_bound": (
          telemetry_env_id is not None
          and len(telemetry_files) == len(replicate_summaries)
        ),
      }
      for key in (
        "success_rate",
        "fall_rate",
        "timeout_rate",
        "mean_reached_riser",
        "mean_return",
        "mean_episode_time_s",
        "intervention_per_riser",
        "correction_mean",
        "mean_correction_p95",
        "would_intervene_per_riser",
        "counterfactual_correction_mean",
        "mean_counterfactual_correction_p95",
        "geometric_active_fraction",
        "intervention_fraction",
        "would_intervene_fraction",
        "nominal_violation_fraction",
        "filtered_violation_fraction",
        "mean_maximum_roll_signal",
        "mean_maximum_pitch_signal",
        "mean_maximum_angular_velocity_signal",
        "mean_slip_signal",
        "mean_contact_mismatch_fraction",
      ):
        values = [float(summary[key]) for summary in replicate_summaries]
        aggregate[key] = sum(values) / len(values)
        aggregate[f"{key}_std"] = (
          math.sqrt(sum((value - aggregate[key]) ** 2 for value in values) / (len(values) - 1))
          if len(values) > 1
          else 0.0
        )
      successful_times = [
        float(summary["mean_success_time_s"])
        for summary in replicate_summaries
        if summary["mean_success_time_s"] is not None
      ]
      aggregate["mean_success_time_s"] = (
        sum(successful_times) / len(successful_times)
        if successful_times
        else None
      )
      for key in (
        "minimum_cbf_h",
        "minimum_nominal_margin",
        "minimum_filtered_margin",
      ):
        finite_values = [
          float(summary[key])
          for summary in replicate_summaries
          if summary[key] is not None
        ]
        aggregate[key] = min(finite_values) if finite_values else None
      failure_type_counts = {
        failure_type: sum(
          int(summary.get("failure_type_counts", {}).get(failure_type, 0))
          for summary in replicate_summaries
        )
        for failure_type in (
          "lateral_heading_drift",
          "contact_stability",
          "non_lateral_high_cbf_demand",
          "non_lateral_balance_or_phase",
          "other_non_lateral",
        )
      }
      fall_count = sum(failure_type_counts.values())
      aggregate["failure_type_counts"] = failure_type_counts
      aggregate["failure_type_fractions"] = {
        key: value / max(1, fall_count)
        for key, value in failure_type_counts.items()
      }
      output[domain] = aggregate
  return output


def _total_actor_kl(
  runner,
  base_state: dict[str, torch.Tensor],
  actor_state: dict[str, torch.Tensor] | None = None,
) -> float:
  obs, _, _, _ = runner.alg.latest_policy_evaluation_data()
  current_state = _actor_state(runner.alg.actor)
  evaluated_state = current_state if actor_state is None else actor_state
  with torch.no_grad():
    runner.alg.actor.load_state_dict(base_state, strict=True)
    base_mean = runner.alg.actor(obs).detach().clone()
    base_std = runner.alg.actor.distribution.std_param.detach().clone()
    runner.alg.actor.load_state_dict(evaluated_state, strict=True)
    candidate_mean = runner.alg.actor(obs).detach()
    # Cross-round drift constrains the deterministic deployment behavior.  A
    # separately bounded/reduced exploration std must not by itself exhaust
    # the mean-policy KL budget after a rejected candidate.
    kl = 0.5 * torch.sum(
      ((candidate_mean - base_mean) / base_std.clamp_min(1.0e-6)) ** 2,
      dim=-1,
    ).mean()
    runner.alg.actor.load_state_dict(current_state, strict=True)
  return float(kl)


def _policy_step_metrics(
  runner,
  actor_state: dict[str, torch.Tensor],
  base_metrics: dict[str, Any],
) -> dict[str, Any]:
  """Recompute PPO precheck metrics for a candidate-family actor state."""
  observations, actions, old_log_prob, old_params = (
    runner.alg.latest_policy_evaluation_data()
  )
  old_log_prob = old_log_prob.squeeze(-1)
  current_state = _actor_state(runner.alg.actor)
  with torch.no_grad():
    runner.alg.actor.load_state_dict(actor_state, strict=True)
    runner.alg.actor(observations, stochastic_output=True)
    new_log_prob = runner.alg.actor.get_output_log_prob(actions)
    new_params = tuple(runner.alg.actor.output_distribution_params)
    ratio = torch.exp(new_log_prob - old_log_prob)
    metrics = dict(base_metrics)
    metrics.update(
      mean_kl=float(
        runner.alg.actor.get_kl_divergence(old_params, new_params).mean()
      ),
      clip_fraction=float(
        (torch.abs(ratio - 1.0) > runner.alg.clip_param).float().mean()
      ),
      action_saturation_fraction=float(
        (runner.alg.actor.output_mean.abs() > 0.95).float().mean()
      ),
    )
    metrics.update(runner.alg.retention_anchor_kl_metrics())
    runner.alg.actor.load_state_dict(current_state, strict=True)
  return metrics


def _collect_and_update(
  runner,
  obs,
  *,
  critic_only: bool,
  hard_case_bank,
  hard_case_fraction: float,
  neighbor_command_fraction: float,
  neighbor_forward_scale_range: tuple[float, float],
  neighbor_delay_step_offset_range: tuple[int, int],
  hard_case_pre_steps: int,
  hard_case_generator: torch.Generator,
  persistent_hard_case_slots: bool = False,
  late_failure_hard_cases: bool = False,
  late_failure_minimum_steps: int = 50,
  late_failure_maximum_steps: int = 150,
  late_failure_minimum_riser: int = 5,
  dominant_failure_type: str | None = None,
):
  from rsl_rl.utils import check_nan
  from src.tasks.stairs_cbf.hard_cases import (
    capture_hard_case_state,
    classify_target_failure_mode,
    hard_case_state_shape_mismatches,
    MIXED_FAILURE_TYPE,
    reset_rollout_with_hard_cases,
    restore_hard_case_state,
    select_late_failure_candidate,
  )

  del obs
  runner.alg.set_critic_only(critic_only)
  runner.alg.clear_cbf_rollout()
  runner.alg.train_mode()
  obs, start_metrics = reset_rollout_with_hard_cases(
    runner.env,
    hard_case_bank,
    hard_case_fraction=hard_case_fraction,
    neighbor_command_fraction=neighbor_command_fraction,
    neighbor_forward_scale_range=neighbor_forward_scale_range,
    neighbor_delay_step_offset_range=neighbor_delay_step_offset_range,
    generator=hard_case_generator,
  )
  hard_active = torch.zeros(
    runner.env.num_envs, dtype=torch.bool, device=runner.env.device
  )
  neighbor_active = torch.zeros_like(hard_active)
  hard_ids = torch.tensor(
    start_metrics["hard_case_start_ids"],
    dtype=torch.long,
    device=runner.env.device,
  )
  neighbor_ids = torch.tensor(
    start_metrics["neighbor_command_start_ids"],
    dtype=torch.long,
    device=runner.env.device,
  )
  if len(hard_ids) > 0:
    hard_active[hard_ids] = True
  if len(neighbor_ids) > 0:
    neighbor_active[neighbor_ids] = True
  hard_slots = hard_active.clone()
  normal_active = ~(hard_active | neighbor_active)
  normal_completed = 0
  normal_falls = 0
  normal_successes = 0
  history_steps = (
    late_failure_maximum_steps
    if late_failure_hard_cases
    else hard_case_pre_steps
  )
  state_history = deque(maxlen=history_steps + 1)
  riser_history = deque(maxlen=history_steps + 1)
  centerline_history = deque(maxlen=history_steps + 1)
  heading_history = deque(maxlen=history_steps + 1)
  correction_history = deque(maxlen=history_steps + 1)
  episode_side_edge_breach = torch.zeros(
    runner.env.num_envs, dtype=torch.bool, device=runner.env.device
  )
  episode_max_abs_centerline_error = torch.zeros(
    runner.env.num_envs, device=runner.env.device
  )
  episode_max_abs_heading_error = torch.zeros_like(
    episode_max_abs_centerline_error
  )
  episode_correction_max = torch.zeros_like(episode_max_abs_centerline_error)
  valid_steps = torch.zeros(
    runner.env.num_envs, dtype=torch.long, device=runner.env.device
  )
  previous_intervention = torch.zeros(
    runner.env.num_envs, dtype=torch.bool, device=runner.env.device
  )
  bank_added = 0
  target_falls_seen = 0
  target_failure_type_counts: dict[str, int] = {}
  dominant_failure_type_rejected = 0
  late_failure_candidates_found = 0
  hard_case_restart_count = 0
  # Use no_grad rather than inference_mode because critic normalization is
  # intentionally updated and must remain rollback-compatible.
  with torch.no_grad():
    for _ in range(runner.cfg["num_steps_per_env"]):
      unwrapped = runner.env.unwrapped
      state_history.append(capture_hard_case_state(unwrapped))
      action_term = unwrapped.action_manager.get_term("joint_pos")
      command_term = unwrapped.command_manager.get_term("twist")
      terrain = unwrapped.scene.terrain
      assert terrain is not None
      edge_x = action_term._edge_x[
        terrain.terrain_levels, terrain.terrain_types
      ]
      root_x = unwrapped.scene["robot"].data.root_link_pos_w[:, 0:1]
      riser_history.append(torch.sum(root_x >= edge_x, dim=1).detach().clone())
      centerline_history.append(
        getattr(
          command_term,
          "centerline_error",
          torch.zeros(runner.env.num_envs, device=runner.env.device),
        ).detach().clone()
      )
      heading_history.append(
        getattr(
          command_term,
          "heading_error",
          torch.zeros(runner.env.num_envs, device=runner.env.device),
        ).detach().clone()
      )
      correction_history.append(
        action_term.target_intervention_norm.detach().clone()
      )
      centerline_error = centerline_history[-1]
      heading_error = heading_history[-1]
      abs_centerline_error = centerline_error.abs()
      episode_max_abs_centerline_error = torch.maximum(
        episode_max_abs_centerline_error, abs_centerline_error
      )
      episode_max_abs_heading_error = torch.maximum(
        episode_max_abs_heading_error, heading_error.abs()
      )
      stair_half_width = float(
        getattr(command_term.cfg, "stair_half_width", 1.20)
      )
      patches = terrain.flat_patches["stair_targets"][
        terrain.terrain_levels, terrain.terrain_types
      ]
      center_y = patches[:, 0, 1]
      foot_y = unwrapped.scene["robot"].data.site_pos_w[
        :, action_term._site_local_ids, 1
      ]
      foot_edge_breach = torch.max(
        torch.abs(foot_y - center_y.unsqueeze(1)), dim=1
      ).values >= stair_half_width
      episode_side_edge_breach |= (
        abs_centerline_error >= stair_half_width
      ) | foot_edge_breach
      actions = runner.alg.act(obs)
      obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
      check_nan(obs, rewards, dones)
      extras = dict(extras)
      extras["online_hard_case_transition"] = hard_active.detach().clone()
      actual_intervention = extras.get("cbf_intervened")
      magnitude = extras.get("cbf_intervention_magnitude")
      riser_index = extras.get("online_stair_index")
      if magnitude is not None:
        episode_correction_max = torch.maximum(
          episode_correction_max, magnitude
        )
      if (
        not late_failure_hard_cases
        and
        actual_intervention is not None
        and magnitude is not None
        and riser_index is not None
        and len(state_history) == hard_case_pre_steps + 1
      ):
        event = actual_intervention.bool() & ~previous_intervention
        eligible = event & (valid_steps >= hard_case_pre_steps)
        event_ids = eligible.nonzero(as_tuple=False).flatten()
        if len(event_ids) > 0:
          bank_added += hard_case_bank.add_batched(
            state_history[0],
            event_ids,
            magnitude[event_ids],
            riser_index[event_ids],
          )
      done_mask = dones.bool()
      timeouts = extras.get(
        "time_outs", torch.zeros_like(done_mask, dtype=torch.bool)
      ).bool()
      fell = extras.get(
        "online_fell", torch.zeros_like(done_mask, dtype=torch.bool)
      ).bool()
      if late_failure_hard_cases:
        fall_ids = fell.nonzero(as_tuple=False).flatten()
        target_falls_seen += len(fall_ids)
        if len(fall_ids) > 0:
          states = list(state_history)
          risers = list(riser_history)
          centerline = list(centerline_history)
          heading = list(heading_history)
          corrections = list(correction_history)
          for env_id in fall_ids.tolist():
            failure_type = classify_target_failure_mode(
              side_edge_breach=bool(episode_side_edge_breach[env_id]),
              max_abs_centerline_error=float(
                episode_max_abs_centerline_error[env_id]
              ),
              max_abs_heading_error=float(
                episode_max_abs_heading_error[env_id]
              ),
              correction_max=float(episode_correction_max[env_id]),
              stair_half_width=stair_half_width,
            )
            target_failure_type_counts[failure_type] = (
              target_failure_type_counts.get(failure_type, 0) + 1
            )
            if (
              dominant_failure_type is not None
              and failure_type != dominant_failure_type
            ):
              dominant_failure_type_rejected += 1
              continue
            episode_steps = min(int(valid_steps[env_id]) + 1, len(states))
            if episode_steps <= late_failure_minimum_steps:
              continue
            start = len(states) - episode_steps
            candidate = select_late_failure_candidate(
              torch.stack(risers[start:])[:, env_id],
              torch.stack(centerline[start:])[:, env_id],
              torch.stack(heading[start:])[:, env_id],
              torch.stack(corrections[start:])[:, env_id],
              minimum_steps_before_fall=late_failure_minimum_steps,
              maximum_steps_before_fall=late_failure_maximum_steps,
              minimum_riser=late_failure_minimum_riser,
              failure_type=(
                dominant_failure_type
                if dominant_failure_type is not None
                else MIXED_FAILURE_TYPE
              ),
            )
            if candidate is None:
              continue
            late_failure_candidates_found += 1
            bank_added += hard_case_bank.add_late_failure(
              states[start + candidate.history_index], env_id, candidate
            )
      completed_normal = done_mask & normal_active & ~timeouts
      normal_completed += int(completed_normal.sum())
      normal_falls += int((completed_normal & fell).sum())
      normal_successes += int((completed_normal & ~fell).sum())
      if persistent_hard_case_slots:
        restart_ids = (done_mask & hard_slots).nonzero(as_tuple=False).flatten()
        if len(restart_ids) > 0:
          if len(hard_case_bank) < len(restart_ids):
            raise RuntimeError("hard-case bank cannot refill persistent slots")
          replay = hard_case_bank.sample(
            len(restart_ids),
            device=runner.env.device,
            generator=hard_case_generator,
          )
          unwrapped = runner.env.unwrapped
          mismatches = hard_case_state_shape_mismatches(
            capture_hard_case_state(unwrapped), replay
          )
          if mismatches:
            raise RuntimeError(
              "persistent hard-case replay became incompatible: "
              + "; ".join(mismatches[:5])
            )
          restore_hard_case_state(unwrapped, replay, restart_ids)
          unwrapped.scene.write_data_to_sim()
          unwrapped.sim.forward()
          for sensor in unwrapped.scene.sensors.values():
            sensor._invalidate_cache()
          unwrapped.sim.sense()
          unwrapped.observation_manager._obs_buffer = None
          unwrapped.obs_buf = unwrapped.observation_manager.compute(
            update_history=False
          )
          obs = runner.env.get_observations()
          hard_case_restart_count += len(restart_ids)
      previous_intervention = torch.where(
        done_mask,
        torch.zeros_like(previous_intervention),
        actual_intervention.bool()
        if actual_intervention is not None
        else torch.zeros_like(previous_intervention),
      )
      valid_steps = torch.where(
        done_mask, torch.zeros_like(valid_steps), valid_steps + 1
      )
      episode_side_edge_breach[done_mask] = False
      episode_max_abs_centerline_error[done_mask] = 0.0
      episode_max_abs_heading_error[done_mask] = 0.0
      episode_correction_max[done_mask] = 0.0
      obs = obs.to(runner.device)
      rewards = rewards.to(runner.device)
      dones = dones.to(runner.device)
      runner.alg.process_env_step(obs, rewards, dones, extras)
      # v14 keeps fixed hard-case slots throughout the rollout so the actor
      # batch has the advertised 80/20 transition mixture. Legacy protocols
      # keep their original one-shot hard reset semantics.
      if persistent_hard_case_slots:
        hard_active.copy_(hard_slots)
      else:
        hard_active &= ~done_mask
      neighbor_active &= ~done_mask
      if persistent_hard_case_slots:
        normal_active = ~(hard_active | neighbor_active)
      else:
        normal_active = torch.where(
          done_mask, torch.ones_like(normal_active), normal_active
        )
    credit_metrics = runner.alg.relabel_pre_intervention_costs()
    fall_credit_metrics = (
      runner.alg.redistribute_failure_focused_fall_penalty()
      if runner.alg.failure_focused_refinement
      else {}
    )
    completion_metrics = {
      "normal_start_completed_episode_count": normal_completed,
      "normal_start_success_count": normal_successes,
      "normal_start_fall_count": normal_falls,
      "normal_start_success_rate": normal_successes / max(1, normal_completed),
      "normal_start_fall_rate": normal_falls / max(1, normal_completed),
    }
    runner.alg.last_update_metrics.update(completion_metrics)
    runner.alg.compute_returns(obs)
    multiplier_metrics = (
      runner.alg.update_cost_multipliers()
      if runner.alg.task_first_constrained and not critic_only
      else {}
    )
    advantage_metrics = runner.alg.prepare_constrained_advantages()
  losses = runner.alg.update()
  losses.update(credit_metrics)
  losses.update(fall_credit_metrics)
  losses.update(advantage_metrics)
  losses.update(multiplier_metrics)
  losses.update(completion_metrics)
  losses.update(start_metrics)
  losses.update(
    {
      "hard_case_bank_added": bank_added,
      "hard_case_bank_size_after_rollout": len(hard_case_bank),
      "hard_case_bank_total_events": hard_case_bank.total_added,
      "hard_case_pre_steps": hard_case_pre_steps,
      "persistent_hard_case_slots": persistent_hard_case_slots,
      "hard_case_restart_count": hard_case_restart_count,
      "late_failure_hard_cases": late_failure_hard_cases,
      "late_failure_target_falls_seen": target_falls_seen,
      "target_failure_type_counts": target_failure_type_counts,
      "dominant_failure_type": dominant_failure_type,
      "dominant_failure_type_rejected": dominant_failure_type_rejected,
      "late_failure_candidates_found": late_failure_candidates_found,
      "late_failure_minimum_steps": late_failure_minimum_steps,
      "late_failure_maximum_steps": late_failure_maximum_steps,
      "late_failure_minimum_riser": late_failure_minimum_riser,
      "hard_case_bank_audit": hard_case_bank.audit_metadata(),
    }
  )
  return obs, losses


def _actor_observation_batch(obs) -> torch.Tensor:
  """Extract the actor-visible observation without admitting critic state."""
  if isinstance(obs, torch.Tensor):
    return obs
  if isinstance(obs, dict) or callable(getattr(obs, "get", None)):
    for key in ("policy", "actor", "observations"):
      value = obs.get(key)
      if isinstance(value, torch.Tensor):
        return value
  raise TypeError("specialist matching requires a tensor actor observation")


def _collect_and_update_specialist(
  runner,
  obs,
  *,
  critic_only: bool,
  specialist_mode: str,
  failure_bank,
  success_pool,
  success_bank,
  failure_fraction: float,
  success_fraction: float,
  specialist_generator: torch.Generator,
  minimum_riser: int,
  protocol_version: int = 17,
  defer_update: bool = False,
):
  """Collect one specialist rollout; v19 may defer a paired PPO update."""
  from rsl_rl.utils import check_nan
  from src.tasks.stairs_cbf.hard_cases import (
    SPECIALIST_FAILURE_TYPES,
    capture_hard_case_state,
    classify_target_failure_mode,
    classify_v19_failure_mode,
    hard_case_state_shape_mismatches,
    match_specialist_success_counterexamples,
    match_v19_success_counterexamples,
    finalize_v19_replay_bank_update,
    reset_rollout_with_specialist_banks,
    restore_hard_case_state,
    select_specialist_failure_candidates,
    select_specialist_success_candidates,
    select_v19_contact_candidates,
    select_v19_lateral_failure_candidates,
    select_v19_lateral_success_candidates,
    specialist_history_window,
  )
  from src.tasks.stairs_cbf.mdp import specialist_failure_signal_components

  if protocol_version not in (17, 19):
    raise ValueError("specialist collector protocol must be v17 or v19")
  v19 = protocol_version == 19
  allowed_modes = (
    ("lateral", "contact_stability") if v19 else ("lateral", "cbf", "balance")
  )
  if specialist_mode not in allowed_modes:
    raise ValueError(f"unsupported specialist mode: {specialist_mode!r}")
  if defer_update and not v19:
    raise ValueError("deferred specialist rollout updates are reserved for v19")
  if minimum_riser < 1:
    raise ValueError("specialist minimum riser must be positive")
  persistent_slots = bool(failure_fraction or success_fraction)
  bank_snapshots = (
    (
      failure_bank.state_dict(),
      success_pool.state_dict(),
      success_bank.state_dict(),
    )
    if v19 and persistent_slots
    else None
  )
  runner.alg.set_critic_only(critic_only)
  runner.alg.clear_cbf_rollout()
  runner.alg.train_mode()
  del obs
  obs, start_metrics = reset_rollout_with_specialist_banks(
    runner.env,
    failure_bank,
    success_bank,
    failure_fraction=failure_fraction,
    success_fraction=success_fraction,
    matched_pair_sampling=v19,
    generator=specialist_generator,
  )
  device = runner.env.device
  failure_slots = torch.zeros(runner.env.num_envs, dtype=torch.bool, device=device)
  success_slots = torch.zeros_like(failure_slots)
  failure_ids = torch.tensor(
    start_metrics["failure_start_ids"], dtype=torch.long, device=device
  )
  success_ids = torch.tensor(
    start_metrics["success_start_ids"], dtype=torch.long, device=device
  )
  if len(failure_ids):
    failure_slots[failure_ids] = True
  if len(success_ids):
    success_slots[success_ids] = True
  normal_slots = ~(failure_slots | success_slots)
  maximum_history_steps = (
    150
    if v19 and specialist_mode == "lateral"
    else 100
    if v19
    else specialist_history_window(specialist_mode)[1]
  )
  history_capacity = max(maximum_history_steps + 1, 384)
  state_history = deque(maxlen=history_capacity)
  actor_observation_history = deque(maxlen=history_capacity)
  riser_history = deque(maxlen=history_capacity)
  gait_phase_history = deque(maxlen=history_capacity)
  support_foot_history = deque(maxlen=history_capacity)
  delivered_command_history = deque(maxlen=history_capacity)
  root_velocity_history = deque(maxlen=history_capacity)
  cbf_active_history = deque(maxlen=history_capacity)
  touchdown_history = deque(maxlen=history_capacity)
  component_history = {
    name: deque(maxlen=history_capacity)
    for name in (
      "centerline",
      "heading",
      "edge",
      "intervention",
      "nominal_margin",
      "roll",
      "pitch",
      "angular_velocity",
      "slip",
      "contact_mismatch",
      "centerline_signed",
      "heading_signed",
      "centerline_rate",
      "heading_rate",
      "left_contact",
      "right_contact",
      "left_slip",
      "right_slip",
    )
  }
  valid_steps = torch.zeros(
    runner.env.num_envs, dtype=torch.long, device=device
  )
  episode_side_edge_breach = torch.zeros_like(failure_slots)
  episode_max_abs_centerline_error = torch.zeros(
    runner.env.num_envs, device=device
  )
  episode_max_abs_heading_error = torch.zeros_like(
    episode_max_abs_centerline_error
  )
  episode_correction_max = torch.zeros_like(episode_max_abs_centerline_error)
  episode_max_left_slip = torch.zeros_like(episode_max_abs_centerline_error)
  episode_max_right_slip = torch.zeros_like(episode_max_abs_centerline_error)
  episode_contact_mismatch_sum = torch.zeros_like(
    episode_max_abs_centerline_error
  )
  episode_first_lateral_event_step = torch.full(
    (runner.env.num_envs,), -1, dtype=torch.long, device=device
  )
  episode_first_contact_event_step = torch.full_like(
    episode_first_lateral_event_step, -1
  )
  episode_contact_instability_streak = torch.zeros_like(
    episode_first_lateral_event_step
  )
  contact_history_initialized = torch.zeros_like(failure_slots)
  previous_contact = torch.zeros(
    runner.env.num_envs, 2, dtype=torch.bool, device=device
  )
  failure_added = 0
  success_pool_added = 0
  target_falls_seen = 0
  target_failures_admitted = 0
  successful_episodes_seen = 0
  rejected_failure_type_counts: dict[str, int] = {}
  failure_restart_count = 0
  success_restart_count = 0
  normal_completed = 0
  normal_successes = 0
  normal_falls = 0

  with torch.no_grad():
    for _ in range(runner.cfg["num_steps_per_env"]):
      unwrapped = runner.env.unwrapped
      state_history.append(capture_hard_case_state(unwrapped))
      actor_observation_history.append(
        _actor_observation_batch(obs).detach().clone()
      )
      action_term = unwrapped.action_manager.get_term("joint_pos")
      command_term = unwrapped.command_manager.get_term("twist")
      terrain = unwrapped.scene.terrain
      if terrain is None:
        raise RuntimeError("specialist refinement requires stair terrain")
      edge_x = action_term._edge_x[
        terrain.terrain_levels, terrain.terrain_types
      ]
      root = unwrapped.scene["robot"]
      root_x = root.data.root_link_pos_w[:, 0:1]
      riser = torch.sum(root_x >= edge_x, dim=1).detach().clone()
      riser_history.append(riser)
      components = specialist_failure_signal_components(unwrapped)
      for name, values in components.items():
        component_history[name].append(values.detach().clone())
      phase = ((unwrapped.episode_length_buf * unwrapped.step_dt) / 0.6) % 1.0
      gait_phase_history.append(phase.detach().clone())
      contact = unwrapped.scene["feet_ground_contact"].data.found
      if contact is None:
        raise RuntimeError("specialist matching requires foot contact state")
      if contact.ndim == 3 and contact.shape[-1] == 1:
        contact = contact.squeeze(-1)
      contact = contact.bool()
      touchdown = (
        contact
        & ~previous_contact
        & contact_history_initialized.unsqueeze(1)
      )
      touchdown_history.append(touchdown.detach().clone())
      previous_contact[:] = contact
      contact_history_initialized[:] = True
      scheduled_support = (phase >= 0.5).long()
      support = torch.where(
        contact[:, 0] & ~contact[:, 1],
        torch.zeros_like(scheduled_support),
        torch.where(
          contact[:, 1] & ~contact[:, 0],
          torch.ones_like(scheduled_support),
          scheduled_support,
        ),
      )
      support_foot_history.append(support.detach().clone())
      delivered_command_history.append(
        command_term.delivered_command.detach().clone()
      )
      root_velocity_history.append(root.data.root_link_lin_vel_b.detach().clone())
      cbf_active_history.append(
        (action_term.target_intervention_norm > 0.01).detach().clone()
      )
      centerline_error = getattr(
        command_term,
        "centerline_error",
        torch.zeros(runner.env.num_envs, device=device),
      )
      heading_error = getattr(
        command_term,
        "heading_error",
        torch.zeros(runner.env.num_envs, device=device),
      )
      stair_half_width = float(
        getattr(command_term.cfg, "stair_half_width", 1.20)
      )
      episode_max_abs_centerline_error = torch.maximum(
        episode_max_abs_centerline_error, centerline_error.abs()
      )
      episode_max_abs_heading_error = torch.maximum(
        episode_max_abs_heading_error, heading_error.abs()
      )
      episode_correction_max = torch.maximum(
        episode_correction_max, action_term.target_intervention_norm
      )
      episode_max_left_slip = torch.maximum(
        episode_max_left_slip, components["left_slip"]
      )
      episode_max_right_slip = torch.maximum(
        episode_max_right_slip, components["right_slip"]
      )
      episode_contact_mismatch_sum += components["contact_mismatch"]
      patches = terrain.flat_patches["stair_targets"][
        terrain.terrain_levels, terrain.terrain_types
      ]
      center_y = patches[:, 0, 1]
      foot_y = root.data.site_pos_w[:, action_term._site_local_ids, 1]
      foot_edge_breach = torch.max(
        torch.abs(foot_y - center_y.unsqueeze(1)), dim=1
      ).values >= stair_half_width
      episode_side_edge_breach |= (
        centerline_error.abs() >= stair_half_width
      ) | foot_edge_breach
      severe_contact_slip = torch.maximum(
        components["left_slip"], components["right_slip"]
      ) >= 0.50
      episode_contact_instability_streak = torch.where(
        severe_contact_slip,
        episode_contact_instability_streak + 1,
        torch.zeros_like(episode_contact_instability_streak),
      )
      new_contact_event = (
        (episode_first_contact_event_step < 0)
        & (episode_contact_instability_streak >= 3)
      )
      episode_first_contact_event_step = torch.where(
        new_contact_event,
        (valid_steps - 2).clamp_min(0),
        episode_first_contact_event_step,
      )
      lateral_event = (
        (centerline_error.abs() >= (2.0 / 3.0) * stair_half_width)
        | (heading_error.abs() >= math.pi / 2.0)
        | foot_edge_breach
      )
      episode_first_lateral_event_step = torch.where(
        (episode_first_lateral_event_step < 0) & lateral_event,
        valid_steps,
        episode_first_lateral_event_step,
      )

      actions = runner.alg.act(obs)
      obs, rewards, dones, extras = runner.env.step(actions.to(device))
      check_nan(obs, rewards, dones)
      extras = dict(extras)
      # v19 failure precursors use unit actor weight. Matched-success slots
      # remain in the scalar-reward PPO surrogate with weight 1.25.
      extras["online_hard_case_transition"] = failure_slots.detach().clone()
      extras["online_success_counterexample_transition"] = (
        success_slots.detach().clone()
      )
      done_mask = dones.bool()
      timeouts = extras.get(
        "time_outs", torch.zeros_like(done_mask, dtype=torch.bool)
      ).bool()
      fell = extras.get(
        "online_fell", torch.zeros_like(done_mask, dtype=torch.bool)
      ).bool()
      magnitude = extras.get("cbf_intervention_magnitude")
      if magnitude is not None:
        episode_correction_max = torch.maximum(
          episode_correction_max, magnitude
        )
      completed_normal = done_mask & normal_slots & ~timeouts
      normal_completed += int(completed_normal.sum())
      normal_falls += int((completed_normal & fell).sum())
      normal_successes += int((completed_normal & ~fell).sum())

      histories = None
      terminal_ids = completed_normal.nonzero(as_tuple=False).flatten()
      if len(terminal_ids):
        histories = {
          "states": list(state_history),
          "observations": list(actor_observation_history),
          "riser": list(riser_history),
          "phase": list(gait_phase_history),
          "support": list(support_foot_history),
          "command": list(delivered_command_history),
          "velocity": list(root_velocity_history),
          "cbf_active": list(cbf_active_history),
          "touchdown": list(touchdown_history),
          "components": {
            name: list(values) for name, values in component_history.items()
          },
        }
      for env_id in terminal_ids.tolist():
        assert histories is not None
        episode_steps = min(int(valid_steps[env_id]) + 1, len(histories["states"]))
        start = len(histories["states"]) - episode_steps
        if bool(fell[env_id]):
          target_falls_seen += 1
          failure_type = (
            classify_v19_failure_mode(
              specialist_mode=specialist_mode,
              side_edge_breach=bool(episode_side_edge_breach[env_id]),
              max_abs_centerline_error=float(
                episode_max_abs_centerline_error[env_id]
              ),
              max_abs_heading_error=float(
                episode_max_abs_heading_error[env_id]
              ),
              correction_max=float(episode_correction_max[env_id]),
              maximum_left_slip_speed=float(episode_max_left_slip[env_id]),
              maximum_right_slip_speed=float(episode_max_right_slip[env_id]),
              mean_contact_mismatch=float(
                episode_contact_mismatch_sum[env_id]
                / max(1, episode_steps)
              ),
              stair_half_width=stair_half_width,
              first_lateral_event_step=(
                None
                if int(episode_first_lateral_event_step[env_id]) < 0
                else int(episode_first_lateral_event_step[env_id])
              ),
              first_contact_event_step=(
                None
                if int(episode_first_contact_event_step[env_id]) < 0
                else int(episode_first_contact_event_step[env_id])
              ),
            )
            if v19
            else classify_target_failure_mode(
              side_edge_breach=bool(episode_side_edge_breach[env_id]),
              max_abs_centerline_error=float(
                episode_max_abs_centerline_error[env_id]
              ),
              max_abs_heading_error=float(
                episode_max_abs_heading_error[env_id]
              ),
              correction_max=float(episode_correction_max[env_id]),
              stair_half_width=stair_half_width,
            )
          )
          if failure_type != SPECIALIST_FAILURE_TYPES[specialist_mode]:
            rejected_failure_type_counts[failure_type] = (
              rejected_failure_type_counts.get(failure_type, 0) + 1
            )
            continue
          target_failures_admitted += 1
          selection_args = (
            torch.stack(histories["riser"][start:])[:, env_id],
            {
              name: torch.stack(values[start:])[:, env_id]
              for name, values in histories["components"].items()
            },
            torch.stack(histories["phase"][start:])[:, env_id],
            torch.stack(histories["support"][start:])[:, env_id],
            torch.stack(histories["command"][start:])[:, env_id],
            torch.stack(histories["velocity"][start:])[:, env_id],
            torch.stack(histories["cbf_active"][start:])[:, env_id],
          )
          if v19 and specialist_mode == "lateral":
            candidates = select_v19_lateral_failure_candidates(
              *selection_args,
              minimum_riser=minimum_riser,
              total_risers=edge_x.shape[1],
            )
          elif v19:
            candidates = select_v19_contact_candidates(
              *selection_args,
              torch.stack(histories["touchdown"][start:])[:, env_id],
              minimum_riser=minimum_riser,
              outcome="failure",
            )
          else:
            candidates = select_specialist_failure_candidates(
              specialist_mode,
              *selection_args,
              minimum_riser=minimum_riser,
              failure_type=failure_type,
            )
          for candidate in candidates:
            history_index = start + candidate.history_index
            failure_added += failure_bank.add_specialist_candidate(
              histories["states"][history_index],
              env_id,
              candidate,
              histories["observations"][history_index][env_id],
            )
        else:
          successful_episodes_seen += 1
          selection_args = (
            torch.stack(histories["riser"][start:])[:, env_id],
            {
              name: torch.stack(values[start:])[:, env_id]
              for name, values in histories["components"].items()
            },
            torch.stack(histories["phase"][start:])[:, env_id],
            torch.stack(histories["support"][start:])[:, env_id],
            torch.stack(histories["command"][start:])[:, env_id],
            torch.stack(histories["velocity"][start:])[:, env_id],
            torch.stack(histories["cbf_active"][start:])[:, env_id],
          )
          if v19 and specialist_mode == "lateral":
            candidates = select_v19_lateral_success_candidates(
              *selection_args,
              minimum_riser=minimum_riser,
              total_risers=edge_x.shape[1],
            )
          elif v19:
            candidates = select_v19_contact_candidates(
              *selection_args,
              torch.stack(histories["touchdown"][start:])[:, env_id],
              minimum_riser=minimum_riser,
              outcome="success",
            )
          else:
            candidates = select_specialist_success_candidates(
              specialist_mode,
              *selection_args,
              minimum_riser=minimum_riser,
            )
          for candidate in candidates:
            history_index = start + candidate.history_index
            success_pool_added += success_pool.add_specialist_candidate(
              histories["states"][history_index],
              env_id,
              candidate,
              histories["observations"][history_index][env_id],
            )

      if persistent_slots:
        restart_groups = (
          (
            "failure",
            (done_mask & failure_slots).nonzero(as_tuple=False).flatten(),
            failure_bank,
          ),
          (
            "success",
            (done_mask & success_slots).nonzero(as_tuple=False).flatten(),
            success_bank,
          ),
        )
        restored_any = False
        for label, restart_ids, bank in restart_groups:
          if not len(restart_ids):
            continue
          if len(bank) < len(restart_ids):
            raise RuntimeError(
              f"specialist {label} bank cannot refill persistent slots"
            )
          replay = bank.sample(
            len(restart_ids), device=device, generator=specialist_generator
          )
          mismatches = hard_case_state_shape_mismatches(
            capture_hard_case_state(unwrapped), replay
          )
          if mismatches:
            raise RuntimeError(
              f"persistent specialist {label} replay became incompatible: "
              + "; ".join(mismatches[:5])
            )
          restore_hard_case_state(unwrapped, replay, restart_ids)
          restored_any = True
          if label == "failure":
            failure_restart_count += len(restart_ids)
          else:
            success_restart_count += len(restart_ids)
        if restored_any:
          unwrapped.scene.write_data_to_sim()
          unwrapped.sim.forward()
          for sensor in unwrapped.scene.sensors.values():
            sensor._invalidate_cache()
          unwrapped.sim.sense()
          unwrapped.observation_manager._obs_buffer = None
          unwrapped.obs_buf = unwrapped.observation_manager.compute(
            update_history=False
          )
          obs = runner.env.get_observations()

      valid_steps = torch.where(
        done_mask, torch.zeros_like(valid_steps), valid_steps + 1
      )
      episode_side_edge_breach[done_mask] = False
      episode_max_abs_centerline_error[done_mask] = 0.0
      episode_max_abs_heading_error[done_mask] = 0.0
      episode_correction_max[done_mask] = 0.0
      episode_max_left_slip[done_mask] = 0.0
      episode_max_right_slip[done_mask] = 0.0
      episode_contact_mismatch_sum[done_mask] = 0.0
      episode_first_lateral_event_step[done_mask] = -1
      episode_first_contact_event_step[done_mask] = -1
      episode_contact_instability_streak[done_mask] = 0
      contact_history_initialized[done_mask] = False
      previous_contact[done_mask] = False
      obs = obs.to(runner.device)
      rewards = rewards.to(runner.device)
      dones = dones.to(runner.device)
      runner.alg.process_env_step(obs, rewards, dones, extras)

    matching = (
      (
        match_v19_success_counterexamples(
          failure_bank, success_pool, success_bank
        )
        if v19
        else match_specialist_success_counterexamples(
          failure_bank, success_pool, success_bank
        )
      )
      if len(failure_bank) and len(success_pool)
      else {
        "specialist_mode": specialist_mode,
        "failure_entry_count": len(failure_bank),
        "success_pool_entry_count": len(success_pool),
        "matched_entry_count": len(success_bank),
        "one_match_per_replayed_failure": False,
        "matches": [],
      }
    )
    bank_update_transaction = {
      "attempted": False,
      "committed": None,
      "rollback_reason": None,
      "post_update_preflight": None,
      "restored_preflight": None,
      "usable_preflight": None,
    }
    if bank_snapshots is not None:
      pair_count = min(
        int(start_metrics["failure_start_count"]),
        int(start_metrics["success_start_count"]),
      )
      bank_update_transaction = finalize_v19_replay_bank_update(
        failure_bank,
        success_pool,
        success_bank,
        bank_snapshots,
        pair_count,
      )
      matching["bank_update_committed"] = bank_update_transaction["committed"]
    credit_metrics = runner.alg.relabel_pre_intervention_costs()
    fall_credit_metrics = runner.alg.redistribute_failure_focused_fall_penalty()
    completion_metrics = {
      "normal_start_completed_episode_count": normal_completed,
      "normal_start_success_count": normal_successes,
      "normal_start_fall_count": normal_falls,
      "normal_start_success_rate": normal_successes / max(1, normal_completed),
      "normal_start_fall_rate": normal_falls / max(1, normal_completed),
    }
    rollout_metadata = {
      **completion_metrics,
      **start_metrics,
      "specialist_mode": specialist_mode,
      "protocol_version": protocol_version,
      "bank_update_transaction": bank_update_transaction,
    }
    runner.alg.last_update_metrics.update(rollout_metadata)
    runner.alg.compute_returns(obs)
    if defer_update:
      advantage_metrics = {
        "v19_grouped_advantage_normalization_deferred": True,
      }
      rollout_batch = runner.alg.capture_rollout_batch()
      runner.alg.clear_captured_rollout()
      losses: dict[str, Any] = {}
    else:
      advantage_metrics = runner.alg.prepare_constrained_advantages()
      rollout_batch = None
  if not defer_update:
    losses = runner.alg.update()
  losses.update(credit_metrics)
  losses.update(fall_credit_metrics)
  losses.update(advantage_metrics)
  losses.update(completion_metrics)
  losses.update(start_metrics)
  losses.update(
    {
      "specialist_mode": specialist_mode,
      "history_capacity_steps": history_capacity,
      "failure_bank_added": failure_added,
      "success_pool_added": success_pool_added,
      "failure_bank_size_after_rollout": len(failure_bank),
      "success_pool_size_after_rollout": len(success_pool),
      "success_bank_size_after_matching": len(success_bank),
      "target_falls_seen": target_falls_seen,
      "target_failures_admitted": target_failures_admitted,
      "successful_episodes_seen": successful_episodes_seen,
      "rejected_failure_type_counts": rejected_failure_type_counts,
      "failure_restart_count": failure_restart_count,
      "success_restart_count": success_restart_count,
      "failure_bank_audit": failure_bank.audit_metadata(),
      "success_pool_audit": success_pool.audit_metadata(),
      "success_bank_audit": success_bank.audit_metadata(),
      "success_matching": matching,
      "bank_update_transaction": bank_update_transaction,
    }
  )
  if defer_update:
    return obs, losses, rollout_batch
  return obs, losses


def _save_checkpoint(
  runner,
  path: Path,
  *,
  iteration: int,
  metadata: dict[str, Any],
  hard_case_bank=None,
  hard_case_generator: torch.Generator | None = None,
  specialist_success_pool=None,
  specialist_success_bank=None,
) -> None:
  payload = runner.alg.save()
  payload["iter"] = iteration
  payload["infos"] = {"online_refinement": metadata}
  if hard_case_bank is not None:
    payload["hard_case_bank"] = hard_case_bank.state_dict()
  if hard_case_generator is not None:
    payload["hard_case_generator_state"] = hard_case_generator.get_state()
  if specialist_success_pool is not None:
    payload["specialist_success_pool"] = specialist_success_pool.state_dict()
  if specialist_success_bank is not None:
    payload["specialist_success_bank"] = specialist_success_bank.state_dict()
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument(
    "--resume-online-checkpoint",
    type=Path,
    help="Accepted 799-D checkpoint to refine; base checkpoint remains the KL/retention reference.",
  )
  parser.add_argument(
    "--resume-hard-case-bank",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Restore hard cases only when resuming within the same target domain.",
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--rollout-steps", type=int, default=256)
  parser.add_argument("--critic-burn-in-rounds", type=int, default=2)
  parser.add_argument("--critic-burn-in-max-rounds", type=int, default=4)
  parser.add_argument(
    "--critic-min-explained-variance", type=float, default=0.50
  )
  parser.add_argument("--online-rounds", type=int, default=2)
  parser.add_argument("--eval-num-envs", type=int, default=8)
  parser.add_argument("--eval-num-episodes", type=int, default=8)
  parser.add_argument(
    "--candidate-fractions",
    nargs="+",
    type=float,
    default=(1.0,),
    help="Actor fractions screened along each fresh PPO update direction.",
  )
  parser.add_argument("--candidate-screen-num-envs", type=int, default=4)
  parser.add_argument("--candidate-screen-repeats", type=int, default=1)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
  parser.add_argument("--pre-intervention-weight", type=float, default=0.20)
  parser.add_argument("--std-scale-from-base", type=float, default=0.35)
  parser.add_argument("--safe-bc-weight", type=float, default=0.0)
  parser.add_argument(
    "--task-first-constrained",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      "Remove fixed CBF/fall reward shaping and optimize task advantage minus "
      "adaptive fall/intervention cost advantages from separate critics."
    ),
  )
  parser.add_argument("--fall-multiplier-learning-rate", type=float, default=1.0)
  parser.add_argument(
    "--intervention-multiplier-learning-rate", type=float, default=0.10
  )
  parser.add_argument("--maximum-cost-multiplier", type=float, default=20.0)
  parser.add_argument("--intervention-budget-slack", type=float, default=1.05)
  parser.add_argument("--hard-case-policy-weight", type=float, default=0.0)
  parser.add_argument(
    "--correction-distillation-weight", type=float, default=0.0
  )
  parser.add_argument("--correction-success-horizon", type=int, default=100)
  parser.add_argument("--risk-horizon", type=int, default=50)
  parser.add_argument("--strong-intervention-fraction", type=float, default=0.5)
  parser.add_argument("--risk-loss-coef", type=float, default=1.0)
  parser.add_argument("--hard-case-fraction", type=float, default=0.20)
  parser.add_argument("--base-anchor-weight", type=float, default=0.01)
  parser.add_argument("--d0-retention-bank", type=Path)
  parser.add_argument("--neighbor-retention-bank", type=Path)
  parser.add_argument("--d0-retention-anchor-weight", type=float, default=0.0)
  parser.add_argument(
    "--neighbor-retention-anchor-weight", type=float, default=0.0
  )
  parser.add_argument(
    "--d0-retention-anchor-kl-budget", type=float, default=0.002
  )
  parser.add_argument(
    "--neighbor-retention-anchor-kl-budget", type=float, default=0.002
  )
  parser.add_argument(
    "--retention-anchor-adaptation-rate", type=float, default=10.0
  )
  parser.add_argument(
    "--maximum-retention-anchor-weight", type=float, default=0.20
  )
  parser.add_argument("--retention-anchor-batch-size", type=int, default=4096)
  parser.add_argument(
    "--intervention-advantage-weight", type=float, default=0.075
  )
  parser.add_argument(
    "--neighbor-command-fraction",
    type=float,
    default=0.15,
    help="Fraction of bottom starts with bounded neighboring joystick speed/delay.",
  )
  parser.add_argument(
    "--neighbor-forward-scale-range",
    nargs=2,
    type=float,
    default=(0.90, 1.10),
    metavar=("LOW", "HIGH"),
  )
  parser.add_argument(
    "--neighbor-delay-step-offset-range",
    nargs=2,
    type=int,
    default=(-2, 2),
    metavar=("LOW", "HIGH"),
  )
  parser.add_argument("--hard-case-pre-steps", type=int, default=10)
  parser.add_argument("--hard-case-capacity", type=int, default=256)
  parser.add_argument(
    "--minimum-normal-complete-episodes",
    type=int,
    default=0,
    help="Reject a rollout that lacks this many completed non-hard-case episodes.",
  )
  parser.add_argument(
    "--late-critic-risers", nargs="*", type=int, default=(7, 8, 9)
  )
  parser.add_argument("--critic-min-samples-per-late-riser", type=int, default=0)
  parser.add_argument("--critic-min-fall-events", type=int, default=0)
  parser.add_argument("--risk-maximum-brier", type=float, default=1.0)
  parser.add_argument("--risk-minimum-auc", type=float, default=0.0)
  parser.add_argument(
    "--minimum-pre-fall-cost-value-rise", type=float
  )
  parser.add_argument(
    "--adaptive-std",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      "Legacy exploration adaptation. This is disabled by default and cannot "
      "be combined with a non-zero frozen base-policy KL anchor."
    ),
  )
  parser.add_argument("--target-intervention-per-riser", type=float, default=0.10)
  parser.add_argument("--std-adaptation-rate", type=float, default=0.10)
  parser.add_argument(
    "--maximum-target-fall-rate",
    type=float,
    default=0.0,
    help="Hard candidate safety gate; formal shielded refinement defaults to zero falls.",
  )
  parser.add_argument(
    "--independence-audit",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Evaluate the final actor with and without the CBF in simulation.",
  )
  parser.add_argument(
    "--resume-std-scale",
    type=float,
    default=1.0,
    help="Additional bounded exploration scaling after loading an accepted checkpoint.",
  )
  parser.add_argument(
    "--fall-penalty-weight",
    type=float,
    help="Override the fall-only reward weight; MJLab multiplies it by dt.",
  )
  parser.add_argument(
    "--train-runtime-filter",
    choices=("on", "off"),
    default="on",
    help="Execute the CBF during rollout collection. Off is simulation-only finalization.",
  )
  parser.add_argument(
    "--gate-runtime-filter",
    choices=("on", "off"),
    default="on",
    help="Deployment mode used by the transactional D0/target/neighbor gate.",
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--gate-device",
    default="cuda:0",
    help="Physics device for paired candidate acceptance (GPU by default).",
  )
  parser.add_argument(
    "--gate-repeats",
    type=int,
    default=3,
    help="Independent fixed-seed GPU evaluation replicates per policy/domain.",
  )
  parser.add_argument(
    "--train-domain",
    default="DQ",
    help="Target domain used for online rollouts (DQ quick prototype or D4 formal).",
  )
  parser.add_argument("--neighbor-domain", default="DQN")
  parser.add_argument(
    "--baseline-domains",
    nargs="+",
    default=["D0", "D1", "D2", "D3", "D4", "D5", "DQ", "DQN"],
  )
  parser.add_argument(
    "--reuse-baseline-eval",
    type=Path,
    help="Reuse a matched-arm baseline JSON after strict protocol validation.",
  )
  args = parser.parse_args()
  any_anchor_weight = max(
    args.base_anchor_weight,
    args.d0_retention_anchor_weight,
    args.neighbor_retention_anchor_weight,
  ) > 0.0
  if args.adaptive_std and any_anchor_weight:
    raise ValueError(
      "--adaptive-std changes the action distribution independently of the "
      "frozen reference policy; use --no-adaptive-std with KL anchors"
    )
  if args.resume_std_scale != 1.0 and any_anchor_weight:
    raise ValueError(
      "--resume-std-scale must remain 1.0 with a non-zero policy KL anchor"
    )
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.online import (
    CandidateGateThresholds,
    adaptive_cbf_std_factor,
    candidate_gate,
    candidate_gate_intervals,
    candidate_precheck,
    cbf_independence_gate,
    critic_readiness_reasons,
    safe_improvement_score,
    safety_demand_per_riser,
  )
  from src.tasks.stairs_cbf.hard_cases import HardCaseStateBank

  if not 0.0 <= args.hard_case_fraction <= 1.0:
    raise ValueError("--hard-case-fraction must be in [0, 1]")
  if not 0.0 <= args.neighbor_command_fraction <= 1.0:
    raise ValueError("--neighbor-command-fraction must be in [0, 1]")
  if args.hard_case_fraction + args.neighbor_command_fraction > 1.0:
    raise ValueError("hard-case and neighboring-command fractions exceed one")
  if not 0.0 <= args.maximum_target_fall_rate <= 1.0:
    raise ValueError("--maximum-target-fall-rate must be in [0, 1]")
  if args.hard_case_pre_steps < 1:
    raise ValueError("--hard-case-pre-steps must be positive")
  if not 0 <= args.critic_burn_in_rounds <= args.critic_burn_in_max_rounds:
    raise ValueError("critic burn-in minimum/max rounds are inconsistent")
  if args.minimum_normal_complete_episodes < 0:
    raise ValueError("minimum normal completed episodes must be non-negative")
  if not 0.0 <= args.hard_case_policy_weight <= 1.0:
    raise ValueError("hard-case policy weight must be in [0, 1]")
  if args.intervention_budget_slack < 1.0:
    raise ValueError("intervention budget slack must be at least one")
  if not args.candidate_fractions or any(
    not 0.0 < fraction <= 1.5 for fraction in args.candidate_fractions
  ):
    raise ValueError("candidate fractions must be non-empty and in (0, 1.5]")
  if len(set(args.candidate_fractions)) != len(args.candidate_fractions):
    raise ValueError("candidate fractions must be unique")
  if args.candidate_screen_num_envs < 1 or args.candidate_screen_repeats < 1:
    raise ValueError("candidate screen environment/repeat counts must be positive")
  retention_values = torch.tensor(
    [
      args.d0_retention_anchor_weight,
      args.neighbor_retention_anchor_weight,
      args.d0_retention_anchor_kl_budget,
      args.neighbor_retention_anchor_kl_budget,
      args.retention_anchor_adaptation_rate,
      args.maximum_retention_anchor_weight,
    ],
    dtype=torch.float64,
  )
  if not bool(torch.isfinite(retention_values).all()):
    raise ValueError("retention anchor arguments must be finite")
  if bool((retention_values[:5] < 0.0).any()):
    raise ValueError("retention anchor weights, budgets, and rate are non-negative")
  if args.maximum_retention_anchor_weight <= 0.0:
    raise ValueError("maximum retention anchor weight must be positive")
  if max(
    args.d0_retention_anchor_weight,
    args.neighbor_retention_anchor_weight,
  ) > args.maximum_retention_anchor_weight:
    raise ValueError("initial retention anchor weight exceeds its maximum")
  if args.retention_anchor_batch_size < 1:
    raise ValueError("retention anchor batch size must be positive")
  if args.d0_retention_anchor_weight > 0.0 and args.d0_retention_bank is None:
    raise ValueError("--d0-retention-anchor-weight requires --d0-retention-bank")
  if (
    args.neighbor_retention_anchor_weight > 0.0
    and args.neighbor_retention_bank is None
  ):
    raise ValueError(
      "--neighbor-retention-anchor-weight requires --neighbor-retention-bank"
    )

  task = f"Unitree-G1-Stairs-Online-{args.train_domain}"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env_cfg.actions["joint_pos"].enabled = args.train_runtime_filter == "on"
  if args.fall_penalty_weight is not None:
    env_cfg.rewards["fall_termination"].weight = args.fall_penalty_weight
  if args.task_first_constrained:
    # Safety is represented exactly once through cost GAE and adaptive duals.
    # Keeping either fixed term would recreate the v11 scalarization failure.
    env_cfg.rewards["cbf_dual"].weight = 0.0
    env_cfg.rewards["fall_termination"].weight = 0.0
  agent_cfg = load_rl_cfg(task)
  agent_cfg.seed = args.seed
  agent_cfg.num_steps_per_env = args.rollout_steps
  agent_cfg.algorithm.actor_learning_rate = args.actor_learning_rate
  agent_cfg.algorithm.critic_learning_rate = args.critic_learning_rate
  agent_cfg.algorithm.pre_intervention_weight = (
    0.0 if args.task_first_constrained else args.pre_intervention_weight
  )
  agent_cfg.algorithm.base_anchor_weight = args.base_anchor_weight
  agent_cfg.algorithm.d0_retention_anchor_weight = (
    args.d0_retention_anchor_weight
  )
  agent_cfg.algorithm.neighbor_retention_anchor_weight = (
    args.neighbor_retention_anchor_weight
  )
  agent_cfg.algorithm.d0_retention_anchor_kl_budget = (
    args.d0_retention_anchor_kl_budget
  )
  agent_cfg.algorithm.neighbor_retention_anchor_kl_budget = (
    args.neighbor_retention_anchor_kl_budget
  )
  agent_cfg.algorithm.retention_anchor_adaptation_rate = (
    args.retention_anchor_adaptation_rate
  )
  agent_cfg.algorithm.maximum_retention_anchor_weight = (
    args.maximum_retention_anchor_weight
  )
  agent_cfg.algorithm.retention_anchor_batch_size = (
    args.retention_anchor_batch_size
  )
  agent_cfg.algorithm.intervention_advantage_weight = (
    0.0
    if args.task_first_constrained
    else args.intervention_advantage_weight
  )
  agent_cfg.algorithm.std_scale_from_base = args.std_scale_from_base
  agent_cfg.algorithm.safe_bc_weight = args.safe_bc_weight
  agent_cfg.algorithm.task_first_constrained = args.task_first_constrained
  agent_cfg.algorithm.fall_multiplier_learning_rate = (
    args.fall_multiplier_learning_rate
  )
  agent_cfg.algorithm.intervention_multiplier_learning_rate = (
    args.intervention_multiplier_learning_rate
  )
  agent_cfg.algorithm.maximum_cost_multiplier = args.maximum_cost_multiplier
  agent_cfg.algorithm.hard_case_policy_weight = args.hard_case_policy_weight
  agent_cfg.algorithm.correction_distillation_weight = (
    args.correction_distillation_weight
  )
  agent_cfg.algorithm.correction_success_horizon = (
    args.correction_success_horizon
  )
  agent_cfg.algorithm.risk_horizon = args.risk_horizon
  agent_cfg.algorithm.strong_intervention_fraction = (
    args.strong_intervention_fraction
  )
  agent_cfg.algorithm.risk_loss_coef = args.risk_loss_coef
  agent_cfg.algorithm.use_counterfactual_cbf_credit = (
    args.train_runtime_filter == "off"
  )
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("online refinement task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  hard_case_bank = HardCaseStateBank(capacity=args.hard_case_capacity)
  hard_case_generator = torch.Generator(device="cpu")
  hard_case_generator.manual_seed(args.seed + 100003)
  if args.resume_online_checkpoint is None:
    warm_start = runner.load_base_checkpoint(
      str(args.base_checkpoint), map_location=args.device
    )
  else:
    warm_start = runner.load_online_checkpoint(
      str(args.resume_online_checkpoint.resolve()),
      map_location=args.device,
    )
    # A backtracked candidate's saved Adam moments correspond to the full PPO
    # step, not the accepted fractional parameter point. Start each accepted
    # round with the configured conservative optimizer instead.
    runner.alg.scale_exploration_std(args.resume_std_scale)
    warm_start |= {
      "resume_online_checkpoint": str(args.resume_online_checkpoint.resolve()),
      "resume_std_scale": args.resume_std_scale,
    }
    resume_payload = torch.load(
      args.resume_online_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    if args.resume_hard_case_bank and "hard_case_bank" in resume_payload:
      hard_case_bank.load_state_dict(resume_payload["hard_case_bank"])
    if args.resume_hard_case_bank and "hard_case_generator_state" in resume_payload:
      hard_case_generator.set_state(resume_payload["hard_case_generator_state"])
  obs, _ = env.reset()
  current_actor_state = _actor_state(runner.alg.actor)
  base_payload = torch.load(
    args.base_checkpoint.resolve(), map_location=args.device, weights_only=False
  )
  from src.tasks.stairs_cbf.online import backtrack_actor_state

  # The anchor always points to the original pretrained mean policy, including
  # when this run resumes a later accepted online checkpoint.
  runner.alg.set_base_actor_reference(base_payload["actor_state_dict"])

  retention_bank_metadata: dict[str, dict[str, Any]] = {}
  d0_retention_payload = None
  neighbor_retention_payload = None
  if args.d0_retention_bank is not None:
    d0_bank_path = args.d0_retention_bank.resolve()
    d0_retention_payload = torch.load(
      d0_bank_path, map_location="cpu", weights_only=False
    )
  else:
    d0_bank_path = None
  if args.neighbor_retention_bank is not None:
    neighbor_bank_path = args.neighbor_retention_bank.resolve()
    neighbor_retention_payload = torch.load(
      neighbor_bank_path, map_location="cpu", weights_only=False
    )
  else:
    neighbor_bank_path = None
  if d0_retention_payload is not None or neighbor_retention_payload is not None:
    resumed_retention_reference = runner.alg.retention_actor_reference is not None
    retention_bank_metadata = runner.alg.set_retention_anchor_banks(
      d0_payload=d0_retention_payload,
      neighbor_payload=neighbor_retention_payload,
      neighbor_domain=args.neighbor_domain,
    )
    if not resumed_retention_reference:
      start_checkpoint = (
        args.resume_online_checkpoint.resolve()
        if args.resume_online_checkpoint is not None
        else args.base_checkpoint.resolve()
      )
      start_checkpoint_sha256 = _file_sha256(start_checkpoint)
      bank_checkpoint_sha256 = {
        metadata["checkpoint_sha256"]
        for metadata in retention_bank_metadata.values()
      }
      if bank_checkpoint_sha256 != {start_checkpoint_sha256}:
        raise ValueError(
          "retention banks were not collected from the deployed start checkpoint"
        )
    for name, path in (
      ("d0", d0_bank_path),
      ("neighbor", neighbor_bank_path),
    ):
      if name in retention_bank_metadata and path is not None:
        retention_bank_metadata[name].update(
          path=str(path), file_sha256=_file_sha256(path)
        )

  # Keep the accepted bounded std and frozen normalizer, but use the original
  # base MLP as the cross-round drift/retention reference.
  base_actor_state = backtrack_actor_state(
    base_payload["actor_state_dict"], current_actor_state, 0.0
  )
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  baseline_eval_source: dict[str, Any] | None = None
  if args.reuse_baseline_eval is None:
    baseline_eval = _evaluate_state(
      runner,
      current_actor_state,
      domains=tuple(args.baseline_domains),
      num_envs=args.eval_num_envs,
      num_episodes=args.eval_num_episodes,
      seed=args.seed,
      device=args.device,
      repeats=args.gate_repeats,
      runtime_filter=args.gate_runtime_filter == "on",
    )
  else:
    reused_path = args.reuse_baseline_eval.resolve()
    baseline_eval = json.loads(reused_path.read_text())
    if set(baseline_eval) != set(args.baseline_domains):
      raise ValueError("reused baseline domains differ from --baseline-domains")
    expected_seeds = [args.seed + repeat for repeat in range(args.gate_repeats)]
    expected_episodes = args.eval_num_episodes * args.gate_repeats
    for domain in args.baseline_domains:
      result = baseline_eval[domain]
      if (
        result.get("task") != f"Unitree-G1-Stairs-Online-{domain}"
        or int(result.get("num_episodes", -1)) != expected_episodes
        or int(result.get("repeats", -1)) != args.gate_repeats
        or result.get("seeds") != expected_seeds
        or result.get("runtime_filter")
        is not (args.gate_runtime_filter == "on")
        or result.get("paired_one_initial_episode_per_env") is not True
      ):
        raise ValueError(f"reused {domain} baseline protocol differs")
    baseline_eval_source = {
      "path": str(reused_path),
      "sha256": _file_sha256(reused_path),
    }
  (output_dir / "baseline_ood_matrix.json").write_text(
    json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n"
  )
  if args.train_domain not in baseline_eval:
    raise ValueError("baseline domains must include the online train domain")
  if args.task_first_constrained:
    target_baseline = baseline_eval[args.train_domain]
    runner.alg.set_cost_budgets(
      fall_rate=float(target_baseline["fall_rate"]),
      intervention_per_riser=safety_demand_per_riser(target_baseline),
      intervention_slack=args.intervention_budget_slack,
    )

  burn_in: list[dict[str, Any]] = []
  burn_readiness_reasons: list[str] = []
  while (
    len(burn_in) < args.critic_burn_in_rounds
    or (
      bool(burn_in)
      and bool(burn_readiness_reasons)
      and len(burn_in) < args.critic_burn_in_max_rounds
    )
  ):
    obs, metrics = _collect_and_update(
      runner,
      obs,
      critic_only=True,
      hard_case_bank=hard_case_bank,
      hard_case_fraction=args.hard_case_fraction,
      neighbor_command_fraction=args.neighbor_command_fraction,
      neighbor_forward_scale_range=tuple(args.neighbor_forward_scale_range),
      neighbor_delay_step_offset_range=tuple(
        args.neighbor_delay_step_offset_range
      ),
      hard_case_pre_steps=args.hard_case_pre_steps,
      hard_case_generator=hard_case_generator,
    )
    if (
      metrics["normal_start_completed_episode_count"]
      < args.minimum_normal_complete_episodes
    ):
      raise RuntimeError(
        "rollout lacks complete normal-start episodes: "
        f"{metrics['normal_start_completed_episode_count']} < "
        f"{args.minimum_normal_complete_episodes}"
      )
    burn_readiness_reasons = []
    if (
      metrics["explained_variance_before_update"]
      < args.critic_min_explained_variance
    ):
      burn_readiness_reasons.append(
        "global task critic explained variance below threshold"
      )
    if args.task_first_constrained:
      burn_readiness_reasons.extend(
        critic_readiness_reasons(
          metrics,
          late_risers=tuple(args.late_critic_risers),
          minimum_samples_per_riser=args.critic_min_samples_per_late_riser,
          minimum_fall_events=args.critic_min_fall_events,
          maximum_risk_brier=args.risk_maximum_brier,
          minimum_risk_auc=args.risk_minimum_auc,
          minimum_pre_fall_cost_rise=args.minimum_pre_fall_cost_value_rise,
        )
      )
    metrics["critic_readiness_reasons"] = list(burn_readiness_reasons)
    burn_in.append(metrics)
    (output_dir / "critic_burn_in.json").write_text(
      json.dumps(burn_in, indent=2, sort_keys=True) + "\n"
    )
  if burn_in and burn_readiness_reasons:
    raise RuntimeError(
      "critic calibration did not satisfy local readiness gates: "
      + "; ".join(burn_readiness_reasons)
    )
  runner.alg.set_critic_only(False)

  thresholds = CandidateGateThresholds(
    maximum_target_fall_rate=args.maximum_target_fall_rate,
    require_task_improvement=args.task_first_constrained,
  )
  accepted_state = runner.snapshot_candidate_state()
  accepted_total_kl = _total_actor_kl(runner, base_actor_state)
  rounds: list[dict[str, Any]] = []
  for round_index in range(1, args.online_rounds + 1):
    before = runner.snapshot_candidate_state()
    old_actor_state = _actor_state(runner.alg.actor)
    obs, update_metrics = _collect_and_update(
      runner,
      obs,
      critic_only=False,
      hard_case_bank=hard_case_bank,
      hard_case_fraction=args.hard_case_fraction,
      neighbor_command_fraction=args.neighbor_command_fraction,
      neighbor_forward_scale_range=tuple(args.neighbor_forward_scale_range),
      neighbor_delay_step_offset_range=tuple(
        args.neighbor_delay_step_offset_range
      ),
      hard_case_pre_steps=args.hard_case_pre_steps,
      hard_case_generator=hard_case_generator,
    )
    if (
      update_metrics["normal_start_completed_episode_count"]
      < args.minimum_normal_complete_episodes
    ):
      raise RuntimeError(
        "online rollout lacks complete normal-start episodes: "
        f"{update_metrics['normal_start_completed_episode_count']} < "
        f"{args.minimum_normal_complete_episodes}"
      )
    full_candidate_actor_state = _actor_state(runner.alg.actor)
    old_total_kl = _total_actor_kl(
      runner, base_actor_state, actor_state=old_actor_state
    )
    candidate_variants: list[dict[str, Any]] = []
    for fraction in args.candidate_fractions:
      state = backtrack_actor_state(
        old_actor_state, full_candidate_actor_state, fraction
      )
      variant_metrics = _policy_step_metrics(runner, state, update_metrics)
      variant_total_kl = _total_actor_kl(
        runner, base_actor_state, actor_state=state
      )
      finite = runner.parameters_are_finite() and all(
        bool(torch.isfinite(value).all()) for value in state.values()
      )
      variant_reasons = candidate_precheck(
        update_metrics=variant_metrics,
        total_kl_from_base=variant_total_kl,
        parameters_finite=finite,
        thresholds=thresholds,
      )
      candidate_variants.append(
        {
          "fraction": fraction,
          "state": state,
          "update_metrics": variant_metrics,
          "total_kl_from_base": variant_total_kl,
          "precheck_reasons": variant_reasons,
          "screen_eval": {},
          "screen_score": None,
          "screen_score_delta": None,
        }
      )

    viable_variants = [
      variant for variant in candidate_variants if not variant["precheck_reasons"]
    ]
    screen_old_eval: dict[str, Any] = {}
    if len(viable_variants) > 1:
      screen_old_eval = _evaluate_state(
        runner,
        old_actor_state,
        domains=(args.train_domain,),
        num_envs=args.candidate_screen_num_envs,
        num_episodes=args.candidate_screen_num_envs,
        seed=args.seed + 1000 * round_index,
        device=args.gate_device,
        repeats=args.candidate_screen_repeats,
        runtime_filter=args.gate_runtime_filter == "on",
      )[args.train_domain]
      screen_old_score = safe_improvement_score(
        screen_old_eval,
        total_kl_from_base=old_total_kl,
      )["total"]
      for variant in viable_variants:
        screen_eval = _evaluate_state(
          runner,
          variant["state"],
          domains=(args.train_domain,),
          num_envs=args.candidate_screen_num_envs,
          num_episodes=args.candidate_screen_num_envs,
          seed=args.seed + 1000 * round_index,
          device=args.gate_device,
          repeats=args.candidate_screen_repeats,
          runtime_filter=args.gate_runtime_filter == "on",
        )[args.train_domain]
        if (
          screen_eval["initial_state_signatures"]
          != screen_old_eval["initial_state_signatures"]
        ):
          raise RuntimeError(
            "candidate-family screen initial states are not paired"
          )
        variant["screen_eval"] = screen_eval
        variant["screen_score"] = safe_improvement_score(
          screen_eval,
          total_kl_from_base=variant["total_kl_from_base"],
        )["total"]
        variant["screen_score_delta"] = (
          variant["screen_score"] - screen_old_score
        )

    if viable_variants:
      selected_variant = max(
        viable_variants,
        key=lambda variant: (
          float(variant["screen_score_delta"])
          if variant["screen_score_delta"] is not None
          else -float(variant["total_kl_from_base"])
        ),
      )
      candidate_actor_state = selected_variant["state"]
      update_metrics = selected_variant["update_metrics"]
      total_kl = float(selected_variant["total_kl_from_base"])
      precheck_reasons: list[str] = []
      selected_fraction: float | None = float(selected_variant["fraction"])
      runner.alg.actor.load_state_dict(candidate_actor_state, strict=True)
      update_metrics.update(
        runner.alg.adapt_retention_anchor_weights(
          d0_kl=update_metrics.get("d0_retention_anchor_kl"),
          neighbor_kl=update_metrics.get("neighbor_retention_anchor_kl"),
        )
      )
    else:
      candidate_actor_state = full_candidate_actor_state
      total_kl = _total_actor_kl(
        runner, base_actor_state, actor_state=candidate_actor_state
      )
      precheck_reasons = ["all candidate-family fractions failed precheck"]
      precheck_reasons.extend(
        f"fraction {variant['fraction']}: {reason}"
        for variant in candidate_variants
        for reason in variant["precheck_reasons"]
      )
      selected_fraction = None

    candidate_variant_records = [
      {key: value for key, value in variant.items() if key != "state"}
      for variant in candidate_variants
    ]

    candidate_path = output_dir / f"candidate_round_{round_index:03d}.pt"
    _save_checkpoint(
      runner,
      candidate_path,
      iteration=round_index,
      metadata={
        "accepted": False,
        "stage": "candidate",
        "update_metrics": update_metrics,
        "total_kl_from_base": total_kl,
        "selected_candidate_fraction": selected_fraction,
        "candidate_variants": candidate_variant_records,
      },
      hard_case_bank=hard_case_bank,
      hard_case_generator=hard_case_generator,
    )

    old_eval: dict[str, dict[str, Any]] = {}
    candidate_eval: dict[str, dict[str, Any]] = {}
    gate_intervals: dict[str, tuple[float, float, float]] = {}
    gate_scores: dict[str, dict[str, float]] = {}
    accepted = False
    reasons = list(precheck_reasons)
    if not reasons:
      old_eval = _evaluate_state(
        runner,
        old_actor_state,
        domains=("D0", args.train_domain, args.neighbor_domain),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
        runtime_filter=args.gate_runtime_filter == "on",
      )
      candidate_eval = _evaluate_state(
        runner,
        candidate_actor_state,
        domains=("D0", args.train_domain, args.neighbor_domain),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
        runtime_filter=args.gate_runtime_filter == "on",
      )
      accepted, reasons = candidate_gate(
        update_metrics=update_metrics,
        old_eval=old_eval,
        candidate_eval=candidate_eval,
        base_d0_success=baseline_eval["D0"]["success_rate"],
        old_total_kl_from_base=old_total_kl,
        total_kl_from_base=total_kl,
        parameters_finite=runner.parameters_are_finite(),
        thresholds=thresholds,
        target_domain=args.train_domain,
        retention_domain="D0",
        neighbor_domain=args.neighbor_domain,
      )
      gate_intervals = candidate_gate_intervals(
        old_eval=old_eval,
        candidate_eval=candidate_eval,
        thresholds=thresholds,
        target_domain=args.train_domain,
        retention_domain="D0",
        neighbor_domain=args.neighbor_domain,
        old_total_kl_from_base=old_total_kl,
        total_kl_from_base=total_kl,
      )
      gate_scores = {
        "old": safe_improvement_score(
          old_eval[args.train_domain],
          total_kl_from_base=old_total_kl,
        ),
        "candidate": safe_improvement_score(
          candidate_eval[args.train_domain],
          total_kl_from_base=total_kl,
        ),
      }

    adaptive_std_factor = 1.0
    if accepted:
      if args.adaptive_std:
        adaptive_std_factor = adaptive_cbf_std_factor(
          update_metrics["cbf_intervention_per_riser"],
          target_intervention_per_riser=args.target_intervention_per_riser,
          adaptation_rate=args.std_adaptation_rate,
          fall_count=update_metrics["fall_event_count"],
        )
        runner.alg.scale_exploration_std(adaptive_std_factor)
      # A selected interpolation/extrapolation no longer matches Adam moments
      # from the full step.  Every accepted round starts with fresh online
      # momentum and the configured base learning rate.
      runner.alg.reset_online_optimizer()
      accepted_state = runner.snapshot_candidate_state()
      accepted_total_kl = total_kl
      accepted_path = output_dir / f"accepted_round_{round_index:03d}.pt"
      _save_checkpoint(
        runner,
        accepted_path,
        iteration=round_index,
        metadata={
          "accepted": True,
          "update_metrics": update_metrics,
          "total_kl_from_base": total_kl,
          "old_eval": old_eval,
          "candidate_eval": candidate_eval,
          "gate_intervals": gate_intervals,
          "adaptive_std_factor": adaptive_std_factor,
          "selected_candidate_fraction": selected_fraction,
          "candidate_variants": candidate_variant_records,
        },
        hard_case_bank=hard_case_bank,
        hard_case_generator=hard_case_generator,
      )
    else:
      runner.restore_candidate_state(before)
      runner.reduce_after_rejection()
      accepted_state = runner.snapshot_candidate_state()
      # Report drift on the same populated rollout distribution used by the
      # candidate gate.  The value initialized before the first rollout is
      # based on empty storage and is not a meaningful final diagnostic.
      accepted_total_kl = old_total_kl

    record = {
      "round": round_index,
      "accepted": accepted,
      "rejection_reasons": reasons,
      "update_metrics": update_metrics,
      "total_kl_from_base": total_kl,
      "old_total_kl_from_base": old_total_kl,
      "old_eval": old_eval,
      "candidate_eval": candidate_eval,
      "gate_intervals": gate_intervals,
      "safe_improvement_scores": gate_scores,
      "adaptive_std_factor": adaptive_std_factor,
      "selected_candidate_fraction": selected_fraction,
      "candidate_variants": candidate_variant_records,
      "candidate_screen_old_eval": screen_old_eval,
      "candidate_checkpoint": str(candidate_path),
    }
    rounds.append(record)
    (output_dir / "online_rounds.json").write_text(
      json.dumps(rounds, indent=2, sort_keys=True) + "\n"
    )

  runner.restore_candidate_state(accepted_state)
  final_eval = _evaluate_state(
    runner,
    _actor_state(runner.alg.actor),
    domains=("D0", args.train_domain, args.neighbor_domain),
    num_envs=args.eval_num_envs,
    num_episodes=args.eval_num_episodes,
    seed=args.seed,
    device=args.gate_device,
    repeats=args.gate_repeats,
    runtime_filter=args.gate_runtime_filter == "on",
  )
  independence_audit: dict[str, Any] | None = None
  if args.independence_audit:
    final_actor_state = _actor_state(runner.alg.actor)
    if args.gate_runtime_filter == "on":
      filter_on = final_eval[args.train_domain]
      filter_off = _evaluate_state(
        runner,
        final_actor_state,
        domains=(args.train_domain,),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
        runtime_filter=False,
      )[args.train_domain]
    else:
      filter_off = final_eval[args.train_domain]
      filter_on = _evaluate_state(
        runner,
        final_actor_state,
        domains=(args.train_domain,),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
        runtime_filter=True,
      )[args.train_domain]
    independent, independence_reasons = cbf_independence_gate(
      filter_on_eval=filter_on,
      filter_off_eval=filter_off,
    )
    if (
      filter_on["initial_state_signatures"]
      != filter_off["initial_state_signatures"]
    ):
      independent = False
      independence_reasons.append(
        "CBF-on/off paired initial-state signature differs"
      )
    independence_audit = {
      "passed": independent,
      "reasons": independence_reasons,
      "filter_on": filter_on,
      "filter_off": filter_off,
    }
  final_path = output_dir / "accepted_final.pt"
  result = {
    "task": task,
    "train_domain": args.train_domain,
    "neighbor_domain": args.neighbor_domain,
    "seed": args.seed,
    "train_runtime_filter": args.train_runtime_filter,
    "gate_runtime_filter": args.gate_runtime_filter,
    "counterfactual_cbf_credit": args.train_runtime_filter == "off",
    "task_first_constrained": args.task_first_constrained,
    "task_reward_cbf_dual_weight": env_cfg.rewards["cbf_dual"].weight,
    "task_reward_fall_weight": env_cfg.rewards["fall_termination"].weight,
    "fall_cost_budget": runner.alg.fall_cost_budget,
    "intervention_cost_budget": runner.alg.intervention_cost_budget,
    "fall_multiplier_final": runner.alg.fall_multiplier,
    "intervention_multiplier_final": runner.alg.intervention_multiplier,
    "intervention_budget_slack": args.intervention_budget_slack,
    "base_anchor_weight": args.base_anchor_weight,
    "d0_retention_anchor_weight_initial": args.d0_retention_anchor_weight,
    "neighbor_retention_anchor_weight_initial": (
      args.neighbor_retention_anchor_weight
    ),
    "d0_retention_anchor_weight_final": (
      runner.alg.d0_retention_anchor_weight
    ),
    "neighbor_retention_anchor_weight_final": (
      runner.alg.neighbor_retention_anchor_weight
    ),
    "d0_retention_anchor_kl_budget": (
      runner.alg.d0_retention_anchor_kl_budget
    ),
    "neighbor_retention_anchor_kl_budget": (
      runner.alg.neighbor_retention_anchor_kl_budget
    ),
    "retention_anchor_adaptation_rate": (
      runner.alg.retention_anchor_adaptation_rate
    ),
    "maximum_retention_anchor_weight": (
      runner.alg.maximum_retention_anchor_weight
    ),
    "retention_anchor_batch_size": runner.alg.retention_anchor_batch_size,
    "retention_anchor_banks": retention_bank_metadata,
    "retention_actor_input_only": bool(retention_bank_metadata),
    "ppo_policy_gradient_domains": [args.train_domain],
    "hard_case_policy_weight": args.hard_case_policy_weight,
    "correction_distillation_weight": args.correction_distillation_weight,
    "correction_success_horizon": args.correction_success_horizon,
    "risk_horizon": args.risk_horizon,
    "candidate_fractions": args.candidate_fractions,
    "candidate_screen_num_envs": args.candidate_screen_num_envs,
    "candidate_screen_repeats": args.candidate_screen_repeats,
    "minimum_normal_complete_episodes": args.minimum_normal_complete_episodes,
    "late_critic_risers": args.late_critic_risers,
    "critic_min_samples_per_late_riser": (
      args.critic_min_samples_per_late_riser
    ),
    "critic_min_fall_events": args.critic_min_fall_events,
    "risk_maximum_brier": args.risk_maximum_brier,
    "risk_minimum_auc": args.risk_minimum_auc,
    "minimum_pre_fall_cost_value_rise": (
      args.minimum_pre_fall_cost_value_rise
    ),
    "hard_case_fraction": args.hard_case_fraction,
    "neighbor_command_fraction": args.neighbor_command_fraction,
    "neighbor_forward_scale_range": args.neighbor_forward_scale_range,
    "neighbor_delay_step_offset_range": args.neighbor_delay_step_offset_range,
    "hard_case_pre_steps": args.hard_case_pre_steps,
    "hard_case_bank_size": len(hard_case_bank),
    "hard_case_bank_total_events": hard_case_bank.total_added,
    "adaptive_std": args.adaptive_std,
    "target_intervention_per_riser": args.target_intervention_per_riser,
    "maximum_target_fall_rate": args.maximum_target_fall_rate,
    "paired_interval_method": "bootstrap",
    "fall_penalty_weight": env_cfg.rewards["fall_termination"].weight,
    "warm_start": warm_start,
    "base_checkpoint": str(args.base_checkpoint),
    "resume_hard_case_bank": args.resume_hard_case_bank,
    "critic_burn_in": burn_in,
    "critic_min_explained_variance": args.critic_min_explained_variance,
    "baseline_eval": baseline_eval,
    "baseline_eval_source": baseline_eval_source,
    "rounds": rounds,
    "accepted_total_kl_from_base": accepted_total_kl,
    "final_eval": final_eval,
    "final_cbf_independence_audit": independence_audit,
    "final_checkpoint": str(final_path),
  }
  _save_checkpoint(
    runner,
    final_path,
    iteration=args.online_rounds,
    metadata=result,
    hard_case_bank=hard_case_bank,
    hard_case_generator=hard_case_generator,
  )
  (output_dir / "online_refinement_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  env.close()


if __name__ == "__main__":
  main()

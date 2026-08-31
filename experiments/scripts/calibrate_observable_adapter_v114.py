"""Calibrate the closest v92 Eq. (23) adapter with one held-out screen.

v92 was the only deployable CBF adapter to reach 47/64 on an untouched
filter-off evaluation.  v114 does not retrain or search a new direction.  It
interpolates the provenance-locked v92 actor delta at four predeclared scales,
evaluates the groups in one simulator pass, and runs one independent gate only
when the selected screen group reaches the fixed 75 percent threshold.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from cbf_teacher_v31_protocol import (
  CLEARANCE_BARRIER_SLOPE,
  CONTEXTS,
  FILTER_ALPHA,
  RECOVERY_DISTANCE_M,
  TASK_ID,
  environment_parameters,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_observable_cbf_adapter_v49 import _expand_actor_state
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _initial_state_signature,
  _seed_everything,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "heldout-scaled-observable-cbf-adapter-v114"


def interpolate_actor_state(
  base: dict[str, torch.Tensor],
  adapter: dict[str, torch.Tensor],
  scale: float,
) -> dict[str, torch.Tensor]:
  """Interpolate the full actor state while preserving non-floating tensors."""
  if set(base) != set(adapter) or not 0.0 <= scale <= 2.0:
    raise ValueError("v114 actor states or scale are incompatible")
  output: dict[str, torch.Tensor] = {}
  for key in base:
    base_value = base[key].detach()
    adapter_value = adapter[key].detach().to(base_value.device)
    if base_value.shape != adapter_value.shape:
      raise ValueError(f"v114 actor tensor shape differs for {key!r}")
    if torch.is_floating_point(base_value):
      output[key] = (
        base_value + float(scale) * (adapter_value - base_value)
      ).clone()
    else:
      if not torch.equal(base_value, adapter_value):
        raise ValueError(f"v114 non-floating actor tensor differs for {key!r}")
      output[key] = base_value.clone()
  return output


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--search-config", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--adapter-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-adapter-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--screen-scales", default="0,0.5,1.0,1.5")
  parser.add_argument("--episodes-per-scale", type=int, default=64)
  parser.add_argument("--screen-seed", type=int, required=True)
  parser.add_argument("--independent-gate-envs", type=int, default=256)
  parser.add_argument("--independent-gate-seed", type=int, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_scales(raw: str) -> list[float]:
  try:
    scales = [float(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v114 scales must be comma-separated numbers") from exc
  if (
    len(scales) < 2
    or len(set(scales)) != len(scales)
    or scales[0] != 0.0
    or any(not 0.0 <= scale <= 2.0 for scale in scales)
  ):
    raise ValueError("v114 scales must be unique, start at zero, and lie in [0, 2]")
  return scales


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v114 checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _subset_observations(observations, indices: torch.Tensor):
  return {key: value[indices] for key, value in observations.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def _screen_scaled_actors(
  actors: list[torch.nn.Module],
  runner,
  base_env,
  action_term,
  *,
  scales: list[float],
  episodes_per_scale: int,
  seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
  expected_envs = len(scales) * episodes_per_scale
  if base_env.num_envs < expected_envs or len(actors) != len(scales):
    raise ValueError("v114 actor groups do not fit the environment")
  considered = torch.arange(base_env.num_envs, device=base_env.device) < expected_envs
  group_by_env = torch.full(
    (base_env.num_envs,), -1, dtype=torch.long, device=base_env.device
  )
  group_by_env[considered] = torch.arange(
    len(scales), device=base_env.device
  ).repeat_interleave(episodes_per_scale)
  action_term.set_runtime_filter_mask(
    torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  )
  _seed_everything(seed)
  base_env.seed(seed)
  observations, _ = runner.env.reset()
  signature = _initial_state_signature(
    observations,
    base_env,
    action_term,
    base_env.command_manager.get_term("twist"),
  )
  active = considered.clone()
  success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  fell = torch.zeros_like(success)
  steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
  reached_risers = torch.zeros_like(steps)
  counterfactual_count = torch.zeros(
    len(scales), dtype=torch.long, device=base_env.device
  )
  transition_count = torch.zeros_like(counterfactual_count)
  maximum_steps = int(base_env.max_episode_length) + 2
  for actor in actors:
    actor.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      actions = torch.zeros(
        (base_env.num_envs, 12), dtype=torch.float32, device=base_env.device
      )
      for group, actor in enumerate(actors):
        ids = (group_by_env == group).nonzero(as_tuple=False).flatten()
        actions[ids] = actor(
          _subset_observations(observations, ids), stochastic_output=False
        )
      next_observations, _, dones, extras = runner.env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        reached_risers[ids] = torch.maximum(
          reached_risers[ids], extras["online_stair_index"][ids].long()
        )
        groups = group_by_env[ids]
        transition_count.scatter_add_(
          0, groups, torch.ones_like(groups, dtype=torch.long)
        )
        counterfactual_count.scatter_add_(
          0, groups, extras["cbf_would_intervene"][ids].long()
        )
      steps += active.long()
      completed = dones.bool() & active
      if bool(completed.any()):
        reached_top = base_env.termination_manager.get_term("reached_top").bool()
        success[completed] = reached_top[completed]
        fell[completed] = extras["online_fell"][completed].bool()
        active &= ~completed
        if not bool(active.any()):
          observations = next_observations
          break
      observations = next_observations
  if bool(active.any()):
    raise RuntimeError("v114 did not finish every screen episode")

  summaries: list[dict[str, Any]] = []
  rows: list[dict[str, Any]] = []
  for group, scale in enumerate(scales):
    mask = group_by_env == group
    ids = mask.nonzero(as_tuple=False).flatten()
    success_count = int(success[mask].sum())
    summaries.append(
      {
        "scale": scale,
        "screen_seed": seed,
        "initial_state_signature": signature,
        "episode_count": episodes_per_scale,
        "success_count": success_count,
        "success_rate": success_count / episodes_per_scale,
        "fall_count": int(fell[mask].sum()),
        "fall_rate": float(fell[mask].float().mean()),
        "mean_reached_riser": float(reached_risers[mask].float().mean()),
        "counterfactual_intervention_fraction": float(
          counterfactual_count[group] / transition_count[group].clamp_min(1)
        ),
      }
    )
    rows.extend(
      {
        "scale": scale,
        "group_environment_id": int(index),
        "environment_id": int(environment_id),
        "success": bool(success[environment_id]),
        "fell": bool(fell[environment_id]),
        "steps": int(steps[environment_id]),
        "reached_risers": int(reached_risers[environment_id]),
      }
      for index, environment_id in enumerate(ids)
    )
  selected = max(
    summaries,
    key=lambda item: (
      item["success_rate"],
      item["mean_reached_riser"],
      -item["scale"],
    ),
  )
  return summaries, rows, float(selected["scale"])


def _evaluate_actor(
  actor,
  runner,
  base_env,
  action_term,
  *,
  seed: int,
  gate_envs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  if not 1 <= gate_envs <= base_env.num_envs:
    raise ValueError("v114 independent gate count is invalid")
  action_term.set_runtime_filter_mask(
    torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  )
  _seed_everything(seed)
  base_env.seed(seed)
  observations, _ = runner.env.reset()
  signature = _initial_state_signature(
    observations,
    base_env,
    action_term,
    base_env.command_manager.get_term("twist"),
  )
  considered = torch.arange(base_env.num_envs, device=base_env.device) < gate_envs
  active = considered.clone()
  success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  fell = torch.zeros_like(success)
  steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
  reached_risers = torch.zeros_like(steps)
  counterfactual_count = transition_count = 0
  maximum_steps = int(base_env.max_episode_length) + 2
  actor.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      actions = actor(observations, stochastic_output=False)
      next_observations, _, dones, extras = runner.env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        reached_risers[ids] = torch.maximum(
          reached_risers[ids], extras["online_stair_index"][ids].long()
        )
        counterfactual_count += int(extras["cbf_would_intervene"][ids].sum())
        transition_count += len(ids)
      steps += active.long()
      completed = dones.bool() & active
      if bool(completed.any()):
        reached_top = base_env.termination_manager.get_term("reached_top").bool()
        success[completed] = reached_top[completed]
        fell[completed] = extras["online_fell"][completed].bool()
        active &= ~completed
        if not bool(active.any()):
          observations = next_observations
          break
      observations = next_observations
  if bool(active.any()):
    raise RuntimeError("v114 did not finish the independent gate")
  ids = considered.nonzero(as_tuple=False).flatten().cpu()
  rows = [
    {
      "environment_id": int(environment_id),
      "success": bool(success[environment_id]),
      "fell": bool(fell[environment_id]),
      "steps": int(steps[environment_id]),
      "reached_risers": int(reached_risers[environment_id]),
    }
    for environment_id in ids
  ]
  success_count = int(success[considered].sum())
  return {
    "seed": seed,
    "num_episodes": gate_envs,
    "initial_state_signature": signature,
    "runtime_filter": False,
    "success_count": success_count,
    "success_rate": success_count / gate_envs,
    "fall_count": int(fell[considered].sum()),
    "fall_rate": float(fell[considered].float().mean()),
    "mean_reached_riser": float(reached_risers[considered].float().mean()),
    "counterfactual_intervention_fraction": (
      counterfactual_count / max(1, transition_count)
    ),
    "passed_75_percent": success_count / gate_envs >= 0.75,
  }, rows


def main() -> None:
  args = _parse_args()
  scales = _parse_scales(args.screen_scales)
  if args.episodes_per_scale < 1 or args.independent_gate_envs < 1:
    raise ValueError("v114 episode counts must be positive")
  repo = args.repo.resolve()
  base_checkpoint = args.base_checkpoint.resolve()
  adapter_checkpoint = args.adapter_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v114 requires a clean committed worktree")
  if output.exists():
    raise FileExistsError(output)
  if (
    not base_checkpoint.is_file()
    or not adapter_checkpoint.is_file()
    or not args.search_config.resolve().is_file()
  ):
    raise FileNotFoundError("v114 checkpoint or search protocol is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v114 velocity-CBF protocol differs")
  base_sha = file_sha256(base_checkpoint)
  adapter_sha = file_sha256(adapter_checkpoint)
  if base_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v114 base checkpoint SHA-256 differs")
  if adapter_sha != _normalized_sha(args.expected_adapter_sha256):
    raise RuntimeError("v114 adapter checkpoint SHA-256 differs")
  output.mkdir(parents=True)
  started = time.monotonic()
  source_commit = _git(repo, "rev-parse", "HEAD")
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": METHOD_ID,
      "git_commit": source_commit,
      "base_checkpoint_sha256": base_sha,
      "adapter_checkpoint_sha256": adapter_sha,
      "screen_scales": scales,
      "episodes_per_scale": args.episodes_per_scale,
      "screen_seed": args.screen_seed,
      "independent_gate_policy": "run_only_if_selected_screen_rate_gte_0.75",
    },
  )

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import (
    configure_deployable_cbf_geometry_observation,
    configure_deployable_cbf_geometry_runner,
  )
  from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
  from src.tasks.stairs_cbf.paper_dual_v35 import configure_paper_dual_reward
  from src.tasks.stairs_cbf.velocity_cbf_action import (
    InstrumentedCurrentVelocityCbfAction,
    configure_v34_cbf,
  )

  total_screen_envs = len(scales) * args.episodes_per_scale
  num_envs = max(total_screen_envs, args.independent_gate_envs)
  _seed_everything(args.screen_seed)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  shift = configure_v31_context(
    env_cfg,
    context=args.context,
    runtime_filter=False,
    context_spec=environment_parameters(args.context),
    clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
    recovery_distance_m=RECOVERY_DISTANCE_M,
    filter_alpha=FILTER_ALPHA,
  )
  cbf = configure_v34_cbf(
    env_cfg,
    mode=CURRENT_CBF_MODE,
    runtime_filter=False,
    parameters=None,
    measure_compute_time=False,
  )
  reward = configure_paper_dual_reward(
    env_cfg, "raw_moderate", runtime_filter_during_training=True
  )
  geometry = configure_deployable_cbf_geometry_observation(env_cfg)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = args.screen_seed
  agent_cfg = load_rl_cfg(TASK_ID)
  configure_deployable_cbf_geometry_runner(agent_cfg)
  agent_cfg.seed = args.screen_seed
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("v114 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v114 requires the current velocity-CBF action")

  try:
    base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    adapter_payload = torch.load(
      adapter_checkpoint, map_location="cpu", weights_only=False
    )
    expanded_base, expansion = _expand_actor_state(
      base_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    adapter_state = adapter_payload["actor_state_dict"]
    if int(adapter_state["mlp.0.weight"].shape[1]) != 410:
      raise RuntimeError("v114 source adapter must use the 410-D v92 interface")
    actors: list[torch.nn.Module] = []
    scaled_states: list[dict[str, torch.Tensor]] = []
    actor_hashes: list[str] = []
    for scale in scales:
      state = interpolate_actor_state(expanded_base, adapter_state, scale)
      actor = copy.deepcopy(runner.alg.actor).to(args.device)
      actor.load_state_dict(state, strict=True)
      actor.eval()
      for parameter in actor.parameters():
        parameter.requires_grad_(False)
      actors.append(actor)
      scaled_states.append(state)
      actor_hashes.append(actor_state_sha256(state))
    screen, screen_rows, selected_scale = _screen_scaled_actors(
      actors,
      runner,
      base_env,
      action_term,
      scales=scales,
      episodes_per_scale=args.episodes_per_scale,
      seed=args.screen_seed,
    )
    selected_index = scales.index(selected_scale)
    selected_screen = screen[selected_index]
    selected_state = scaled_states[selected_index]
    selected_actor = actors[selected_index]
    candidate_payload = copy.deepcopy(adapter_payload)
    candidate_payload["actor_state_dict"] = {
      key: value.detach().cpu() for key, value in selected_state.items()
    }
    candidate_payload["actor_observation_interface"] = (
      "legacy_405_plus_deployable_cbf_geometry_5"
    )
    infos = dict(candidate_payload.get("infos") or {})
    infos[METHOD_ID] = {
      "source_adapter_checkpoint_sha256": adapter_sha,
      "base_checkpoint_sha256": base_sha,
      "selected_scale": selected_scale,
      "selection_rule": "success_then_progress_then_smaller_scale",
      "selected_screen": selected_screen,
    }
    candidate_payload["infos"] = infos
    candidate_path = output / "candidate.pt"
    _atomic_torch(candidate_path, candidate_payload)
    _atomic_json(output / "screen_summary.json", screen)
    _write_csv(output / "screen_episodes.csv", screen_rows)

    independent_gate = None
    if selected_screen["success_rate"] >= 0.75:
      independent_gate, gate_rows = _evaluate_actor(
        selected_actor,
        runner,
        base_env,
        action_term,
        seed=args.independent_gate_seed,
        gate_envs=args.independent_gate_envs,
      )
      independent_gate["selected_scale"] = selected_scale
      independent_gate["candidate_checkpoint_sha256"] = file_sha256(candidate_path)
      _atomic_json(output / "independent_gate_summary.json", independent_gate)
      _write_csv(output / "independent_gate_episodes.csv", gate_rows)

    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": source_commit,
      "context": args.context,
      "base_checkpoint": str(base_checkpoint),
      "base_checkpoint_sha256": base_sha,
      "source_adapter_checkpoint": str(adapter_checkpoint),
      "source_adapter_checkpoint_sha256": adapter_sha,
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "candidate_actor_sha256": actor_hashes[selected_index],
      "screen_scales": scales,
      "scaled_actor_sha256": dict(zip((str(value) for value in scales), actor_hashes)),
      "screen_seed": args.screen_seed,
      "screen_episode_count_per_scale": args.episodes_per_scale,
      "scale_screen": screen,
      "selected_scale": selected_scale,
      "selected_screen": selected_screen,
      "selection_rule": "success_rate_then_mean_reached_riser_then_smaller_scale",
      "independent_gate_threshold": 0.75,
      "independent_gate_run": independent_gate is not None,
      "independent_gate": independent_gate,
      "selected": bool(
        independent_gate is not None and independent_gate["passed_75_percent"]
      ),
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "actor_expansion": expansion,
      "all_finite": all(
        math.isfinite(float(row["success_rate"]))
        and math.isfinite(float(row["mean_reached_riser"]))
        for row in screen
      ),
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()

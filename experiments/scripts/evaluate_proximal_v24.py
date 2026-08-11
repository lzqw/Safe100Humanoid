"""Deterministic contact-aware evaluator for v24 Contact Completion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from proximal_v23_io import actor_state_sha256
from proximal_v24_protocol import (
    CONTEXT_MODE,
    TARGET_FAILURE_TYPE,
    pure_contact_context_audit,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deployment-context", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _initial_state_signature(obs, base_env, action_term, command_term) -> str:
    signature = hashlib.sha256()
    tensors = [
        obs["actor"],
        base_env.scene["robot"].data.root_link_pos_w,
        base_env.scene["robot"].data.root_link_quat_w,
        base_env.scene["robot"].data.joint_pos,
        base_env.command_manager.get_command("twist"),
        getattr(
            command_term,
            "raw_command",
            base_env.command_manager.get_command("twist"),
        ),
        getattr(
            command_term,
            "delay_steps",
            torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device),
        ),
        getattr(
            command_term,
            "_delay_queue",
            torch.zeros(base_env.num_envs, 1, 3, device=base_env.device),
        ),
        getattr(
            action_term,
            "_deployment_action_queue",
            torch.zeros(
                base_env.num_envs,
                1,
                action_term.action_dim,
                device=base_env.device,
            ),
        ),
    ]
    for tensor in tensors:
        signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return signature.hexdigest()


def _masked_max(current: torch.Tensor, value: torch.Tensor, active: torch.Tensor):
    return torch.where(active, torch.maximum(current, value), current)


def main() -> None:
    args = _parse_args()
    if args.num_envs != args.num_episodes or args.num_envs < 1:
        raise ValueError("v24 pairing requires one initial episode per environment")
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(repo))
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import src.tasks  # noqa: F401
    from src.tasks.stairs_cbf.deployment_context import (
        load_frozen_deployment_context,
    )
    from src.tasks.stairs_cbf.hard_cases import classify_v19_failure_mode
    from src.tasks.stairs_cbf.mdp import specialist_failure_signal_components
    from src.tasks.stairs_cbf.proximal_context import apply_cbf_proximal_context

    cfg = load_env_cfg(args.task, play=True)
    context_metadata = None
    context_audit = None
    if args.deployment_context is not None:
        context = load_frozen_deployment_context(args.deployment_context.resolve())
        context_audit = pure_contact_context_audit(context)
        context_metadata = apply_cbf_proximal_context(cfg, context, role="target")
    elif args.task.endswith("DQHMED"):
        raise ValueError("the v24 target task requires its frozen contact context")
    cfg.rewards.pop("specialist_failure_signal", None)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.actions["joint_pos"].enabled = True
    if "deployable_failure" in cfg.observations:
        raise RuntimeError("v24 evaluator contains a forbidden failure observation")
    if "specialist_failure_signal" in cfg.rewards:
        raise RuntimeError("v24 evaluator contains a specialist reward term")

    base_env = ManagerBasedRlEnv(cfg, device=args.device)
    agent_cfg = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task)
    if runner_cls is None:
        raise RuntimeError("v24 evaluation task has no runner")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
    try:
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        if int(runner.alg.actor.obs_dim) != 405:
            raise RuntimeError("v24 evaluator actor is not the original 405-D policy")
        actor_hash = actor_state_sha256(runner.alg.actor.state_dict())
        policy = runner.get_inference_policy(args.device)
        obs, _ = env.reset()
        action_term = base_env.action_manager.get_term("joint_pos")
        command_term = base_env.command_manager.get_term("twist")
        if action_term.cfg.enabled is not True:
            raise RuntimeError("v24 runtime CBF is disabled")
        initial_signature = _initial_state_signature(
            obs, base_env, action_term, command_term
        )
        num_risers = int(action_term._edge_x.shape[-1])
        active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        returns = torch.zeros(args.num_envs, device=args.device)
        steps = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        telemetry_steps = torch.zeros_like(steps)
        max_riser = torch.zeros_like(steps)
        intervention_count = torch.zeros_like(steps)
        correction_sum = torch.zeros(args.num_envs, device=args.device)
        slip_signal_sum = torch.zeros_like(correction_sum)
        contact_mismatch_sum = torch.zeros_like(correction_sum)
        max_left_slip = torch.zeros_like(correction_sum)
        max_right_slip = torch.zeros_like(correction_sum)
        max_centerline_error = torch.zeros_like(correction_sum)
        max_heading_error = torch.zeros_like(correction_sum)
        correction_max = torch.zeros_like(correction_sum)
        side_edge_breach = torch.zeros_like(active)
        first_lateral_event_step = torch.full_like(steps, -1)
        first_contact_event_step = torch.full_like(steps, -1)
        contact_instability_streak = torch.zeros_like(steps)
        completed: list[dict[str, Any]] = []
        maximum_steps = int(base_env.max_episode_length) + 2
        with torch.inference_mode():
            for _ in range(maximum_steps):
                robot = base_env.scene["robot"]
                terrain = base_env.scene.terrain
                if terrain is None:
                    raise RuntimeError("v24 evaluation requires stair terrain")
                root_x = robot.data.root_link_pos_w[:, 0:1]
                edge_x = action_term._edge_x[
                    terrain.terrain_levels,
                    terrain.terrain_types,
                ]
                current_riser = torch.sum(root_x >= edge_x, dim=1)
                max_riser = torch.where(
                    active,
                    torch.maximum(max_riser, current_riser),
                    max_riser,
                )
                components = specialist_failure_signal_components(base_env)
                active_float = active.float()
                slip_signal_sum += components["slip"] * active_float
                contact_mismatch_sum += components["contact_mismatch"] * active_float
                max_left_slip = _masked_max(
                    max_left_slip, components["left_slip"], active
                )
                max_right_slip = _masked_max(
                    max_right_slip, components["right_slip"], active
                )
                centerline_error = getattr(
                    command_term,
                    "centerline_error",
                    torch.zeros(args.num_envs, device=args.device),
                ).abs()
                heading_error = getattr(
                    command_term,
                    "heading_error",
                    torch.zeros(args.num_envs, device=args.device),
                ).abs()
                max_centerline_error = _masked_max(
                    max_centerline_error, centerline_error, active
                )
                max_heading_error = _masked_max(
                    max_heading_error, heading_error, active
                )
                patches = terrain.flat_patches["stair_targets"][
                    terrain.terrain_levels, terrain.terrain_types
                ]
                center_y = patches[:, 0, 1]
                foot_y = robot.data.site_pos_w[:, action_term._site_local_ids, 1]
                stair_half_width = float(
                    getattr(command_term.cfg, "stair_half_width", 1.20)
                )
                foot_edge_breach = (
                    torch.max(torch.abs(foot_y - center_y.unsqueeze(1)), dim=1).values
                    >= stair_half_width
                )
                side_edge_breach |= active & (
                    (centerline_error >= stair_half_width) | foot_edge_breach
                )
                severe_contact_slip = (
                    torch.maximum(components["left_slip"], components["right_slip"])
                    >= 0.50
                )
                contact_instability_streak = torch.where(
                    active & severe_contact_slip,
                    contact_instability_streak + 1,
                    torch.where(
                        active,
                        torch.zeros_like(contact_instability_streak),
                        contact_instability_streak,
                    ),
                )
                new_contact_event = (
                    active
                    & (first_contact_event_step < 0)
                    & (contact_instability_streak >= 3)
                )
                first_contact_event_step = torch.where(
                    new_contact_event,
                    (telemetry_steps - 2).clamp_min(0),
                    first_contact_event_step,
                )
                lateral_event = (
                    (centerline_error >= (2.0 / 3.0) * stair_half_width)
                    | (heading_error >= math.pi / 2.0)
                    | foot_edge_breach
                )
                first_lateral_event_step = torch.where(
                    active & (first_lateral_event_step < 0) & lateral_event,
                    telemetry_steps,
                    first_lateral_event_step,
                )
                telemetry_steps += active.long()

                actions = policy(obs)
                next_obs, rewards, dones, extras = env.step(actions)
                returns += rewards * active_float
                steps += active.long()
                intervention_count += action_term.intervened.long() * active.long()
                correction_sum += action_term.target_intervention_norm * active_float
                correction_max = _masked_max(
                    correction_max, action_term.target_intervention_norm, active
                )
                record_mask = dones.bool() & active
                if bool(record_mask.any()):
                    extras = dict(extras)
                    fell = extras.get(
                        "online_fell",
                        base_env.termination_manager.get_term("fell_over"),
                    ).bool()
                    timed_out = extras.get(
                        "time_outs",
                        base_env.termination_manager.get_term("time_out"),
                    ).bool()
                    success = base_env.termination_manager.get_term(
                        "reached_top"
                    ).bool()
                    for env_id in (
                        record_mask.nonzero(as_tuple=False).flatten().tolist()
                    ):
                        reached = int(max_riser[env_id])
                        episode_steps = max(1, int(steps[env_id]))
                        observed_steps = max(1, int(telemetry_steps[env_id]))
                        failure_type = "success"
                        if bool(fell[env_id]):
                            failure_type = classify_v19_failure_mode(
                                specialist_mode=CONTEXT_MODE,
                                side_edge_breach=bool(side_edge_breach[env_id]),
                                max_abs_centerline_error=float(
                                    max_centerline_error[env_id]
                                ),
                                max_abs_heading_error=float(max_heading_error[env_id]),
                                correction_max=float(correction_max[env_id]),
                                maximum_left_slip_speed=float(max_left_slip[env_id]),
                                maximum_right_slip_speed=float(max_right_slip[env_id]),
                                mean_contact_mismatch=float(
                                    contact_mismatch_sum[env_id] / observed_steps
                                ),
                                stair_half_width=stair_half_width,
                                first_lateral_event_step=(
                                    None
                                    if int(first_lateral_event_step[env_id]) < 0
                                    else int(first_lateral_event_step[env_id])
                                ),
                                first_contact_event_step=(
                                    None
                                    if int(first_contact_event_step[env_id]) < 0
                                    else int(first_contact_event_step[env_id])
                                ),
                            )
                        elif bool(timed_out[env_id]):
                            failure_type = "timeout"
                        elif not bool(success[env_id]):
                            failure_type = "other_non_success"
                        recovery_takeover = bool(success[env_id]) and bool(
                            intervention_count[env_id] > 0
                        )
                        completed.append(
                            {
                                "evaluation_seed": args.seed,
                                "environment_id": env_id,
                                "success": bool(success[env_id]),
                                "fell": bool(fell[env_id]),
                                "timed_out": bool(timed_out[env_id]),
                                "failure_type": failure_type,
                                "return": float(returns[env_id]),
                                "steps": episode_steps,
                                "max_riser": reached,
                                "completion_fraction": reached / num_risers,
                                "mean_slip_signal": float(
                                    slip_signal_sum[env_id] / observed_steps
                                ),
                                "maximum_left_contact_slip_speed": float(
                                    max_left_slip[env_id]
                                ),
                                "maximum_right_contact_slip_speed": float(
                                    max_right_slip[env_id]
                                ),
                                "mean_contact_mismatch": float(
                                    contact_mismatch_sum[env_id] / observed_steps
                                ),
                                "intervention_count": int(intervention_count[env_id]),
                                "intervention_per_riser": float(
                                    intervention_count[env_id] / max(1, reached)
                                ),
                                "mean_correction_norm": float(
                                    correction_sum[env_id] / episode_steps
                                ),
                                "recovery_takeover": recovery_takeover,
                            }
                        )
                    active &= ~record_mask
                    if not bool(active.any()):
                        obs = next_obs
                        break
                obs = next_obs
        if bool(active.any()) or len(completed) != args.num_episodes:
            raise RuntimeError("v24 evaluator did not finish every initial episode")
        completed.sort(key=lambda row: int(row["environment_id"]))
        failure_type_counts: dict[str, int] = {}
        for row in completed:
            if row["fell"]:
                failure_type_counts[row["failure_type"]] = (
                    failure_type_counts.get(row["failure_type"], 0) + 1
                )
        fall_count = sum(bool(row["fell"]) for row in completed)
        contact_fall_count = failure_type_counts.get(TARGET_FAILURE_TYPE, 0)
        summary = {
            "schema_version": 1,
            "task": args.task,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "num_episodes": len(completed),
            "runtime_filter": True,
            "deterministic_policy_mean": True,
            "one_initial_episode_per_env": True,
            "original_observation_interface": True,
            "actor_observation_dim": 405,
            "actor_state_sha256": actor_hash,
            "initial_state_signature": initial_signature,
            "success_rate": sum(row["success"] for row in completed) / len(completed),
            "fall_rate": fall_count / len(completed),
            "timeout_rate": sum(row["timed_out"] for row in completed) / len(completed),
            "mean_return": sum(row["return"] for row in completed) / len(completed),
            "mean_reached_riser": sum(row["max_riser"] for row in completed)
            / len(completed),
            "mean_slip_signal": sum(row["mean_slip_signal"] for row in completed)
            / len(completed),
            "mean_contact_mismatch": sum(
                row["mean_contact_mismatch"] for row in completed
            )
            / len(completed),
            "intervention_per_riser": sum(
                row["intervention_per_riser"] for row in completed
            )
            / len(completed),
            "mean_correction_norm": sum(
                row["mean_correction_norm"] for row in completed
            )
            / len(completed),
            "recovery_takeover_count": sum(
                row["recovery_takeover"] for row in completed
            ),
            "recovery_takeover_rate": sum(row["recovery_takeover"] for row in completed)
            / len(completed),
            "failure_type_counts": failure_type_counts,
            "contact_fall_count": contact_fall_count,
            "contact_failure_purity_over_falls": contact_fall_count
            / max(1, fall_count),
            "deployment_context": context_metadata,
            "pure_contact_context_audit": context_audit,
        }
    finally:
        env.close()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
        writer.writeheader()
        writer.writerows(completed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

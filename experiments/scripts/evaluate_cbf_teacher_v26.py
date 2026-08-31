"""Evaluate one v26 policy/filter condition on paired initial episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cbf_teacher_v26_protocol import ENVIRONMENT_VARIANT, TASK_ID
from proximal_v23_io import actor_state_sha256, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--riser-height", type=float, required=True)
    parser.add_argument("--clearance-slope", type=float, default=0.0)
    parser.add_argument("--recovery-distance", type=float, default=0.15)
    parser.add_argument("--filter-alpha", type=float, default=10.0)
    parser.add_argument("--runtime-filter", choices=("on", "off"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _initial_state_signature(obs, base_env, action_term, command_term) -> str:
    signature = hashlib.sha256()
    terrain = base_env.scene.terrain
    if terrain is None:
        raise RuntimeError("v26 initial-state signature requires stair terrain")
    tensors = (
        obs["actor"],
        base_env.scene.env_origins,
        base_env.scene["robot"].data.root_link_pos_w,
        base_env.scene["robot"].data.root_link_quat_w,
        base_env.scene["robot"].data.root_link_lin_vel_w,
        base_env.scene["robot"].data.root_link_ang_vel_w,
        base_env.scene["robot"].data.joint_pos,
        base_env.scene["robot"].data.joint_vel,
        terrain.terrain_levels,
        terrain.terrain_types,
        action_term._edge_x[terrain.terrain_levels, terrain.terrain_types],
        action_term._edge_top_z[terrain.terrain_levels, terrain.terrain_types],
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
    )
    for tensor in tensors:
        signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return signature.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if args.num_envs != args.num_episodes or args.num_envs < 1:
        raise ValueError("v26 requires one initial episode per environment")
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout:
        raise RuntimeError("v26 evaluation requires a clean committed worktree")
    runtime_filter = args.runtime_filter == "on"
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
    from src.tasks.stairs_cbf.teacher_v26 import (
        HigherRiserCbfAction,
        configure_v26_higher_riser,
    )

    task = TASK_ID
    env_cfg = load_env_cfg(task, play=True)
    shift_metadata = configure_v26_higher_riser(
        env_cfg,
        riser_height_m=args.riser_height,
        runtime_filter=runtime_filter,
        clearance_barrier_slope=args.clearance_slope,
        recovery_distance_m=args.recovery_distance,
        filter_alpha=args.filter_alpha,
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if "deployable_failure" in env_cfg.observations:
        raise RuntimeError("v26 evaluator contains a forbidden actor observation")
    if "specialist_failure_signal" in env_cfg.rewards:
        raise RuntimeError("v26 evaluator contains a forbidden specialist reward")

    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    agent_cfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task)
    if runner_cls is None:
        raise RuntimeError("v26 task has no runner")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
    completed: list[dict[str, Any]] = []
    try:
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        if int(runner.alg.actor.obs_dim) != 405:
            raise RuntimeError("v26 evaluator actor is not the original 405-D policy")
        actor_hash = actor_state_sha256(runner.alg.actor.state_dict())
        policy = runner.get_inference_policy(args.device)
        obs, _ = env.reset()
        action_term = base_env.action_manager.get_term("joint_pos")
        if not isinstance(action_term, HigherRiserCbfAction):
            raise TypeError("v26 evaluator did not build its additive action term")
        command_term = base_env.command_manager.get_term("twist")
        initial_signature = _initial_state_signature(
            obs, base_env, action_term, command_term
        )
        num_risers = int(action_term._edge_x.shape[-1])
        active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        returns = torch.zeros(args.num_envs, device=args.device)
        steps = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        max_riser = torch.zeros_like(steps)
        kick_count = torch.zeros_like(steps)
        overlap_steps = torch.zeros_like(steps)
        intervention_count = torch.zeros_like(steps)
        would_intervene_count = torch.zeros_like(steps)
        correction_sum = torch.zeros(args.num_envs, device=args.device)
        counterfactual_correction_sum = torch.zeros_like(correction_sum)
        maximum_reprojection_error = torch.zeros_like(correction_sum)
        swing_selection_mismatch_count = torch.zeros_like(steps)
        maximum_steps = int(base_env.max_episode_length) + 2

        with torch.inference_mode():
            for _ in range(maximum_steps):
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)
                extras = dict(extras)
                active_float = active.float()
                active_long = active.long()
                returns += rewards * active_float
                steps += active_long
                current_riser = extras["online_stair_index"].long()
                max_riser = torch.where(
                    active, torch.maximum(max_riser, current_riser), max_riser
                )
                kick = extras["v26_toe_riser_kick"].bool()
                overlap = extras["v26_toe_riser_overlap"].bool()
                intervened = extras["cbf_intervened"].bool()
                would_intervene = extras["cbf_would_intervene"].bool()
                correction = extras["cbf_intervention_magnitude"]
                counterfactual_correction = extras["cbf_counterfactual_magnitude"]
                reprojection = extras["v26_teacher_reprojection_error"]
                selection_match = extras["v26_swing_selection_matches"].bool()
                kick_count += kick.long() * active_long
                overlap_steps += overlap.long() * active_long
                intervention_count += intervened.long() * active_long
                would_intervene_count += would_intervene.long() * active_long
                correction_sum += correction * active_float
                counterfactual_correction_sum += (
                    counterfactual_correction * active_float
                )
                maximum_reprojection_error = torch.where(
                    active,
                    torch.maximum(maximum_reprojection_error, reprojection),
                    maximum_reprojection_error,
                )
                swing_selection_mismatch_count += (
                    ~selection_match
                ).long() * active_long

                record_mask = dones.bool() & active
                if bool(record_mask.any()):
                    fell = extras["online_fell"].bool()
                    timed_out = extras.get(
                        "time_outs", torch.zeros_like(record_mask)
                    ).bool()
                    success = base_env.termination_manager.get_term(
                        "reached_top"
                    ).bool()
                    for env_id in (
                        record_mask.nonzero(as_tuple=False).flatten().tolist()
                    ):
                        reached = int(max_riser[env_id])
                        episode_steps = max(1, int(steps[env_id]))
                        succeeded = bool(success[env_id])
                        had_kick = int(kick_count[env_id]) > 0
                        failure_type = (
                            "success"
                            if succeeded
                            else "toe_riser_under_clearance"
                            if had_kick
                            else "balance_or_other_fall"
                            if bool(fell[env_id])
                            else "timeout_or_other_nonfall"
                        )
                        completed.append(
                            {
                                "evaluation_seed": args.seed,
                                "environment_id": env_id,
                                "success": succeeded,
                                "fell": bool(fell[env_id]),
                                "timed_out": bool(timed_out[env_id]),
                                "failure_type": failure_type,
                                "toe_riser_kick": had_kick,
                                "toe_riser_kick_count": int(kick_count[env_id]),
                                "toe_riser_overlap_fraction": (
                                    int(overlap_steps[env_id]) / episode_steps
                                ),
                                "return": float(returns[env_id]),
                                "steps": episode_steps,
                                "max_riser": reached,
                                "completion_fraction": reached / num_risers,
                                "intervention_count": int(intervention_count[env_id]),
                                "intervention_per_riser": (
                                    int(intervention_count[env_id]) / max(1, reached)
                                ),
                                "would_intervene_count": int(
                                    would_intervene_count[env_id]
                                ),
                                "would_intervene_per_riser": (
                                    int(would_intervene_count[env_id]) / max(1, reached)
                                ),
                                "mean_correction_norm": (
                                    float(correction_sum[env_id]) / episode_steps
                                ),
                                "mean_counterfactual_correction_norm": (
                                    float(counterfactual_correction_sum[env_id])
                                    / episode_steps
                                ),
                                "teacher_reprojection_max_abs_error": float(
                                    maximum_reprojection_error[env_id]
                                ),
                                "swing_selection_mismatch_count": int(
                                    swing_selection_mismatch_count[env_id]
                                ),
                            }
                        )
                    active &= ~record_mask
                    if not bool(active.any()):
                        break
        if bool(active.any()) or len(completed) != args.num_episodes:
            raise RuntimeError("v26 evaluator did not finish every initial episode")
        completed.sort(key=lambda row: int(row["environment_id"]))
        failure_count = sum(not row["success"] for row in completed)
        aligned_failure_count = sum(
            (not row["success"]) and row["toe_riser_kick"] for row in completed
        )
        total_reached_risers = sum(row["max_riser"] for row in completed)
        total_interventions = sum(row["intervention_count"] for row in completed)
        total_would_intervene = sum(row["would_intervene_count"] for row in completed)
        summary = {
            "schema_version": 1,
            "task": task,
            "environment_variant": ENVIRONMENT_VARIANT,
            "checkpoint_sha256": file_sha256(checkpoint),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "num_episodes": len(completed),
            "runtime_filter": runtime_filter,
            "riser_height_m": args.riser_height,
            "clearance_barrier_slope": args.clearance_slope,
            "recovery_distance_m": args.recovery_distance,
            "filter_alpha": args.filter_alpha,
            "deterministic_policy_mean": True,
            "one_initial_episode_per_env": True,
            "original_observation_interface": True,
            "actor_observation_dim": 405,
            "actor_state_sha256": actor_hash,
            "initial_state_signature": initial_signature,
            "success_count": sum(row["success"] for row in completed),
            "success_rate": sum(row["success"] for row in completed) / len(completed),
            "fall_count": sum(row["fell"] for row in completed),
            "fall_rate": sum(row["fell"] for row in completed) / len(completed),
            "timeout_count": sum(row["timed_out"] for row in completed),
            "timeout_rate": sum(row["timed_out"] for row in completed) / len(completed),
            "failure_count": failure_count,
            "toe_riser_failure_count": aligned_failure_count,
            "alignment_coverage": aligned_failure_count / max(1, failure_count),
            "kick_episode_count": sum(row["toe_riser_kick"] for row in completed),
            "kick_rate": sum(row["toe_riser_kick"] for row in completed)
            / len(completed),
            "mean_kick_count": sum(row["toe_riser_kick_count"] for row in completed)
            / len(completed),
            "mean_return": sum(row["return"] for row in completed) / len(completed),
            "mean_reached_riser": sum(row["max_riser"] for row in completed)
            / len(completed),
            "total_reached_risers": total_reached_risers,
            "total_intervention_count": total_interventions,
            "total_would_intervene_count": total_would_intervene,
            "intervention_per_riser": total_interventions
            / max(1, total_reached_risers),
            "would_intervene_per_riser": total_would_intervene
            / max(1, total_reached_risers),
            "mean_correction_norm": sum(
                row["mean_correction_norm"] for row in completed
            )
            / len(completed),
            "mean_counterfactual_correction_norm": sum(
                row["mean_counterfactual_correction_norm"] for row in completed
            )
            / len(completed),
            "teacher_reprojection_max_abs_error": max(
                row["teacher_reprojection_max_abs_error"] for row in completed
            ),
            "swing_selection_mismatch_count": sum(
                row["swing_selection_mismatch_count"] for row in completed
            ),
            "failure_type_counts": {
                failure_type: sum(
                    row["failure_type"] == failure_type for row in completed
                )
                for failure_type in (
                    "toe_riser_under_clearance",
                    "balance_or_other_fall",
                    "timeout_or_other_nonfall",
                )
            },
            "shift": shift_metadata,
        }
    finally:
        env.close()

    if summary["teacher_reprojection_max_abs_error"] > 1.0e-6:
        raise RuntimeError("v26 evaluator detected teacher reprojection corruption")
    if summary["swing_selection_mismatch_count"] != 0:
        raise RuntimeError("v26 evaluator detected divergent swing-foot selection")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_csv.with_name(f".{args.output_csv.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(completed[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(completed)
    temporary.replace(args.output_csv)
    _atomic_json(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

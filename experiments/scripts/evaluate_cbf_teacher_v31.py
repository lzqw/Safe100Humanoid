"""Evaluate one v31 policy/filter/context condition on fixed initial episodes."""

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
from cbf_teacher_v31_protocol import (
    BASE_CHECKPOINT_SHA256,
    CLEARANCE_BARRIER_SLOPE,
    CONTEXTS,
    FILTER_ALPHA,
    PROTOCOL_ID,
    RECOVERY_DISTANCE_M,
    TASK_ID,
    environment_parameters,
)
from proximal_v23_io import actor_state_sha256, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
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
        raise RuntimeError("v31 initial signature requires stair terrain")
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
        raise ValueError("v31 evaluation uses one initial episode per environment")
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("v31 evaluation input is missing")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout:
        raise RuntimeError("v31 evaluation requires a clean committed worktree")
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "frozen_before_v31_preflight_and_formal"
    ):
        raise RuntimeError("v31 evaluation protocol id differs")
    committed_protocol = subprocess.run(
        ["git", "show", f"HEAD:{protocol_path.relative_to(repo)}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if committed_protocol != protocol_path.read_bytes():
        raise RuntimeError("v31 evaluation protocol is not committed")
    context_spec = protocol.get("contexts", {}).get(args.context)
    expected_spec = environment_parameters(args.context)
    if context_spec != expected_spec:
        raise RuntimeError("v31 evaluation context differs from protocol")
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
    from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
    from src.tasks.stairs_cbf.teacher_v26 import HigherRiserCbfAction

    env_cfg = load_env_cfg(TASK_ID, play=True)
    shift = configure_v31_context(
        env_cfg,
        context=args.context,
        runtime_filter=runtime_filter,
        context_spec=context_spec,
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError("v31 task has no standard inference runner")
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
            raise RuntimeError("v31 evaluator actor is not the original 405-D policy")
        actor_state = runner.alg.actor.state_dict()
        actor_hash = actor_state_sha256(actor_state)
        deterministic_hash = actor_state_sha256(
            {
                key: value
                for key, value in actor_state.items()
                if not key.startswith("distribution.")
            }
        )
        policy = runner.get_inference_policy(args.device)
        obs, _ = env.reset()
        action_term = base_env.action_manager.get_term("joint_pos")
        if not isinstance(action_term, HigherRiserCbfAction):
            raise TypeError("v31 evaluator did not build the fixed CBF action")
        command_term = base_env.command_manager.get_term("twist")
        initial_signature = _initial_state_signature(
            obs, base_env, action_term, command_term
        )
        num_risers = int(action_term._edge_x.shape[-1])
        n = args.num_envs
        active = torch.ones(n, dtype=torch.bool, device=args.device)
        returns = torch.zeros(n, device=args.device)
        steps = torch.zeros(n, dtype=torch.long, device=args.device)
        max_riser = torch.zeros_like(steps)
        kick_count = torch.zeros_like(steps)
        overlap_steps = torch.zeros_like(steps)
        intervention_count = torch.zeros_like(steps)
        would_intervene_count = torch.zeros_like(steps)
        nominal_violation_steps = torch.zeros_like(steps)
        correction_sum = torch.zeros(n, device=args.device)
        counterfactual_correction_sum = torch.zeros_like(correction_sum)
        minimum_nominal_margin = torch.full((n,), float("inf"), device=args.device)
        maximum_reprojection_error = torch.zeros_like(correction_sum)
        swing_selection_mismatch_count = torch.zeros_like(steps)
        maximum_steps = int(base_env.max_episode_length) + 2
        step_seconds = float(base_env.step_dt)

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
                counterfactual = extras["cbf_counterfactual_magnitude"]
                reprojection = extras["v26_teacher_reprojection_error"]
                selection_match = extras["v26_swing_selection_matches"].bool()
                nominal_margin = action_term.psi_nominal
                kick_count += kick.long() * active_long
                overlap_steps += overlap.long() * active_long
                intervention_count += intervened.long() * active_long
                would_intervene_count += would_intervene.long() * active_long
                nominal_violation_steps += (
                    nominal_margin < -1.0e-5
                ).long() * active_long
                correction_sum += correction * active_float
                counterfactual_correction_sum += counterfactual * active_float
                minimum_nominal_margin = torch.where(
                    active,
                    torch.minimum(minimum_nominal_margin, nominal_margin),
                    minimum_nominal_margin,
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
                                "unsafe_overlap_steps": int(overlap_steps[env_id]),
                                "return": float(returns[env_id]),
                                "steps": episode_steps,
                                "completion_time_s": episode_steps * step_seconds,
                                "max_riser": reached,
                                "completion_fraction": reached / num_risers,
                                "intervention_count": int(intervention_count[env_id]),
                                "would_intervene_count": int(
                                    would_intervene_count[env_id]
                                ),
                                "nominal_barrier_violation_steps": int(
                                    nominal_violation_steps[env_id]
                                ),
                                "mean_correction_norm": float(correction_sum[env_id])
                                / episode_steps,
                                "mean_counterfactual_correction_norm": float(
                                    counterfactual_correction_sum[env_id]
                                )
                                / episode_steps,
                                "minimum_nominal_barrier_margin": float(
                                    minimum_nominal_margin[env_id]
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
            raise RuntimeError("v31 evaluator did not finish every initial episode")
        completed.sort(key=lambda row: int(row["environment_id"]))
        total_risers = sum(int(row["max_riser"]) for row in completed)
        total_steps = sum(int(row["steps"]) for row in completed)
        total_interventions = sum(int(row["intervention_count"]) for row in completed)
        total_would = sum(int(row["would_intervene_count"]) for row in completed)
        total_kicks = sum(int(row["toe_riser_kick_count"]) for row in completed)
        total_overlap = sum(int(row["unsafe_overlap_steps"]) for row in completed)
        total_violations = sum(
            int(row["nominal_barrier_violation_steps"]) for row in completed
        )
        failure_count = sum(not bool(row["success"]) for row in completed)
        aligned_failures = sum(
            (not bool(row["success"])) and bool(row["toe_riser_kick"])
            for row in completed
        )
        successful_times = [
            float(row["completion_time_s"]) for row in completed if bool(row["success"])
        ]
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "task": TASK_ID,
            "context": args.context,
            "context_spec": context_spec,
            "shift": shift,
            "checkpoint_sha256": file_sha256(checkpoint),
            "base_checkpoint": file_sha256(checkpoint) == BASE_CHECKPOINT_SHA256,
            "actor_state_sha256": actor_hash,
            "actor_deterministic_state_sha256": deterministic_hash,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "num_episodes": len(completed),
            "runtime_filter": runtime_filter,
            "deterministic_policy_mean": True,
            "one_initial_episode_per_env": True,
            "original_observation_interface": True,
            "actor_observation_dim": 405,
            "initial_state_signature": initial_signature,
            "success_count": sum(bool(row["success"]) for row in completed),
            "success_rate": sum(bool(row["success"]) for row in completed)
            / len(completed),
            "fall_count": sum(bool(row["fell"]) for row in completed),
            "fall_rate": sum(bool(row["fell"]) for row in completed) / len(completed),
            "timeout_count": sum(bool(row["timed_out"]) for row in completed),
            "kick_episode_count": sum(bool(row["toe_riser_kick"]) for row in completed),
            "kick_episode_rate": sum(bool(row["toe_riser_kick"]) for row in completed)
            / len(completed),
            "failure_count": failure_count,
            "toe_riser_alignment_coverage": aligned_failures / max(1, failure_count),
            "mean_return": float(np.mean([row["return"] for row in completed])),
            "mean_reached_riser": float(
                np.mean([row["max_riser"] for row in completed])
            ),
            "mean_completion_time_s": float(
                np.mean([row["completion_time_s"] for row in completed])
            ),
            "mean_success_completion_time_s": (
                float(np.mean(successful_times)) if successful_times else None
            ),
            "total_steps": total_steps,
            "total_reached_risers": total_risers,
            "intervention_steps_per_riser": total_interventions / max(1, total_risers),
            "counterfactual_would_intervene_fraction": total_would
            / max(1, total_steps),
            "mean_correction_norm": sum(
                float(row["mean_correction_norm"]) * int(row["steps"])
                for row in completed
            )
            / max(1, total_steps),
            "mean_counterfactual_correction_norm": sum(
                float(row["mean_counterfactual_correction_norm"]) * int(row["steps"])
                for row in completed
            )
            / max(1, total_steps),
            "toe_riser_kick_events_per_riser": total_kicks / max(1, total_risers),
            "unsafe_overlap_steps_per_riser": total_overlap / max(1, total_risers),
            "nominal_barrier_violation_steps_per_riser": total_violations
            / max(1, total_risers),
            "mean_episode_minimum_nominal_barrier_margin": float(
                np.mean([row["minimum_nominal_barrier_margin"] for row in completed])
            ),
            "global_minimum_nominal_barrier_margin": min(
                float(row["minimum_nominal_barrier_margin"]) for row in completed
            ),
            "teacher_reprojection_max_abs_error": max(
                float(row["teacher_reprojection_max_abs_error"]) for row in completed
            ),
            "swing_selection_mismatch_count": sum(
                int(row["swing_selection_mismatch_count"]) for row in completed
            ),
        }
    finally:
        env.close()
    if summary["teacher_reprojection_max_abs_error"] > 1.0e-6:
        raise RuntimeError("v31 evaluator detected teacher transform corruption")
    if summary["swing_selection_mismatch_count"] != 0:
        raise RuntimeError("v31 evaluator detected swing-foot routing corruption")
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

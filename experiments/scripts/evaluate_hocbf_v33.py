"""Evaluate one frozen policy/context/filter condition for v33."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cbf_teacher_v31_protocol import (
    CLEARANCE_BARRIER_SLOPE,
    CONTEXTS,
    FILTER_ALPHA,
    RECOVERY_DISTANCE_M,
    TASK_ID,
    environment_parameters,
)
from hocbf_v33_protocol import CURRENT_CBF_MODE, HOCBF_MODE, PROTOCOL_ID
from proximal_v23_io import actor_state_sha256, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
    parser.add_argument(
        "--cbf-mode", choices=(CURRENT_CBF_MODE, HOCBF_MODE), required=True
    )
    parser.add_argument("--runtime-filter", choices=("on", "off"), required=True)
    parser.add_argument("--omega", type=float)
    parser.add_argument("--lambda-x", type=float)
    parser.add_argument("--lambda-s", type=float)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-label", default="unspecified")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _initial_state_signature(obs, base_env, action_term, command_term) -> str:
    signature = hashlib.sha256()
    terrain = base_env.scene.terrain
    if terrain is None:
        raise RuntimeError("v33 initial signature requires stair terrain")
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
            command_term, "raw_command", base_env.command_manager.get_command("twist")
        ),
        getattr(
            command_term,
            "delay_steps",
            torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device),
        ),
    )
    for tensor in tensors:
        signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return signature.hexdigest()


def _value(tensor: torch.Tensor, index: int, cast=float):
    return cast(tensor[index])


def main() -> None:
    args = _parse_args()
    if args.num_envs != args.num_episodes or args.num_envs < 1:
        raise ValueError("v33 uses one initial episode per environment")
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError("v33 evaluation input is missing")
    config = json.loads(config_path.read_text())
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("v33 config protocol id differs")
    if args.cbf_mode == HOCBF_MODE and None in (
        args.omega,
        args.lambda_x,
        args.lambda_s,
    ):
        raise ValueError("v33 HOCBF parameters are required")
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
    from src.tasks.stairs_cbf.hocbf_action import (
        InstrumentedCurrentCbfAction,
        TaskConsistentHocbfAction,
        configure_v33_cbf,
    )

    runtime_filter = args.runtime_filter == "on"
    context_spec = environment_parameters(args.context)
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
    cbf = configure_v33_cbf(
        env_cfg,
        mode=args.cbf_mode,
        runtime_filter=runtime_filter,
        omega=args.omega,
        forward_task_weight=args.lambda_x,
        correction_smoothness=args.lambda_s,
        measure_compute_time=True,
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError("v33 task has no inference runner")
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
            raise RuntimeError("v33 actor observation is not 405-D")
        actor_state = runner.alg.actor.state_dict()
        actor_hash = actor_state_sha256(actor_state)
        deterministic_hash = actor_state_sha256(
            {k: v for k, v in actor_state.items() if not k.startswith("distribution.")}
        )
        policy = runner.get_inference_policy(args.device)
        obs, _ = env.reset()
        term = base_env.action_manager.get_term("joint_pos")
        if not isinstance(
            term, (InstrumentedCurrentCbfAction, TaskConsistentHocbfAction)
        ):
            raise TypeError("v33 evaluator did not build a v33 action")
        command_term = base_env.command_manager.get_term("twist")
        initial_signature = _initial_state_signature(obs, base_env, term, command_term)
        n = args.num_envs
        device = base_env.device
        active = torch.ones(n, dtype=torch.bool, device=device)
        returns = torch.zeros(n, device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)
        max_riser = torch.zeros_like(steps)
        kick_count = torch.zeros_like(steps)
        overlap_steps = torch.zeros_like(steps)
        intervention_steps = torch.zeros_like(steps)
        intervention_events = torch.zeros_like(steps)
        would_steps = torch.zeros_like(steps)
        previous_intervened = torch.zeros(n, dtype=torch.bool, device=device)
        ever_intervened = torch.zeros_like(previous_intervened)
        max_duration = torch.zeros_like(steps)
        post_steps = torch.zeros_like(steps)
        correction_sum = torch.zeros(n, device=device)
        jerk_sum = torch.zeros_like(correction_sum)
        forward_deviation_sum = torch.zeros_like(correction_sum)
        vertical_change_sum = torch.zeros_like(correction_sum)
        toe_impulse_sum = torch.zeros_like(correction_sum)
        toe_force_peak = torch.zeros_like(correction_sum)
        roll_sum = torch.zeros_like(correction_sum)
        pitch_sum = torch.zeros_like(correction_sum)
        angular_sum = torch.zeros_like(correction_sum)
        support_slip_sum = torch.zeros_like(correction_sum)
        minimum_nominal_margin = torch.full((n,), torch.inf, device=device)
        minimum_filtered_margin = torch.full((n,), torch.inf, device=device)
        maximum_safe_identity_error = torch.zeros_like(correction_sum)
        all_finite = torch.ones(n, dtype=torch.bool, device=device)
        maximum_steps = int(base_env.max_episode_length) + 2
        step_seconds = float(base_env.step_dt)
        num_risers = int(term._edge_x.shape[-1])

        with torch.inference_mode():
            for _ in range(maximum_steps):
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)
                extras = dict(extras)
                af = active.float()
                al = active.long()
                returns += rewards * af
                steps += al
                current_riser = extras["online_stair_index"].long()
                max_riser = torch.where(
                    active, torch.maximum(max_riser, current_riser), max_riser
                )
                kick = extras["v26_toe_riser_kick"].bool()
                overlap = extras["v26_toe_riser_overlap"].bool()
                intervened = extras["cbf_intervened"].bool()
                would = extras["cbf_would_intervene"].bool()
                onset = intervened & ~previous_intervened
                ever_intervened |= intervened & active
                post = ever_intervened & active
                geometric = term.geometric_active & active
                kick_count += kick.long() * al
                overlap_steps += overlap.long() * al
                intervention_steps += intervened.long() * al
                intervention_events += onset.long() * al
                would_steps += would.long() * al
                max_duration = torch.where(
                    active,
                    torch.maximum(
                        max_duration, extras["v33_intervention_duration_steps"].long()
                    ),
                    max_duration,
                )
                post_steps += post.long()
                correction_sum += extras["v33_qddot_correction_norm"] * af
                jerk_sum += extras["v33_qddot_correction_jerk"] * af
                forward_deviation_sum += (
                    extras["v33_foot_forward_acceleration_deviation"] * af
                )
                vertical_change_sum += (
                    extras["v33_foot_vertical_acceleration_change"] * af
                )
                impulse = extras["v33_toe_riser_contact_impulse_step"]
                force = extras["v33_toe_riser_contact_force"]
                toe_impulse_sum += impulse * af
                toe_force_peak = torch.where(
                    active, torch.maximum(toe_force_peak, force), toe_force_peak
                )
                roll_sum += torch.abs(extras["v33_root_roll"]) * post.float()
                pitch_sum += torch.abs(extras["v33_root_pitch"]) * post.float()
                angular_sum += extras["v33_base_angular_velocity"] * post.float()
                support_slip_sum += extras["v33_support_foot_slip"] * post.float()
                minimum_nominal_margin = torch.where(
                    geometric,
                    torch.minimum(
                        minimum_nominal_margin, extras["v33_nominal_hocbf_margin"]
                    ),
                    minimum_nominal_margin,
                )
                minimum_filtered_margin = torch.where(
                    geometric,
                    torch.minimum(
                        minimum_filtered_margin, extras["v33_filtered_hocbf_margin"]
                    ),
                    minimum_filtered_margin,
                )
                maximum_safe_identity_error = torch.where(
                    active,
                    torch.maximum(
                        maximum_safe_identity_error,
                        extras["v33_nominal_safe_target_error"],
                    ),
                    maximum_safe_identity_error,
                )
                finite_values = torch.stack(
                    (
                        extras["v33_h_dot"],
                        extras["v33_estimated_drift"],
                        extras["v33_nominal_hocbf_margin"],
                        extras["v33_filtered_hocbf_margin"],
                        extras["v33_qddot_correction_norm"],
                        extras["v33_qddot_correction_jerk"],
                    ),
                    dim=-1,
                )
                all_finite &= torch.isfinite(finite_values).all(dim=-1) | ~active
                previous_intervened = intervened & active

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
                        episode_steps = max(1, _value(steps, env_id, int))
                        post_count = max(1, _value(post_steps, env_id, int))
                        reached = _value(max_riser, env_id, int)
                        completed.append(
                            {
                                "evaluation_seed": args.seed,
                                "environment_id": env_id,
                                "success": _value(success, env_id, bool),
                                "fell": _value(fell, env_id, bool),
                                "timed_out": _value(timed_out, env_id, bool),
                                "post_intervention_fall": bool(
                                    fell[env_id] & ever_intervened[env_id]
                                ),
                                "ever_intervened": _value(
                                    ever_intervened, env_id, bool
                                ),
                                "toe_riser_kick": _value(kick_count, env_id, int) > 0,
                                "toe_riser_kick_count": _value(kick_count, env_id, int),
                                "unsafe_overlap_steps": _value(
                                    overlap_steps, env_id, int
                                ),
                                "return": _value(returns, env_id),
                                "steps": episode_steps,
                                "completion_time_s": episode_steps * step_seconds,
                                "max_riser": reached,
                                "completion_fraction": reached / num_risers,
                                "intervention_count": _value(
                                    intervention_steps, env_id, int
                                ),
                                "intervention_event_count": _value(
                                    intervention_events, env_id, int
                                ),
                                "would_intervene_count": _value(
                                    would_steps, env_id, int
                                ),
                                "maximum_intervention_duration_steps": _value(
                                    max_duration, env_id, int
                                ),
                                "mean_qddot_correction_norm": _value(
                                    correction_sum, env_id
                                )
                                / episode_steps,
                                "mean_qddot_correction_jerk": _value(jerk_sum, env_id)
                                / episode_steps,
                                "mean_foot_forward_acceleration_deviation": _value(
                                    forward_deviation_sum, env_id
                                )
                                / episode_steps,
                                "mean_foot_vertical_acceleration_change": _value(
                                    vertical_change_sum, env_id
                                )
                                / episode_steps,
                                "toe_riser_contact_force_peak": _value(
                                    toe_force_peak, env_id
                                ),
                                "toe_riser_contact_impulse": _value(
                                    toe_impulse_sum, env_id
                                ),
                                "post_intervention_mean_abs_root_roll": _value(
                                    roll_sum, env_id
                                )
                                / post_count,
                                "post_intervention_mean_abs_root_pitch": _value(
                                    pitch_sum, env_id
                                )
                                / post_count,
                                "post_intervention_mean_base_angular_velocity": _value(
                                    angular_sum, env_id
                                )
                                / post_count,
                                "post_intervention_mean_support_foot_slip": _value(
                                    support_slip_sum, env_id
                                )
                                / post_count,
                                "minimum_nominal_hocbf_margin": _value(
                                    torch.nan_to_num(
                                        minimum_nominal_margin, posinf=0.0
                                    ),
                                    env_id,
                                ),
                                "minimum_filtered_hocbf_margin": _value(
                                    torch.nan_to_num(
                                        minimum_filtered_margin, posinf=0.0
                                    ),
                                    env_id,
                                ),
                                "maximum_nominal_safe_target_error": _value(
                                    maximum_safe_identity_error, env_id
                                ),
                                "all_hocbf_telemetry_finite": _value(
                                    all_finite, env_id, bool
                                ),
                            }
                        )
                    active &= ~record_mask
                    if not bool(active.any()):
                        break
        if bool(active.any()) or len(completed) != args.num_episodes:
            raise RuntimeError("v33 evaluator did not finish all first episodes")
        completed.sort(key=lambda row: int(row["environment_id"]))
        total_steps = sum(int(row["steps"]) for row in completed)
        total_risers = sum(int(row["max_riser"]) for row in completed)
        successful_times = [
            float(row["completion_time_s"]) for row in completed if bool(row["success"])
        ]
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "task": TASK_ID,
            "context": args.context,
            "policy_label": args.policy_label,
            "checkpoint_sha256": file_sha256(checkpoint),
            "actor_state_sha256": actor_hash,
            "actor_deterministic_state_sha256": deterministic_hash,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "num_episodes": len(completed),
            "initial_state_signature": initial_signature,
            "deterministic_policy_mean": True,
            "runtime_filter": runtime_filter,
            "cbf": cbf,
            "shift": shift,
            "success_count": sum(bool(row["success"]) for row in completed),
            "success_rate": sum(bool(row["success"]) for row in completed)
            / len(completed),
            "fall_rate": sum(bool(row["fell"]) for row in completed) / len(completed),
            "post_intervention_fall_rate": sum(
                bool(row["post_intervention_fall"]) for row in completed
            )
            / max(1, sum(bool(row["ever_intervened"]) for row in completed)),
            "rescue_denominator": None,
            "mean_reached_riser": float(
                np.mean([row["max_riser"] for row in completed])
            ),
            "mean_completion_time_s": float(
                np.mean([row["completion_time_s"] for row in completed])
            ),
            "mean_success_completion_time_s": float(np.mean(successful_times))
            if successful_times
            else None,
            "intervention_steps_per_riser": sum(
                int(row["intervention_count"]) for row in completed
            )
            / max(1, total_risers),
            "intervention_events_per_riser": sum(
                int(row["intervention_event_count"]) for row in completed
            )
            / max(1, total_risers),
            "unsafe_overlap_steps_per_riser": sum(
                int(row["unsafe_overlap_steps"]) for row in completed
            )
            / max(1, total_risers),
            "mean_qddot_correction_norm": sum(
                float(row["mean_qddot_correction_norm"]) * int(row["steps"])
                for row in completed
            )
            / max(1, total_steps),
            "mean_qddot_correction_jerk": sum(
                float(row["mean_qddot_correction_jerk"]) * int(row["steps"])
                for row in completed
            )
            / max(1, total_steps),
            "mean_foot_forward_acceleration_deviation": sum(
                float(row["mean_foot_forward_acceleration_deviation"])
                * int(row["steps"])
                for row in completed
            )
            / max(1, total_steps),
            "mean_foot_vertical_acceleration_change": sum(
                float(row["mean_foot_vertical_acceleration_change"]) * int(row["steps"])
                for row in completed
            )
            / max(1, total_steps),
            "mean_toe_riser_contact_impulse": float(
                np.mean([row["toe_riser_contact_impulse"] for row in completed])
            ),
            "toe_riser_contact_force_peak": max(
                float(row["toe_riser_contact_force_peak"]) for row in completed
            ),
            "post_intervention_mean_abs_root_roll": float(
                np.mean(
                    [
                        row["post_intervention_mean_abs_root_roll"]
                        for row in completed
                        if row["ever_intervened"]
                    ]
                )
            )
            if any(row["ever_intervened"] for row in completed)
            else 0.0,
            "post_intervention_mean_abs_root_pitch": float(
                np.mean(
                    [
                        row["post_intervention_mean_abs_root_pitch"]
                        for row in completed
                        if row["ever_intervened"]
                    ]
                )
            )
            if any(row["ever_intervened"] for row in completed)
            else 0.0,
            "post_intervention_mean_support_foot_slip": float(
                np.mean(
                    [
                        row["post_intervention_mean_support_foot_slip"]
                        for row in completed
                        if row["ever_intervened"]
                    ]
                )
            )
            if any(row["ever_intervened"] for row in completed)
            else 0.0,
            "maximum_nominal_safe_target_error": max(
                float(row["maximum_nominal_safe_target_error"]) for row in completed
            ),
            "minimum_filtered_hocbf_margin": min(
                float(row["minimum_filtered_hocbf_margin"]) for row in completed
            ),
            "all_hocbf_telemetry_finite": all(
                bool(row["all_hocbf_telemetry_finite"]) for row in completed
            ),
            "mean_cbf_compute_time_ms": term.mean_compute_time_ms(),
            "total_steps": total_steps,
            "total_reached_risers": total_risers,
        }
    finally:
        env.close()
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

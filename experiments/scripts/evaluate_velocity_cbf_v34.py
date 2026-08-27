"""Evaluate one v34 policy/context/velocity-CBF condition."""

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
from proximal_v23_io import actor_state_sha256, file_sha256
from velocity_cbf_v34_protocol import (
    CURRENT_CBF_MODE,
    OPTIMIZED_CBF_MODE,
    PROTOCOL_ID,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
    parser.add_argument(
        "--cbf-mode", choices=(CURRENT_CBF_MODE, OPTIMIZED_CBF_MODE), required=True
    )
    parser.add_argument("--parameters-json")
    parser.add_argument("--runtime-filter", choices=("on", "off"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-label", default="unspecified")
    parser.add_argument("--candidate", default="unspecified")
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
        raise RuntimeError("v34 signature requires terrain")
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
    )
    for tensor in tensors:
        signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return signature.hexdigest()


def _value(tensor: torch.Tensor, index: int, cast=float):
    return cast(tensor[index])


def main() -> None:
    args = _parse_args()
    if args.num_envs != args.num_episodes or args.num_envs < 1:
        raise ValueError("v34 uses one initial episode per environment")
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    config_path = args.search_config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("v34 search config differs")
    parameters = None
    if args.cbf_mode == OPTIMIZED_CBF_MODE:
        if args.parameters_json is None:
            raise ValueError("optimized v34 evaluation requires parameters")
        parameters = json.loads(args.parameters_json)
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
    from src.tasks.stairs_cbf.velocity_cbf_action import (
        InstrumentedCurrentVelocityCbfAction,
        TaskMetricVelocityCbfAction,
        configure_v34_cbf,
    )

    runtime_filter = args.runtime_filter == "on"
    env_cfg = load_env_cfg(TASK_ID, play=True)
    shift = configure_v31_context(
        env_cfg,
        context=args.context,
        runtime_filter=runtime_filter,
        context_spec=environment_parameters(args.context),
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    cbf = configure_v34_cbf(
        env_cfg,
        mode=args.cbf_mode,
        runtime_filter=runtime_filter,
        parameters=parameters,
        measure_compute_time=True,
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError("v34 task has no inference runner")
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
            raise RuntimeError("v34 actor is not the original 405-D policy")
        actor_state = runner.alg.actor.state_dict()
        actor_hash = actor_state_sha256(actor_state)
        deterministic_hash = actor_state_sha256(
            {k: v for k, v in actor_state.items() if not k.startswith("distribution.")}
        )
        policy = runner.get_inference_policy(args.device)
        # Runner construction and checkpoint loading may consume global torch
        # RNG state.  RslRlVecEnvWrapper.reset() does not accept ``seed``, so
        # reseed the underlying environment immediately before its delegated
        # reset.  Identical --seed values then select identical episodes across
        # actors and source commits.
        base_env.seed(args.seed)
        obs, _ = env.reset()
        term = base_env.action_manager.get_term("joint_pos")
        if not isinstance(
            term,
            (InstrumentedCurrentVelocityCbfAction, TaskMetricVelocityCbfAction),
        ):
            raise TypeError("v34 evaluator built the wrong action")
        initial_signature = _initial_state_signature(
            obs, base_env, term, base_env.command_manager.get_term("twist")
        )
        n, device = args.num_envs, base_env.device
        active = torch.ones(n, dtype=torch.bool, device=device)
        returns = torch.zeros(n, device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)
        max_riser = torch.zeros_like(steps)
        overlap_steps = torch.zeros_like(steps)
        kick_count = torch.zeros_like(steps)
        intervention_steps = torch.zeros_like(steps)
        intervention_events = torch.zeros_like(steps)
        previous_intervened = torch.zeros(n, dtype=torch.bool, device=device)
        ever_intervened = torch.zeros_like(previous_intervened)
        correction_sum = torch.zeros(n, device=device)
        jerk_sum = torch.zeros_like(correction_sum)
        impulse_sum = torch.zeros_like(correction_sum)
        force_peak = torch.zeros_like(correction_sum)
        roll_sum = torch.zeros_like(correction_sum)
        pitch_sum = torch.zeros_like(correction_sum)
        slip_sum = torch.zeros_like(correction_sum)
        post_steps = torch.zeros_like(steps)
        min_nominal = torch.full((n,), torch.inf, device=device)
        min_filtered = torch.full((n,), torch.inf, device=device)
        safe_identity_error = torch.zeros_like(correction_sum)
        all_finite = torch.ones(n, dtype=torch.bool, device=device)
        num_risers = int(term._edge_x.shape[-1])
        step_seconds = float(base_env.step_dt)
        maximum_steps = int(base_env.max_episode_length) + 2

        with torch.inference_mode():
            for _ in range(maximum_steps):
                obs, rewards, dones, extras = env.step(policy(obs))
                extras = dict(extras)
                af, al = active.float(), active.long()
                returns += rewards * af
                steps += al
                current_riser = extras["online_stair_index"].long()
                max_riser = torch.where(
                    active, torch.maximum(max_riser, current_riser), max_riser
                )
                kick = extras["v26_toe_riser_kick"].bool()
                overlap = extras["v26_toe_riser_overlap"].bool()
                intervened = extras["cbf_intervened"].bool()
                onset = intervened & ~previous_intervened
                ever_intervened |= intervened & active
                post = ever_intervened & active
                geometric = term.geometric_active & active
                kick_count += kick.long() * al
                overlap_steps += overlap.long() * al
                intervention_steps += intervened.long() * al
                intervention_events += onset.long() * al
                correction_sum += extras["v34_velocity_correction_norm"] * af
                jerk_sum += extras["v34_velocity_correction_jerk"] * af
                impulse = extras["v34_toe_riser_contact_impulse_step"]
                force = extras["v34_toe_riser_contact_force"]
                impulse_sum += impulse * af
                force_peak = torch.where(
                    active, torch.maximum(force_peak, force), force_peak
                )
                post_steps += post.long()
                roll_sum += torch.abs(extras["v34_root_roll"]) * post.float()
                pitch_sum += torch.abs(extras["v34_root_pitch"]) * post.float()
                slip_sum += extras["v34_support_foot_slip"] * post.float()
                min_nominal = torch.where(
                    geometric,
                    torch.minimum(min_nominal, extras["v34_nominal_margin"]),
                    min_nominal,
                )
                min_filtered = torch.where(
                    geometric,
                    torch.minimum(min_filtered, extras["v34_filtered_margin"]),
                    min_filtered,
                )
                safe_identity_error = torch.where(
                    active,
                    torch.maximum(
                        safe_identity_error,
                        extras["v34_nominal_safe_target_error"],
                    ),
                    safe_identity_error,
                )
                finite = torch.stack(
                    (
                        extras["v34_nominal_margin"],
                        extras["v34_filtered_margin"],
                        extras["v34_velocity_correction_norm"],
                        extras["v34_velocity_correction_jerk"],
                    ),
                    dim=-1,
                )
                all_finite &= torch.isfinite(finite).all(dim=-1) | ~active
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
                                "ever_intervened": _value(
                                    ever_intervened, env_id, bool
                                ),
                                "post_intervention_fall": bool(
                                    fell[env_id] & ever_intervened[env_id]
                                ),
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
                                "mean_velocity_correction_norm": _value(
                                    correction_sum, env_id
                                )
                                / episode_steps,
                                "mean_velocity_correction_jerk": _value(
                                    jerk_sum, env_id
                                )
                                / episode_steps,
                                "toe_riser_contact_impulse": _value(
                                    impulse_sum, env_id
                                ),
                                "toe_riser_contact_force_peak": _value(
                                    force_peak, env_id
                                ),
                                "post_intervention_mean_abs_root_roll": _value(
                                    roll_sum, env_id
                                )
                                / post_count,
                                "post_intervention_mean_abs_root_pitch": _value(
                                    pitch_sum, env_id
                                )
                                / post_count,
                                "post_intervention_mean_support_foot_slip": _value(
                                    slip_sum, env_id
                                )
                                / post_count,
                                "minimum_nominal_margin": _value(
                                    torch.nan_to_num(min_nominal, posinf=0.0), env_id
                                ),
                                "minimum_filtered_margin": _value(
                                    torch.nan_to_num(min_filtered, posinf=0.0), env_id
                                ),
                                "maximum_nominal_safe_target_error": _value(
                                    safe_identity_error, env_id
                                ),
                                "all_finite": _value(all_finite, env_id, bool),
                            }
                        )
                    active &= ~record_mask
                    if not bool(active.any()):
                        break
        if bool(active.any()) or len(completed) != args.num_episodes:
            raise RuntimeError("v34 evaluator did not finish all first episodes")
        completed.sort(key=lambda row: int(row["environment_id"]))
        total_steps = sum(int(row["steps"]) for row in completed)
        total_risers = sum(int(row["max_riser"]) for row in completed)
        ever = [row for row in completed if bool(row["ever_intervened"])]
        successful_times = [
            float(row["completion_time_s"]) for row in completed if bool(row["success"])
        ]

        def step_mean(field: str) -> float:
            return sum(
                float(row[field]) * int(row["steps"]) for row in completed
            ) / max(1, total_steps)

        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "task": TASK_ID,
            "context": args.context,
            "candidate": args.candidate,
            "policy_label": args.policy_label,
            "checkpoint_sha256": file_sha256(checkpoint),
            "actor_state_sha256": actor_hash,
            "actor_deterministic_state_sha256": deterministic_hash,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "num_episodes": len(completed),
            "initial_state_signature": initial_signature,
            "explicit_episode_reset_seed": args.seed,
            "episode_reset_after_runner_load": True,
            "runtime_filter": runtime_filter,
            "deterministic_policy_mean": True,
            "cbf": cbf,
            "shift": shift,
            "success_count": sum(bool(row["success"]) for row in completed),
            "success_rate": sum(bool(row["success"]) for row in completed)
            / len(completed),
            "fall_rate": sum(bool(row["fell"]) for row in completed) / len(completed),
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
            "post_intervention_fall_rate": sum(
                bool(row["post_intervention_fall"]) for row in completed
            )
            / max(1, len(ever)),
            "intervention_steps_per_riser": sum(
                int(row["intervention_count"]) for row in completed
            )
            / max(1, total_risers),
            "intervention_events_per_riser": sum(
                int(row["intervention_event_count"]) for row in completed
            )
            / max(1, total_risers),
            "mean_velocity_correction_norm": step_mean("mean_velocity_correction_norm"),
            "mean_velocity_correction_jerk": step_mean("mean_velocity_correction_jerk"),
            "mean_toe_riser_contact_impulse": float(
                np.mean([row["toe_riser_contact_impulse"] for row in completed])
            ),
            "toe_riser_contact_force_peak": max(
                float(row["toe_riser_contact_force_peak"]) for row in completed
            ),
            "unsafe_overlap_steps_per_riser": sum(
                int(row["unsafe_overlap_steps"]) for row in completed
            )
            / max(1, total_risers),
            "post_intervention_mean_abs_root_roll": (
                float(
                    np.mean(
                        [row["post_intervention_mean_abs_root_roll"] for row in ever]
                    )
                )
                if ever
                else 0.0
            ),
            "post_intervention_mean_abs_root_pitch": (
                float(
                    np.mean(
                        [row["post_intervention_mean_abs_root_pitch"] for row in ever]
                    )
                )
                if ever
                else 0.0
            ),
            "post_intervention_mean_support_foot_slip": (
                float(
                    np.mean(
                        [
                            row["post_intervention_mean_support_foot_slip"]
                            for row in ever
                        ]
                    )
                )
                if ever
                else 0.0
            ),
            "minimum_filtered_margin": min(
                float(row["minimum_filtered_margin"]) for row in completed
            ),
            "maximum_nominal_safe_target_error": max(
                float(row["maximum_nominal_safe_target_error"]) for row in completed
            ),
            "all_finite": all(bool(row["all_finite"]) for row in completed),
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

"""Run one frozen v32 continuation, mixed, or functional-preflight job."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cbf_teacher_v31_protocol import (
    CLEARANCE_BARRIER_SLOPE,
    FILTER_ALPHA,
    RECOVERY_DISTANCE_M,
    TASK_ID,
)
from cbf_teacher_v32_protocol import (
    ACTOR_EPOCHS,
    BASE_CHECKPOINT_SHA256,
    BEHAVIOR_LOG_PROB_ATOL,
    CONTINUATION_FINAL_ROUND,
    CONTINUATION_SCHEDULES,
    CONTINUATION_START_ROUND,
    CRITIC_EPOCHS,
    FORMAL_CONTEXTS,
    MIXED_CONTEXTS,
    MIXED_FINAL_ROUND,
    MIXED_SCHEDULE,
    MIXED_SEED,
    NUM_ENVS,
    POLICY_METHOD,
    PREFLIGHT_CASES,
    PREFLIGHT_CONTINUATION_ENVS,
    PREFLIGHT_STEPS,
    PROTOCOL_ID,
    ROLLOUT_STEPS,
    SOURCE_FILES,
    V31_A2_ROUND8_SHA256,
    a2_parameters,
    common_algorithm_parameters,
    continuation_seed,
    environment_parameters,
    learning_rates,
    mixed_context_env_counts,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_cbf_teacher_v31 import _configure_algorithm, _structural_audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v31-formal-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "formal"), required=True)
    parser.add_argument("--kind", choices=("continuation", "mixed"), required=True)
    parser.add_argument("--context", choices=(*FORMAL_CONTEXTS, "mixed"), required=True)
    parser.add_argument("--schedule", choices=CONTINUATION_SCHEDULES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_combination(args: argparse.Namespace) -> None:
    if args.kind == "continuation":
        if args.context not in FORMAL_CONTEXTS:
            raise ValueError("v32 continuation requires F1, F2, or F3")
    elif args.context != "mixed" or args.schedule != MIXED_SCHEDULE:
        raise ValueError("v32 mixed run requires context=mixed and LongDecay")
    if args.resume and args.phase == "preflight":
        raise ValueError("v32 functional preflight cannot be resumed")


def _preflight_seed(kind: str, context: str, schedule: str) -> int:
    for case in PREFLIGHT_CASES:
        if (
            case["kind"] == kind
            and case["context"] == context
            and case["schedule"] == schedule
        ):
            return int(case["seed"])
    raise ValueError(f"no frozen v32 preflight case for {kind}/{context}/{schedule}")


def _validate_protocol(
    repo: Path,
    protocol_path: Path,
    input_checkpoint: Path,
    *,
    phase: str,
    kind: str,
    context: str,
    schedule: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text())
    relative = protocol_path.relative_to(repo)
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    source = protocol.get("source_boundary", {})
    source_hashes = source.get("source_files", {})
    input_hash = file_sha256(input_checkpoint)
    expected_input = (
        V31_A2_ROUND8_SHA256[context]
        if kind == "continuation"
        else BASE_CHECKPOINT_SHA256
    )
    checks = {
        "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
        "status": protocol.get("status") == "frozen_before_v32_preflight_and_formal",
        "protocol_committed": committed == protocol_path.read_bytes(),
        "input_checkpoint": input_hash == expected_input,
        "algorithm": protocol.get("algorithm") == common_algorithm_parameters(),
        "A2_unchanged": protocol.get("A2_configuration_unchanged") == a2_parameters(),
        "source_set": set(source_hashes) == set(SOURCE_FILES),
        "source_hashes": all(
            (repo / name).is_file()
            and file_sha256(repo / name) == source_hashes.get(name)
            for name in SOURCE_FILES
        ),
        "source_commit_ancestor": subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                str(source.get("git_commit", "")),
                "HEAD",
            ],
            cwd=repo,
            check=False,
        ).returncode
        == 0,
    }
    if kind == "continuation":
        checks["context"] = protocol.get("contexts", {}).get(
            context
        ) == environment_parameters(context)
    else:
        checks["mixed_contexts"] = all(
            protocol.get("contexts", {}).get(item) == environment_parameters(item)
            for item in FORMAL_CONTEXTS
        )
    if phase == "formal":
        expected_seed = (
            continuation_seed(context, schedule)
            if kind == "continuation"
            else MIXED_SEED
        )
        matches = [
            run
            for run in protocol.get("formal_runs", [])
            if run["kind"] == kind
            and run["context"] == context
            and run["schedule"] == schedule
        ]
        checks["formal_run"] = len(matches) == 1 and matches[0]["seed"] == expected_seed
        checks["formal_input"] = (
            len(matches) == 1 and matches[0]["source_checkpoint_sha256"] == input_hash
        )
    else:
        seed = _preflight_seed(kind, context, schedule)
        checks["preflight_case"] = any(
            case["kind"] == kind
            and case["context"] == context
            and case["schedule"] == schedule
            and int(case["seed"]) == seed
            for case in protocol.get("preflight", {}).get("cases", [])
        )
    if not all(checks.values()):
        raise RuntimeError(f"v32 protocol validation failed: {checks}")
    return protocol, {
        "path": str(protocol_path),
        "sha256": file_sha256(protocol_path),
        "checks": checks,
    }


def _save_checkpoint(
    runner, path: Path, absolute_round: int, metadata: dict[str, Any]
) -> None:
    payload = runner.alg.save()
    payload["iter"] = absolute_round
    payload["infos"] = {"cbf_teacher_v32": metadata}
    payload["python_random_state"] = random.getstate()
    payload["numpy_random_state"] = np.random.get_state()
    payload["torch_random_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda_random_state_all"])


def _set_learning_rates(runner, actor_lr: float, critic_lr: float) -> None:
    runner.alg.actor_learning_rate = actor_lr
    runner.alg.critic_learning_rate = critic_lr
    runner.alg.learning_rate = actor_lr
    for group in runner.alg.actor_optimizer.param_groups:
        group["lr"] = actor_lr
    for group in runner.alg.critic_optimizer.param_groups:
        group["lr"] = critic_lr
    actual_actor = {
        float(group["lr"]) for group in runner.alg.actor_optimizer.param_groups
    }
    actual_critic = {
        float(group["lr"]) for group in runner.alg.critic_optimizer.param_groups
    }
    if actual_actor != {actor_lr} or actual_critic != {critic_lr}:
        raise RuntimeError("v32 optimizer learning-rate assignment failed")


def _success_term(env) -> torch.Tensor:
    if hasattr(env, "get_termination_term"):
        return env.get_termination_term("reached_top").bool()
    return env.unwrapped.termination_manager.get_term("reached_top").bool()


def _collect_round(runner, *, context_name: str) -> dict[str, Any]:
    from rsl_rl.utils import check_nan

    runner.alg.clear_cbf_rollout()
    runner.alg.train_mode()
    obs, _ = runner.env.reset()
    obs = obs.to(runner.device)
    n = runner.env.num_envs
    episode_returns = torch.zeros(n, device=runner.env.device)
    episode_max_riser = torch.zeros(n, dtype=torch.long, device=runner.env.device)
    completed: list[dict[str, Any]] = []
    episode_count = success_count = fall_count = timeout_count = 0
    reward_sum = 0.0
    last_context_labels = torch.zeros(n, dtype=torch.long, device=runner.env.device)
    with torch.no_grad():
        for _ in range(runner.cfg["num_steps_per_env"]):
            raw_actions = runner.alg.act(obs)
            next_obs, rewards, dones, extras = runner.env.step(
                raw_actions.to(runner.env.device)
            )
            check_nan(next_obs, rewards, dones)
            extras = dict(extras)
            episode_returns += rewards
            reward_sum += float(rewards.sum())
            riser = extras["online_stair_index"].long()
            episode_max_riser = torch.maximum(episode_max_riser, riser)
            labels = extras.get("v32_context_index")
            if labels is not None:
                last_context_labels = labels.long()
            done_mask = dones.bool()
            if bool(done_mask.any()):
                fell = extras["online_fell"].bool()
                timeouts = extras.get(
                    "time_outs", torch.zeros_like(done_mask, dtype=torch.bool)
                ).bool()
                success = _success_term(runner.env)
                ids = done_mask.nonzero(as_tuple=False).flatten()
                for index in ids.tolist():
                    label = (
                        MIXED_CONTEXTS[int(last_context_labels[index])]
                        if context_name == "mixed"
                        else context_name
                    )
                    completed.append(
                        {
                            "context": label,
                            "return": float(episode_returns[index]),
                            "reached_riser": int(episode_max_riser[index]),
                            "success": bool(success[index]),
                            "fell": bool(fell[index]),
                            "timeout": bool(timeouts[index]),
                        }
                    )
                episode_count += len(ids)
                success_count += int((done_mask & success).sum())
                fall_count += int((done_mask & fell).sum())
                timeout_count += int((done_mask & timeouts).sum())
                episode_returns[ids] = 0.0
                episode_max_riser[ids] = 0
            obs = next_obs.to(runner.device)
            runner.alg.process_env_step(
                obs,
                rewards.to(runner.device),
                dones.to(runner.device),
                extras,
            )
        rollout = {
            "rollout_episode_count": episode_count,
            "rollout_success_count": success_count,
            "rollout_fall_count": fall_count,
            "rollout_timeout_count": timeout_count,
            "rollout_success_rate": success_count / max(1, episode_count),
            "rollout_fall_rate": fall_count / max(1, episode_count),
            "rollout_mean_return": (
                float(np.mean([item["return"] for item in completed]))
                if completed
                else None
            ),
            "rollout_mean_reached_riser": (
                float(np.mean([item["reached_riser"] for item in completed]))
                if completed
                else None
            ),
            "rollout_mean_reward_per_transition": reward_sum
            / (n * runner.cfg["num_steps_per_env"]),
            "performance_gate_used": False,
            "rollout_context_metrics": {},
        }
        for context in MIXED_CONTEXTS if context_name == "mixed" else (context_name,):
            rows = [item for item in completed if item["context"] == context]
            rollout["rollout_context_metrics"][context] = {
                "episode_count": len(rows),
                "success_rate": sum(item["success"] for item in rows)
                / max(1, len(rows)),
                "fall_rate": sum(item["fell"] for item in rows) / max(1, len(rows)),
                "mean_return": (
                    float(np.mean([item["return"] for item in rows])) if rows else None
                ),
                "mean_reached_riser": (
                    float(np.mean([item["reached_riser"] for item in rows]))
                    if rows
                    else None
                ),
            }
        teacher_metrics = runner.alg.relabel_teacher_transitions()
        runner.alg.compute_returns(obs)
    update = runner.alg.update()
    update.update(teacher_metrics)
    update.update(rollout)
    transitions = n * runner.cfg["num_steps_per_env"]
    update["cbf_intervention_count"] = round(
        float(teacher_metrics["cbf_intervention_fraction"]) * transitions
    )
    total_risers = sum(int(item["reached_riser"]) for item in completed)
    update["cbf_intervention_per_completed_riser"] = update[
        "cbf_intervention_count"
    ] / max(1, total_risers)
    return update


def _integrity_checks(metrics: dict[str, Any], *, preflight: bool) -> dict[str, bool]:
    expected_updates = (1 if preflight else ACTOR_EPOCHS) * 4
    expected_critic = (1 if preflight else CRITIC_EPOCHS) * 4
    finite_keys = ("actor_loss", "value", "moving_forward_kl", "teacher_loss")
    return {
        "finite_core_metrics": all(
            math.isfinite(float(metrics[key])) for key in finite_keys
        ),
        "raw_action_storage": float(metrics["policy_storage_max_abs_error"]) < 1.0e-6,
        "safe_action_execution_routing": float(
            metrics["executed_action_routing_max_abs_error"]
        )
        < 1.0e-5,
        "teacher_reprojection": float(metrics["teacher_reprojection_max_abs_error"])
        < 1.0e-6,
        "behavior_log_prob": max(
            float(metrics["behavior_reference_log_prob_max_abs_error"]),
            float(metrics["behavior_current_log_prob_max_abs_error"]),
        )
        <= BEHAVIOR_LOG_PROB_ATOL,
        "behavior_distribution_parameters": max(
            float(metrics["behavior_reference_distribution_param_max_abs_error"]),
            float(metrics["behavior_current_distribution_param_max_abs_error"]),
        )
        <= 2.0e-5,
        "actor_optimizer_steps": int(metrics["actor_minibatches_completed"])
        == expected_updates,
        "critic_optimizer_steps": int(metrics["critic_minibatches_completed"])
        == expected_critic,
        "no_KL_or_performance_gate": metrics["target_kl_early_stopping_enabled"]
        is False
        and metrics["hard_kl_rollback_enabled"] is False,
    }


def _round_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    return {
        "absolute_round": record["absolute_round"],
        "continuation_round": record["continuation_round"],
        "actor_learning_rate": record["actor_learning_rate"],
        "critic_learning_rate": record["critic_learning_rate"],
        "rollout_success_rate": metrics.get("rollout_success_rate"),
        "rollout_fall_rate": metrics.get("rollout_fall_rate"),
        "rollout_mean_return": metrics.get("rollout_mean_return"),
        "rollout_mean_reached_riser": metrics.get("rollout_mean_reached_riser"),
        "actor_loss": metrics.get("actor_loss"),
        "value_loss": metrics.get("value"),
        "moving_forward_kl": metrics.get("moving_forward_kl"),
        "behavior_approx_kl": metrics.get("behavior_approx_kl"),
        "clip_fraction": metrics.get("clip_fraction"),
        "actor_gradient_norm": metrics.get("actor_gradient_norm_pre_clip_max"),
        "teacher_transition_count": metrics.get("teacher_eligible_count"),
        "teacher_weight_sum": metrics.get("teacher_weight_sum"),
        "teacher_loss": metrics.get("teacher_loss"),
        "mean_cbf_correction_norm": metrics.get("mean_cbf_correction_norm"),
        "mean_policy_to_target_distance_before_update": metrics.get(
            "mean_policy_to_target_distance_before_update"
        ),
        "mean_policy_to_target_distance_after_update": metrics.get(
            "mean_policy_to_target_distance_after_update"
        ),
        "cbf_intervention_count": metrics.get("cbf_intervention_count"),
        "cbf_intervention_per_completed_riser": metrics.get(
            "cbf_intervention_per_completed_riser"
        ),
        "round_start_actor_sha256": record["round_start_actor_sha256"],
        "round_end_actor_sha256": record["round_end_actor_sha256"],
        "context_assignment_json": json.dumps(
            record.get("context_assignment"), separators=(",", ":"), sort_keys=True
        ),
        "rollout_context_metrics_json": json.dumps(
            metrics.get("rollout_context_metrics", {}),
            separators=(",", ":"),
            sort_keys=True,
        ),
        "integrity_checks_json": json.dumps(
            record["integrity_checks"], separators=(",", ":"), sort_keys=True
        ),
    }


def _write_round_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = [_round_row(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resume_records(
    output: Path, *, first_round: int, final_round: int
) -> tuple[list[dict[str, Any]], Path]:
    marker = output / "execution_started.json"
    metrics_path = output / "round_metrics.json"
    if not marker.is_file() or not metrics_path.is_file():
        raise RuntimeError("v32 resume requires marker and round metrics")
    if (output / "execution_completed.json").exists():
        raise RuntimeError("v32 run is already complete")
    records = json.loads(metrics_path.read_text())
    expected = list(range(first_round, first_round + len(records)))
    actual = [int(item["absolute_round"]) for item in records]
    if not records or actual != expected or actual[-1] >= final_round:
        raise RuntimeError("v32 resume metrics are not a valid incomplete prefix")
    checkpoint = output / f"round_{actual[-1]:02d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return records, checkpoint


def _build_single_env(
    *, protocol: dict[str, Any], context: str, num_envs: int, seed: int, device: str
):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from src.tasks.stairs_cbf.environment_v31 import configure_v31_context

    env_cfg = load_env_cfg(TASK_ID, play=True)
    shift = configure_v31_context(
        env_cfg,
        context=context,
        runtime_filter=True,
        context_spec=protocol["contexts"][context],
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    base_env = ManagerBasedRlEnv(env_cfg, device=device)
    agent_cfg = load_rl_cfg(TASK_ID)
    wrapper = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    return base_env, wrapper, env_cfg, shift


def _build_mixed_env(*, protocol: dict[str, Any], seed: int, device: str):
    from mixed_vec_env_v32 import MixedContextVecEnvV32

    bases = {}
    wrappers = {}
    cfgs = {}
    shifts = {}
    try:
        for index, context in enumerate(MIXED_CONTEXTS):
            base, wrapper, cfg, shift = _build_single_env(
                protocol=protocol,
                context=context,
                num_envs=22,
                seed=seed + 1_000 * (index + 1),
                device=device,
            )
            bases[context] = base
            wrappers[context] = wrapper
            cfgs[context] = cfg
            shifts[context] = shift
        mixed = MixedContextVecEnvV32(wrappers, absolute_round=1)
        return bases, mixed, cfgs, shifts
    except Exception:
        for wrapper in wrappers.values():
            wrapper.close()
        raise


def main() -> None:
    args = _parse_args()
    _validate_combination(args)
    repo = args.repo.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    v31_root = args.v31_formal_root.resolve()
    protocol_path = args.protocol.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v32 execution requires a clean committed worktree")
    if file_sha256(base_checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v32 base checkpoint differs")
    input_checkpoint = (
        v31_root / args.context / "A2" / "round_08.pt"
        if args.kind == "continuation"
        else base_checkpoint
    )
    if not input_checkpoint.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("v32 input checkpoint or protocol is missing")
    protocol, protocol_reference = _validate_protocol(
        repo,
        protocol_path,
        input_checkpoint,
        phase=args.phase,
        kind=args.kind,
        context=args.context,
        schedule=args.schedule,
    )
    preflight = args.phase == "preflight"
    seed = (
        _preflight_seed(args.kind, args.context, args.schedule)
        if preflight
        else continuation_seed(args.context, args.schedule)
        if args.kind == "continuation"
        else MIXED_SEED
    )
    first_round = 9 if args.kind == "continuation" else 1
    formal_final_round = (
        CONTINUATION_FINAL_ROUND if args.kind == "continuation" else MIXED_FINAL_ROUND
    )
    final_round = first_round if preflight else formal_final_round
    rollout_steps = PREFLIGHT_STEPS if preflight else ROLLOUT_STEPS
    num_envs = (
        PREFLIGHT_CONTINUATION_ENVS
        if preflight and args.kind == "continuation"
        else NUM_ENVS
    )
    if output.exists() and not args.resume:
        raise RuntimeError(f"v32 output already exists: {output}")
    records: list[dict[str, Any]] = []
    recovery_checkpoint: Path | None = None
    if args.resume:
        records, recovery_checkpoint = _resume_records(
            output, first_round=first_round, final_round=final_round
        )
    else:
        output.mkdir(parents=True)
        _atomic_json(
            output / "execution_started.json",
            {
                "protocol_id": PROTOCOL_ID,
                "protocol": protocol_reference,
                "phase": args.phase,
                "kind": args.kind,
                "context": args.context,
                "schedule": args.schedule,
                "seed": seed,
                "first_round": first_round,
                "final_round": final_round,
                "num_envs": num_envs,
                "rollout_steps": rollout_steps,
                "input_checkpoint_sha256": file_sha256(input_checkpoint),
                "performance_selection": False,
                "KL_stop_or_rollback": False,
                "infrastructure_resume_allowed": not preflight,
            },
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sys.path.insert(0, str(repo))
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    import src.tasks  # noqa: F401
    from src.tasks.stairs_cbf.teacher_v30 import CbfTeacherV30Runner

    env = None
    runner = None
    try:
        if args.kind == "continuation":
            _, env, env_cfg, shift = _build_single_env(
                protocol=protocol,
                context=args.context,
                num_envs=num_envs,
                seed=seed,
                device=args.device,
            )
            env_cfgs = {args.context: env_cfg}
            shifts = {args.context: shift}
        else:
            _, env, env_cfgs, shifts = _build_mixed_env(
                protocol=protocol, seed=seed, device=args.device
            )
        agent_cfg = load_rl_cfg(TASK_ID)
        agent_cfg.seed = seed
        agent_cfg.num_steps_per_env = rollout_steps
        _configure_algorithm(agent_cfg, "A2", preflight=preflight)
        runner = CbfTeacherV30Runner(
            env, asdict(agent_cfg), log_dir=None, device=args.device
        )
        if preflight:
            runner.alg.critic_learning_epochs = 1
        structural = {
            context: _structural_audit(
                runner, env_cfgs[context], shifts[context], preflight=preflight
            )
            for context in env_cfgs
        }
        if args.kind == "mixed":
            structural["mixed"] = {
                "underlying_envs": 66,
                "exposed_envs": int(env.num_envs),
                "context_capacity": 22,
                "round_1_assignment": env.assignment_metadata(),
                "actor_observation_dim": int(runner.alg.actor.obs_dim),
                "critic_observation_dim": int(runner.alg.critic.obs_dim),
                "episode_context_fixed": True,
            }
            if structural["mixed"]["exposed_envs"] != 64:
                raise RuntimeError("v32 mixed structural env count differs")

        if recovery_checkpoint is not None:
            payload = torch.load(
                recovery_checkpoint, map_location=args.device, weights_only=False
            )
            warm_start = runner.load_recovery_checkpoint(
                str(recovery_checkpoint), map_location=args.device
            )
            _restore_rng_state(payload)
        elif args.kind == "continuation":
            warm_start = runner.load_recovery_checkpoint(
                str(input_checkpoint), map_location=args.device
            )
            if not preflight:
                shutil.copy2(input_checkpoint, output / "round_08.pt")
        else:
            warm_start = runner.load_initial_checkpoint(
                str(input_checkpoint), map_location=args.device
            )
            if not preflight:
                _save_checkpoint(
                    runner,
                    output / "round_00.pt",
                    0,
                    {"boundary": "base", "kind": "mixed", "context": "mixed"},
                )
        initial_actor_hash = (
            actor_state_sha256(actor_state(runner.alg.actor))
            if not records
            else json.loads((output / "execution_started.json").read_text()).get(
                "initial_actor_sha256"
            )
        )
        marker_path = output / "execution_started.json"
        marker = json.loads(marker_path.read_text())
        if marker.get("initial_actor_sha256") is None:
            marker["initial_actor_sha256"] = initial_actor_hash
            _atomic_json(marker_path, marker)

        for absolute_round in range(first_round + len(records), final_round + 1):
            if args.kind == "mixed":
                env.set_round(absolute_round)
                assignment = env.assignment_metadata()
                if assignment["context_counts"] != mixed_context_env_counts(
                    absolute_round
                ):
                    raise RuntimeError(
                        "v32 mixed runtime assignment differs from protocol"
                    )
            else:
                assignment = {args.context: num_envs}
            actor_lr, critic_lr = learning_rates(
                args.kind, args.schedule, absolute_round
            )
            _set_learning_rates(runner, actor_lr, critic_lr)
            runner.alg.freeze_round_reference()
            start_hash = actor_state_sha256(actor_state(runner.alg.actor))
            started = time.monotonic()
            metrics = _collect_round(runner, context_name=args.context)
            metrics["actor_learning_rate"] = actor_lr
            metrics["critic_learning_rate"] = critic_lr
            checks = _integrity_checks(metrics, preflight=preflight)
            if not all(checks.values()):
                raise RuntimeError(f"v32 program-integrity check failed: {checks}")
            end_hash = actor_state_sha256(actor_state(runner.alg.actor))
            record = {
                "absolute_round": absolute_round,
                "continuation_round": absolute_round
                - (CONTINUATION_START_ROUND if args.kind == "continuation" else 0),
                "status": "updated",
                "actor_learning_rate": actor_lr,
                "critic_learning_rate": critic_lr,
                "round_start_actor_sha256": start_hash,
                "round_end_actor_sha256": end_hash,
                "round_reference_is_exact_pi_k": True,
                "performance_gate_used": False,
                "KL_gate_used": False,
                "context_assignment": assignment,
                "integrity_checks": checks,
                "elapsed_seconds": time.monotonic() - started,
                "metrics": metrics,
            }
            records.append(record)
            _save_checkpoint(
                runner,
                output / f"round_{absolute_round:02d}.pt",
                absolute_round,
                {
                    "boundary": "round_end",
                    "kind": args.kind,
                    "context": args.context,
                    "schedule": args.schedule,
                    "actor_sha256": end_hash,
                },
            )
            _atomic_json(output / "round_metrics.json", records)
            _write_round_csv(output / "round_metrics.csv", records)
            print(json.dumps(record, sort_keys=True), flush=True)

        final_checkpoint = output / f"round_{final_round:02d}.pt"
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "policy_method": POLICY_METHOD,
            "phase": args.phase,
            "kind": args.kind,
            "context": args.context,
            "schedule": args.schedule,
            "A2_configuration": a2_parameters(),
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "protocol": protocol_reference,
            "input_checkpoint_sha256": file_sha256(input_checkpoint),
            "seed": seed,
            "first_round": first_round,
            "final_round": final_round,
            "rounds_completed": len(records),
            "rounds": records,
            "structural_audit": structural,
            "warm_start": warm_start,
            "initial_actor_sha256": initial_actor_hash,
            "final_actor_sha256": actor_state_sha256(actor_state(runner.alg.actor)),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "final_policy_rule": f"unconditional round {final_round} actor",
            "monitor_nodes": [8, 16, 24],
            "performance_selection_count": 0,
            "KL_stop_count": 0,
            "KL_rollback_count": 0,
            "infrastructure_resume_count": int(recovery_checkpoint is not None),
            "elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in records),
        }
        if preflight:
            summary["passed"] = all(
                all(record["integrity_checks"].values()) for record in records
            )
            _atomic_json(output / "preflight_case_summary.json", summary)
            if not summary["passed"]:
                raise RuntimeError("v32 functional preflight did not pass")
        else:
            expected_rounds = 16 if args.kind == "continuation" else 24
            if len(records) != expected_rounds or final_round != 24:
                raise RuntimeError("v32 formal run did not reach fixed round 24")
            _atomic_json(output / "training_summary.json", summary)
            _atomic_json(
                output / "execution_completed.json",
                {
                    "protocol_id": PROTOCOL_ID,
                    "kind": args.kind,
                    "context": args.context,
                    "schedule": args.schedule,
                    "final_round": final_round,
                    "rounds_completed": len(records),
                    "final_checkpoint_sha256": file_sha256(final_checkpoint),
                    "training_summary_sha256": file_sha256(
                        output / "training_summary.json"
                    ),
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        _atomic_json(
            output / "run_failure.json",
            {
                "protocol_id": PROTOCOL_ID,
                "phase": args.phase,
                "kind": args.kind,
                "context": args.context,
                "schedule": args.schedule,
                "error_type": type(error).__name__,
                "error": str(error),
                "performance_result": False,
            },
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

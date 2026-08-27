"""Run one fixed v31 preflight case or formal adaptation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cbf_teacher_v31_protocol import (
    ACTOR_EPOCHS,
    ACTOR_LEARNING_RATE,
    ARMS,
    BASE_CHECKPOINT_SHA256,
    CLEARANCE_BARRIER_SLOPE,
    CRITIC_EPOCHS,
    CRITIC_LEARNING_RATE,
    ENTROPY_COEFFICIENT,
    FILTER_ALPHA,
    FORMAL_CONTEXTS,
    GAE_LAMBDA,
    GAMMA,
    MAX_GRAD_NORM,
    MAXIMUM_STD,
    MINI_BATCHES,
    MINIMUM_STD,
    MOVING_KL_BETA,
    NUM_ENVS,
    POLICY_METHOD,
    PPO_CLIP,
    PREFLIGHT_ENVS,
    PREFLIGHT_STEPS,
    PROTOCOL_ID,
    RECOVERY_DISTANCE_M,
    ROLLOUT_STEPS,
    ROUNDS,
    SOURCE_FILES,
    STD_SCALE_FROM_BASE,
    TASK_ID,
    TEACHER_CORRECTION_SCALE,
    TEACHER_HORIZON,
    adaptation_seed,
    arm_parameters,
    common_training_parameters,
    environment_parameters,
    preflight_seed,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "formal"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--context", choices=FORMAL_CONTEXTS, required=True)
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


def _validate_protocol(
    repo: Path,
    path: Path,
    checkpoint: Path,
    *,
    phase: str,
    arm: str,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text())
    source_boundary = protocol.get("source_boundary", {})
    source_hashes = source_boundary.get("source_files", {})
    expected_status = "frozen_before_v31_preflight_and_formal"
    relative = path.relative_to(repo)
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    checks = {
        "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
        "status": protocol.get("status") == expected_status,
        "protocol_committed": committed == path.read_bytes(),
        "base_checkpoint": protocol.get("base_checkpoint", {}).get("sha256")
        == file_sha256(checkpoint)
        == BASE_CHECKPOINT_SHA256,
        "common_training": protocol.get("common_training")
        == common_training_parameters(),
        "source_set": set(source_hashes) == set(SOURCE_FILES),
        "source_hashes": all(
            (repo / relative_source).is_file()
            and file_sha256(repo / relative_source)
            == source_hashes.get(relative_source)
            for relative_source in SOURCE_FILES
        ),
        "source_commit_is_ancestor": subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                str(source_boundary.get("git_commit", "")),
                "HEAD",
            ],
            cwd=repo,
            check=False,
        ).returncode
        == 0,
        "arm_matches": protocol.get("methods", {}).get(arm) == arm_parameters(arm),
        "context_matches": protocol.get("contexts", {}).get(context)
        == environment_parameters(context),
    }
    if phase == "preflight":
        checks["preflight_case_frozen"] = any(
            item["context"] == context
            and item["arm"] == arm
            and item["seed"] == preflight_seed(context, arm)
            for item in protocol.get("preflight", {}).get("cases", [])
        )
    else:
        checks.update(
            {
                "formal_context": context in FORMAL_CONTEXTS,
                "formal_arm": arm in ARMS,
                "formal_seed_frozen": protocol.get("formal", {})
                .get("adaptation_seeds", {})
                .get(context)
                == adaptation_seed(context),
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"v31 protocol validation failed: {checks}")
    return protocol, {
        "path": str(path),
        "sha256": file_sha256(path),
        "checks": checks,
    }


def _configure_algorithm(agent_cfg, arm: str, *, preflight: bool) -> None:
    from src.tasks.stairs_cbf.teacher_v30 import CbfTeacherV30PpoAlgorithmCfg

    arm_cfg = arm_parameters(arm)
    agent_cfg.algorithm = CbfTeacherV30PpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=PPO_CLIP,
        entropy_coef=ENTROPY_COEFFICIENT,
        num_learning_epochs=1 if preflight else ACTOR_EPOCHS,
        num_mini_batches=MINI_BATCHES,
        learning_rate=ACTOR_LEARNING_RATE,
        actor_learning_rate=ACTOR_LEARNING_RATE,
        critic_learning_rate=CRITIC_LEARNING_RATE,
        schedule="fixed",
        gamma=GAMMA,
        lam=GAE_LAMBDA,
        desired_kl=0.003,
        hard_kl_ceiling=0.01,
        max_grad_norm=MAX_GRAD_NORM,
        normalize_advantage_per_mini_batch=False,
        std_scale_from_base=STD_SCALE_FROM_BASE,
        minimum_std=MINIMUM_STD,
        maximum_std=MAXIMUM_STD,
        moving_kl_beta=MOVING_KL_BETA,
        # The inherited constructor validates its historical two-epoch value.
        # The one-off functional preflight switches this to one immediately after
        # construction, before any rollout or optimizer call.
        critic_learning_epochs=CRITIC_EPOCHS,
        freeze_log_std=True,
        teacher_distillation_weight=arm_cfg["teacher_weight"],
        teacher_success_horizon=TEACHER_HORIZON,
        teacher_correction_scale=TEACHER_CORRECTION_SCALE,
        teacher_mode=arm_cfg["teacher_mode"],
        teacher_gate=arm_cfg["teacher_gate"],
        teacher_eta=arm_cfg["teacher_eta"],
        teacher_smooth_l1_beta=0.05,
        v30_smoke_all_arm_diagnostics=False,
    )


def _structural_audit(
    runner, env_cfg, shift: dict[str, Any], *, preflight: bool
) -> dict[str, Any]:
    from src.tasks.stairs_cbf.teacher_v26 import HigherRiserCbfActionCfg
    from src.tasks.stairs_cbf.teacher_v30 import VALID_CONFIGURATIONS, CbfTeacherV30PPO

    alg = runner.alg
    action_cfg = env_cfg.actions["joint_pos"]
    values = {
        "v30_algorithm": isinstance(alg, CbfTeacherV30PPO),
        "cbf_action": isinstance(action_cfg, HigherRiserCbfActionCfg),
        "actor_observation_dim": int(alg.actor.obs_dim),
        "critic_observation_dim": int(alg.critic.obs_dim),
        "actor_groups": list(alg.actor.obs_groups),
        "critic_groups": list(alg.critic.obs_groups),
        "single_actor": 1,
        "single_critic": 1,
        "auxiliary_modules_absent": all(
            module is None
            for module in (alg.fall_critic, alg.intervention_critic, alg.risk_head)
        ),
        "runtime_cbf": bool(action_cfg.enabled),
        "plant_identity": shift["plant_action_transform"] == "identity",
        "actor_epochs": int(alg.num_learning_epochs),
        "critic_epochs": int(alg.critic_learning_epochs),
        "minibatches": int(alg.num_mini_batches),
        "target_kl_early_stopping_enabled": False,
        "hard_kl_rollback_enabled": False,
        "all_six_arm_configurations": len(VALID_CONFIGURATIONS),
        "num_risers": int(action_cfg.num_steps),
        "stair_target_patch_slots": int(shift["stair_target_patch_slots"]),
        "top_platform_patch_included": bool(shift["top_platform_patch_included"]),
        "log_std_trainable_parameters": sum(
            parameter.requires_grad for parameter in alg.actor.distribution.parameters()
        ),
        "reward_terms": sorted(env_cfg.rewards),
        "termination_terms": sorted(env_cfg.terminations),
    }
    expected = {
        "v30_algorithm": True,
        "cbf_action": True,
        "actor_observation_dim": 405,
        "critic_observation_dim": 838,
        "actor_groups": ["actor"],
        "critic_groups": ["actor", "critic", "online_privileged"],
        "single_actor": 1,
        "single_critic": 1,
        "auxiliary_modules_absent": True,
        "runtime_cbf": True,
        "plant_identity": True,
        "actor_epochs": 1 if preflight else ACTOR_EPOCHS,
        "critic_epochs": 1 if preflight else CRITIC_EPOCHS,
        "minibatches": MINI_BATCHES,
        "target_kl_early_stopping_enabled": False,
        "hard_kl_rollback_enabled": False,
        "all_six_arm_configurations": 6,
        "num_risers": int(shift["num_risers"]),
        "stair_target_patch_slots": int(shift["num_risers"]) + 1,
        "top_platform_patch_included": True,
        "log_std_trainable_parameters": 0,
    }
    mismatches = {
        key: {"actual": values[key], "expected": expected_value}
        for key, expected_value in expected.items()
        if values[key] != expected_value
    }
    if mismatches:
        raise RuntimeError(f"v31 structural audit failed: {mismatches}")
    return values


def _save_checkpoint(
    runner, path: Path, round_index: int, metadata: dict[str, Any]
) -> None:
    payload = runner.alg.save()
    payload["iter"] = round_index
    payload["infos"] = {"cbf_teacher_v31": metadata}
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
    """Restore the exact post-round RNG boundary for infrastructure recovery."""
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda_random_state_all"])


def _resume_boundary(
    output_dir: Path, rounds: int
) -> tuple[list[dict[str, Any]], Path]:
    metrics_path = output_dir / "round_metrics.json"
    marker = output_dir / "execution_started.json"
    if not marker.is_file() or not metrics_path.is_file():
        raise RuntimeError("v31 resume requires an execution marker and round metrics")
    if (output_dir / "execution_completed.json").exists():
        raise RuntimeError("v31 run is already complete and cannot be resumed")
    records = json.loads(metrics_path.read_text())
    expected = list(range(1, len(records) + 1))
    if [int(item["round"]) for item in records] != expected:
        raise RuntimeError("v31 resume metrics are not a consecutive round prefix")
    if not records or len(records) >= rounds:
        raise RuntimeError("v31 resume requires one through seven complete rounds")
    checkpoint = output_dir / f"round_{len(records):02d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return records, checkpoint


ROUND_FIELDS = (
    "round",
    "rollout_success_rate",
    "rollout_fall_rate",
    "rollout_mean_return",
    "rollout_mean_reached_riser",
    "actor_loss",
    "value_loss",
    "moving_forward_kl",
    "behavior_approx_kl",
    "clip_fraction",
    "action_mean_shift",
    "actor_gradient_norm",
    "actor_gradient_clipped_fraction",
    "teacher_transition_count",
    "teacher_fraction_among_interventions",
    "teacher_weight_sum",
    "teacher_loss",
    "teacher_huber_loss_mean",
    "teacher_smooth_l1_beta",
    "mean_cbf_correction_norm",
    "mean_residual_target_norm",
    "mean_policy_to_target_distance_before_update",
    "mean_policy_to_target_distance_after_update",
    "teacher_mode",
    "teacher_gate_mode",
    "teacher_eta",
    "cbf_intervention_count",
    "cbf_intervention_per_riser",
    "toe_riser_kick_event_count",
    "round_start_actor_sha256",
    "round_end_actor_sha256",
    "epoch_moving_forward_kl_json",
    "minibatch_diagnostics_json",
    *(f"teacher_error_before_action_{index:02d}" for index in range(12)),
    *(f"teacher_error_after_action_{index:02d}" for index in range(12)),
)


def _write_round_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        metrics = record["metrics"]
        minibatches = metrics.get("minibatch_diagnostics", [])
        before_errors = metrics.get("per_action_teacher_error_before_update", [])
        after_errors = metrics.get("per_action_teacher_error_after_update", [])
        row = {
            "round": record["round"],
            "rollout_success_rate": metrics.get("rollout_success_rate"),
            "rollout_fall_rate": metrics.get("rollout_fall_rate"),
            "rollout_mean_return": metrics.get("rollout_mean_return"),
            "rollout_mean_reached_riser": metrics.get("rollout_mean_reached_riser"),
            "actor_loss": metrics.get("actor_loss"),
            "value_loss": metrics.get("value"),
            "moving_forward_kl": metrics.get("moving_forward_kl"),
            "behavior_approx_kl": metrics.get("behavior_approx_kl"),
            "clip_fraction": metrics.get("clip_fraction"),
            "action_mean_shift": (
                sum(float(item["action_mean_shift"]) for item in minibatches)
                / len(minibatches)
                if minibatches
                else None
            ),
            "actor_gradient_norm": metrics.get("actor_gradient_norm_pre_clip_max"),
            "actor_gradient_clipped_fraction": metrics.get(
                "actor_gradient_clipped_fraction"
            ),
            "teacher_transition_count": metrics.get("teacher_eligible_count"),
            "teacher_fraction_among_interventions": metrics.get(
                "teacher_fraction_among_interventions"
            ),
            "teacher_weight_sum": metrics.get("teacher_weight_sum"),
            "teacher_loss": metrics.get("teacher_loss"),
            "teacher_huber_loss_mean": metrics.get("teacher_huber_loss_mean"),
            "teacher_smooth_l1_beta": metrics.get("teacher_smooth_l1_beta"),
            "mean_cbf_correction_norm": metrics.get("mean_cbf_correction_norm"),
            "mean_residual_target_norm": metrics.get("mean_residual_target_norm"),
            "mean_policy_to_target_distance_before_update": metrics.get(
                "mean_policy_to_target_distance_before_update"
            ),
            "mean_policy_to_target_distance_after_update": metrics.get(
                "mean_policy_to_target_distance_after_update"
            ),
            "teacher_mode": metrics.get("teacher_mode"),
            "teacher_gate_mode": metrics.get("teacher_gate_mode"),
            "teacher_eta": metrics.get("teacher_eta"),
            "cbf_intervention_count": metrics.get("cbf_intervention_count"),
            "cbf_intervention_per_riser": metrics.get("cbf_intervention_per_riser"),
            "toe_riser_kick_event_count": metrics.get("toe_riser_kick_event_count"),
            "round_start_actor_sha256": record["round_start_actor_sha256"],
            "round_end_actor_sha256": record["round_end_actor_sha256"],
            "epoch_moving_forward_kl_json": json.dumps(
                metrics.get("epoch_moving_forward_kl", []),
                separators=(",", ":"),
            ),
            "minibatch_diagnostics_json": json.dumps(
                minibatches, separators=(",", ":"), sort_keys=True
            ),
        }
        row.update(
            {
                f"teacher_error_before_action_{index:02d}": (
                    before_errors[index] if index < len(before_errors) else None
                )
                for index in range(12)
            }
        )
        row.update(
            {
                f"teacher_error_after_action_{index:02d}": (
                    after_errors[index] if index < len(after_errors) else None
                )
                for index in range(12)
            }
        )
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _collect_round(
    runner, *, before_env_step=None, before_process_env_step=None
) -> dict[str, Any]:
    from rsl_rl.utils import check_nan

    runner.alg.clear_cbf_rollout()
    runner.alg.train_mode()
    obs, _ = runner.env.reset()
    obs = obs.to(runner.device)
    n = runner.env.num_envs
    episode_returns = torch.zeros(n, device=runner.env.device)
    episode_max_riser = torch.zeros(n, dtype=torch.long, device=runner.env.device)
    completed_returns: list[float] = []
    completed_risers: list[int] = []
    episode_count = success_count = fall_count = timeout_count = 0
    reward_sum = 0.0
    with torch.no_grad():
        for _ in range(runner.cfg["num_steps_per_env"]):
            raw_actions = runner.alg.act(obs)
            if before_env_step is not None:
                before_env_step(runner, raw_actions)
            next_obs, rewards, dones, extras = runner.env.step(
                raw_actions.to(runner.env.device)
            )
            check_nan(next_obs, rewards, dones)
            extras = dict(extras)
            if before_process_env_step is not None:
                before_process_env_step(runner, dones, extras)
            episode_returns += rewards
            reward_sum += float(rewards.sum())
            riser = extras["online_stair_index"].long()
            episode_max_riser = torch.maximum(episode_max_riser, riser)
            done_mask = dones.bool()
            if bool(done_mask.any()):
                fell = extras["online_fell"].bool()
                timeouts = extras.get(
                    "time_outs", torch.zeros_like(done_mask, dtype=torch.bool)
                ).bool()
                success = runner.env.unwrapped.termination_manager.get_term(
                    "reached_top"
                ).bool()
                ids = done_mask.nonzero(as_tuple=False).flatten()
                completed_returns.extend(
                    float(episode_returns[index]) for index in ids.tolist()
                )
                completed_risers.extend(
                    int(episode_max_riser[index]) for index in ids.tolist()
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
                sum(completed_returns) / len(completed_returns)
                if completed_returns
                else None
            ),
            "rollout_mean_reached_riser": (
                sum(completed_risers) / len(completed_risers)
                if completed_risers
                else None
            ),
            "rollout_mean_reward_per_transition": reward_sum
            / (n * runner.cfg["num_steps_per_env"]),
            "performance_gate_used": False,
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
    return update


def _preflight_checks(summary: dict[str, Any]) -> dict[str, bool]:
    metrics = summary["rounds"][0]["metrics"]
    target_shape = metrics.get("teacher_target_shape", [])
    shapes = metrics.get("teacher_tensor_shapes", {})
    action_shape = [PREFLIGHT_STEPS, PREFLIGHT_ENVS, 12]
    scalar_shape = [PREFLIGHT_STEPS, PREFLIGHT_ENVS]
    checks = {
        "teacher_tensor_shape": target_shape == action_shape
        and all(
            shapes.get(name) == action_shape
            for name in (
                "raw_sampled_action",
                "safe_sampled_action",
                "round_reference_mean",
                "correction_vector",
            )
        )
        and all(
            shapes.get(name) == scalar_shape
            for name in ("intervention", "riser", "fall", "episode")
        )
        and shapes.get("done") == [PREFLIGHT_STEPS, PREFLIGHT_ENVS, 1],
        "configured_teacher_loss_finite": math.isfinite(float(metrics["teacher_loss"])),
        "raw_policy_action_stored": metrics["policy_storage_max_abs_error"] < 1.0e-6,
        "safe_action_executed": metrics["executed_action_routing_max_abs_error"]
        < 1.0e-5,
        "teacher_transform_exact": metrics["teacher_reprojection_max_abs_error"]
        < 1.0e-6,
        "actor_backward_and_steps_complete": metrics["actor_minibatches_completed"]
        == MINI_BATCHES,
        "critic_backward_and_steps_complete": metrics["critic_minibatches_completed"]
        == MINI_BATCHES,
        "one_actor_epoch": metrics["actor_epochs_completed"] == 1,
        "one_critic_epoch": metrics["critic_epochs_completed"] == 1,
        "behavior_log_prob_within_v31_tolerance": max(
            float(metrics["behavior_reference_log_prob_max_abs_error"]),
            float(metrics["behavior_current_log_prob_max_abs_error"]),
        )
        <= 1.0e-3,
        "behavior_distribution_parameters_strict": max(
            float(metrics["behavior_reference_distribution_param_max_abs_error"]),
            float(metrics["behavior_current_distribution_param_max_abs_error"]),
        )
        <= 2.0e-5,
        "dynamic_stair_patch_allocation": summary["structural_audit"][
            "stair_target_patch_slots"
        ]
        == summary["structural_audit"]["num_risers"] + 1,
        "reset_step_and_update_complete": summary["rounds_completed"] == 1,
        "no_nan_or_inf": all(
            math.isfinite(float(metrics[key]))
            for key in (
                "actor_loss",
                "value",
                "moving_forward_kl",
                "teacher_loss",
            )
        ),
        "kl_not_a_gate": metrics["target_kl_early_stopping_enabled"] is False
        and metrics["hard_kl_rollback_enabled"] is False,
    }
    return checks


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v31 execution requires a clean committed worktree")
    if not checkpoint.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("v31 base checkpoint or protocol is missing")
    protocol, protocol_reference = _validate_protocol(
        repo,
        protocol_path,
        checkpoint,
        phase=args.phase,
        arm=args.arm,
        context=args.context,
    )
    preflight = args.phase == "preflight"
    if preflight:
        seed, rounds, num_envs, rollout_steps = (
            preflight_seed(args.context, args.arm),
            1,
            PREFLIGHT_ENVS,
            PREFLIGHT_STEPS,
        )
    else:
        seed = adaptation_seed(args.context)
        rounds, num_envs, rollout_steps = ROUNDS, NUM_ENVS, ROLLOUT_STEPS
    marker = output_dir / "execution_started.json"
    if args.resume and preflight:
        raise RuntimeError("v31 preflight cases cannot be resumed")
    records: list[dict[str, Any]] = []
    recovery_checkpoint: Path | None = None
    if output_dir.exists():
        if not args.resume:
            raise RuntimeError(f"v31 run output already exists: {output_dir}")
        records, recovery_checkpoint = _resume_boundary(output_dir, rounds)
    else:
        if args.resume:
            raise RuntimeError("v31 resume requested but no prior run exists")
        output_dir.mkdir(parents=True)
        _atomic_json(
            marker,
            {
                "protocol_id": PROTOCOL_ID,
                "protocol": protocol_reference,
                "phase": args.phase,
                "arm": args.arm,
                "context": args.context,
                "seed": seed,
                "rounds": rounds,
                "num_envs": num_envs,
                "rollout_steps": rollout_steps,
                "performance_selection": False,
                "kl_stop_or_rollback": False,
                "infrastructure_resume_allowed": True,
            },
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sys.path.insert(0, str(repo))
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import src.tasks  # noqa: F401
    from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
    from src.tasks.stairs_cbf.teacher_v30 import CbfTeacherV30Runner

    env_cfg = load_env_cfg(TASK_ID, play=True)
    shift = configure_v31_context(
        env_cfg,
        context=args.context,
        runtime_filter=True,
        context_spec=protocol["contexts"][args.context],
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(TASK_ID)
    agent_cfg.seed = seed
    agent_cfg.num_steps_per_env = rollout_steps
    _configure_algorithm(agent_cfg, args.arm, preflight=preflight)
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = CbfTeacherV30Runner(
        env, asdict(agent_cfg), log_dir=None, device=args.device
    )
    if preflight:
        runner.alg.critic_learning_epochs = 1
    try:
        structural = _structural_audit(runner, env_cfg, shift, preflight=preflight)
        if recovery_checkpoint is None:
            warm_start = runner.load_initial_checkpoint(
                str(checkpoint), map_location=args.device
            )
            initial_hash = actor_state_sha256(actor_state(runner.alg.actor))
            _save_checkpoint(
                runner,
                output_dir / "round_00.pt",
                0,
                {"boundary": "base", "arm": args.arm, "context": args.context},
            )
        else:
            recovery_payload = torch.load(
                recovery_checkpoint, map_location=args.device, weights_only=False
            )
            warm_start = runner.load_recovery_checkpoint(
                str(recovery_checkpoint), map_location=args.device
            )
            _restore_rng_state(recovery_payload)
            round_zero = torch.load(
                output_dir / "round_00.pt",
                map_location="cpu",
                weights_only=False,
            )
            initial_hash = actor_state_sha256(round_zero["actor_state_dict"])
        for round_index in range(len(records) + 1, rounds + 1):
            runner.alg.freeze_round_reference()
            start_hash = actor_state_sha256(actor_state(runner.alg.actor))
            round_started = time.monotonic()
            metrics = _collect_round(runner)
            end_hash = actor_state_sha256(actor_state(runner.alg.actor))
            record = {
                "round": round_index,
                "status": "updated",
                "round_start_actor_sha256": start_hash,
                "round_end_actor_sha256": end_hash,
                "round_reference_is_exact_pi_k": True,
                "performance_gate_used": False,
                "kl_gate_used": False,
                "elapsed_seconds": time.monotonic() - round_started,
                "metrics": metrics,
            }
            records.append(record)
            _save_checkpoint(
                runner,
                output_dir / f"round_{round_index:02d}.pt",
                round_index,
                {
                    "boundary": "round_end",
                    "arm": args.arm,
                    "context": args.context,
                    "actor_sha256": end_hash,
                },
            )
            _atomic_json(output_dir / "round_metrics.json", records)
            _write_round_csv(output_dir / "round_metrics.csv", records)
            print(json.dumps(record, sort_keys=True), flush=True)
        final_checkpoint = output_dir / f"round_{rounds:02d}.pt"
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "policy_method": POLICY_METHOD,
            "phase": args.phase,
            "arm": args.arm,
            "arm_configuration": arm_parameters(args.arm),
            "context": args.context,
            "environment": environment_parameters(args.context),
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "protocol": protocol_reference,
            "base_checkpoint_sha256": file_sha256(checkpoint),
            "seed": seed,
            "common_training": common_training_parameters(),
            "structural_audit": structural,
            "warm_start": warm_start,
            "initial_actor_sha256": initial_hash,
            "final_actor_sha256": actor_state_sha256(actor_state(runner.alg.actor)),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "final_policy_rule": f"unconditional round {rounds} actor",
            "saved_checkpoint_rounds": list(range(rounds + 1)),
            "rounds": records,
            "rounds_completed": len(records),
            "kl_stop_count": 0,
            "kl_rollback_count": 0,
            "performance_selection_count": 0,
            "infrastructure_resume_count": int(recovery_checkpoint is not None),
            "elapsed_seconds": sum(
                float(record["elapsed_seconds"]) for record in records
            ),
        }
        if preflight:
            checks = _preflight_checks(summary)
            summary.update({"checks": checks, "passed": all(checks.values())})
            _atomic_json(output_dir / "preflight_case_summary.json", summary)
            if not summary["passed"]:
                raise RuntimeError(f"v31 preflight case failed: {checks}")
        else:
            _atomic_json(output_dir / "training_summary.json", summary)
            _atomic_json(
                output_dir / "execution_completed.json",
                {
                    "protocol_id": PROTOCOL_ID,
                    "phase": args.phase,
                    "arm": args.arm,
                    "context": args.context,
                    "rounds_completed": rounds,
                    "final_checkpoint_sha256": file_sha256(final_checkpoint),
                    "training_summary_sha256": file_sha256(
                        output_dir / "training_summary.json"
                    ),
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        _atomic_json(
            output_dir / "run_failure.json",
            {
                "protocol_id": PROTOCOL_ID,
                "phase": args.phase,
                "arm": args.arm,
                "context": args.context,
                "error_type": type(error).__name__,
                "error": str(error),
                "allowed_reason": (
                    "nonfinite_routing_transform_or_optimizer_corruption"
                ),
            },
        )
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()

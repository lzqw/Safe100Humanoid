"""Run the one smoke or the sole fixed eight-round v29 adaptation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cbf_teacher_v29_protocol import (
    ACTOR_LEARNING_RATE,
    ADAPTATION_SEED,
    BASE_CHECKPOINT_SHA256,
    CLEARANCE_BARRIER_SLOPE,
    CRITIC_EPOCHS,
    CRITIC_LEARNING_RATE,
    ENTROPY_COEFFICIENT,
    FILTER_ALPHA,
    GAE_LAMBDA,
    GAMMA,
    HARD_KL_CEILING,
    MAXIMUM_STD,
    MAX_ACTOR_EPOCHS,
    MAX_GRAD_NORM,
    MINI_BATCHES,
    MINIMUM_STD,
    MOVING_KL_BETA,
    NUM_ENVS,
    POLICY_METHOD,
    PPO_CLIP,
    PROTOCOL_ID,
    RECOVERY_DISTANCE_M,
    RISER_HEIGHT_M,
    ROLLOUT_STEPS,
    ROUNDS,
    SMOKE_ENVS,
    SMOKE_SEED,
    SMOKE_STEPS,
    SOURCE_FILES,
    STD_SCALE_FROM_BASE,
    TARGET_KL,
    TASK_ID,
    TEACHER_CORRECTION_SCALE,
    TEACHER_HORIZON,
    TEACHER_WEIGHT,
    fixed_environment_parameters,
    formal_algorithm_parameters,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
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


def _validate_config(repo: Path, config_path: Path, checkpoint: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    implementation = config.get("implementation_boundary", {})
    commit = str(implementation.get("git_commit", ""))
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=repo
    ).returncode == 0
    relative = config_path.relative_to(repo)
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    checks = {
        "protocol_id": config.get("protocol_id") == PROTOCOL_ID,
        "status": config.get("status")
        == "fixed_before_v29_smoke_and_adaptation",
        "implementation_is_ancestor": ancestor,
        "config_committed_at_head": committed == config_path.read_bytes(),
        "base_checkpoint": config.get("base_checkpoint_sha256")
        == file_sha256(checkpoint)
        == BASE_CHECKPOINT_SHA256,
        "algorithm": config.get("training") == formal_algorithm_parameters(),
        "environment": config.get("environment") == fixed_environment_parameters(),
        "source_set": set(implementation.get("source_files", {}))
        == set(SOURCE_FILES),
        "source_hashes": all(
            (repo / relative_source).is_file()
            and file_sha256(repo / relative_source)
            == implementation.get("source_files", {}).get(relative_source)
            for relative_source in SOURCE_FILES
        ),
        "no_outcomes_before_freeze": config.get("prospective_execution", {}).get(
            "smoke_started"
        )
        is False
        and config.get("prospective_execution", {}).get("adaptation_started")
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v29 fixed config validation failed: {checks}")
    return {
        "path": str(config_path),
        "sha256": file_sha256(config_path),
        "implementation_commit": commit,
        "checks": checks,
    }


def _configure_algorithm(agent_cfg):
    from src.tasks.stairs_cbf.teacher_v29 import CbfTeacherV29PpoAlgorithmCfg

    agent_cfg.algorithm = CbfTeacherV29PpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=PPO_CLIP,
        entropy_coef=ENTROPY_COEFFICIENT,
        num_learning_epochs=MAX_ACTOR_EPOCHS,
        num_mini_batches=MINI_BATCHES,
        learning_rate=ACTOR_LEARNING_RATE,
        actor_learning_rate=ACTOR_LEARNING_RATE,
        critic_learning_rate=CRITIC_LEARNING_RATE,
        schedule="fixed",
        gamma=GAMMA,
        lam=GAE_LAMBDA,
        desired_kl=TARGET_KL,
        max_grad_norm=MAX_GRAD_NORM,
        normalize_advantage_per_mini_batch=False,
        std_scale_from_base=STD_SCALE_FROM_BASE,
        minimum_std=MINIMUM_STD,
        maximum_std=MAXIMUM_STD,
        moving_kl_beta=MOVING_KL_BETA,
        hard_kl_ceiling=HARD_KL_CEILING,
        critic_learning_epochs=CRITIC_EPOCHS,
        freeze_log_std=True,
        teacher_distillation_weight=TEACHER_WEIGHT,
        teacher_success_horizon=TEACHER_HORIZON,
        teacher_correction_scale=TEACHER_CORRECTION_SCALE,
    )


def _algorithm_audit(runner, env_cfg, shift: dict[str, Any]) -> dict[str, Any]:
    from src.tasks.stairs_cbf.teacher_v26 import HigherRiserCbfActionCfg
    from src.tasks.stairs_cbf.teacher_v29 import CbfTeacherV29PPO

    alg = runner.alg
    action_cfg = env_cfg.actions["joint_pos"]
    checks = {
        "v29_algorithm": isinstance(alg, CbfTeacherV29PPO),
        "cbf_action": isinstance(action_cfg, HigherRiserCbfActionCfg),
        "actor_observation_width": int(alg.actor.obs_dim),
        "privileged_critic_observation_width": int(alg.critic.obs_dim),
        "actor_observation_groups": list(alg.actor.obs_groups),
        "critic_observation_groups": list(alg.critic.obs_groups),
        "actor_count": 1,
        "critic_count": 1,
        "auxiliary_critics_absent": all(
            module is None
            for module in (alg.fall_critic, alg.intervention_critic, alg.risk_head)
        ),
        "runtime_filter_enabled": bool(action_cfg.enabled),
        "riser_height_m": float(shift["riser_height_m"]),
        "clearance_barrier_slope": float(shift["clearance_barrier_slope"]),
        "recovery_distance_m": float(shift["recovery_distance_m"]),
        "filter_alpha": float(shift["filter_alpha"]),
        "plant_action_transform": shift["plant_action_transform"],
        "actor_observation_fields_added": shift["actor_observation_fields_added"],
        "specialist_reward_absent": "specialist_failure_signal"
        not in env_cfg.rewards,
        "deployable_failure_observation_absent": "deployable_failure"
        not in env_cfg.observations,
        "failure_or_success_replay_disabled": not bool(
            alg.hard_case_policy_weight
            or alg.matched_success_preservation_beta
            or alg.correction_distillation_weight
        ),
        "whole_batch_advantage_normalization": not bool(
            alg.normalize_advantage_per_mini_batch
        ),
        "log_std_trainable_parameter_count": sum(
            parameter.requires_grad
            for parameter in alg.actor.distribution.parameters()
        ),
        "actor_learning_rate": float(alg.actor_learning_rate),
        "critic_learning_rate": float(alg.critic_learning_rate),
        "maximum_actor_epochs": int(alg.num_learning_epochs),
        "critic_epochs": int(alg.critic_learning_epochs),
        "minibatches": int(alg.num_mini_batches),
        "moving_kl_beta": float(alg.moving_kl_beta),
        "teacher_weight": float(alg.teacher_distillation_weight),
    }
    expected = {
        "v29_algorithm": True,
        "cbf_action": True,
        "actor_observation_width": 405,
        "privileged_critic_observation_width": 838,
        "actor_observation_groups": ["actor"],
        "critic_observation_groups": ["actor", "critic", "online_privileged"],
        "actor_count": 1,
        "critic_count": 1,
        "auxiliary_critics_absent": True,
        "runtime_filter_enabled": True,
        "riser_height_m": RISER_HEIGHT_M,
        "clearance_barrier_slope": CLEARANCE_BARRIER_SLOPE,
        "recovery_distance_m": RECOVERY_DISTANCE_M,
        "filter_alpha": FILTER_ALPHA,
        "plant_action_transform": "identity",
        "actor_observation_fields_added": 0,
        "specialist_reward_absent": True,
        "deployable_failure_observation_absent": True,
        "failure_or_success_replay_disabled": True,
        "whole_batch_advantage_normalization": True,
        "log_std_trainable_parameter_count": 0,
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "maximum_actor_epochs": MAX_ACTOR_EPOCHS,
        "critic_epochs": CRITIC_EPOCHS,
        "minibatches": MINI_BATCHES,
        "moving_kl_beta": MOVING_KL_BETA,
        "teacher_weight": TEACHER_WEIGHT,
    }
    mismatches = {
        key: {"actual": checks[key], "expected": value}
        for key, value in expected.items()
        if checks[key] != value
    }
    if mismatches:
        raise RuntimeError(f"v29 structural audit failed: {mismatches}")
    checks["reward_terms"] = sorted(env_cfg.rewards)
    checks["termination_terms"] = sorted(env_cfg.terminations)
    return checks


def _save_checkpoint(runner, path: Path, iteration: int, metadata: dict[str, Any]) -> None:
    payload = runner.alg.save()
    payload["iter"] = iteration
    payload["infos"] = {"cbf_teacher_v29": metadata}
    payload["python_random_state"] = random.getstate()
    payload["numpy_random_state"] = np.random.get_state()
    payload["torch_random_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_round_csv(path: Path, rounds: list[dict[str, Any]]) -> None:
    fields = (
        "round",
        "status",
        "rollout_success_rate",
        "rollout_fall_rate",
        "rollout_mean_return",
        "rollout_mean_reached_riser",
        "actor_loss",
        "critic_loss",
        "moving_forward_kl",
        "clip_fraction",
        "actor_gradient_norm",
        "critic_gradient_norm",
        "actor_epochs_completed",
        "cbf_intervention_count",
        "cbf_intervention_per_riser",
        "mean_correction_norm",
        "toe_riser_kick_count",
        "teacher_eligible_count",
        "teacher_eligible_fraction_among_interventions",
        "teacher_weighted_count",
        "teacher_loss",
        "mean_policy_to_teacher_action_distance",
        "mean_weighted_policy_to_teacher_action_distance",
        "rollback_reason",
        "round_start_actor_sha256",
        "round_end_actor_sha256",
    )
    rows = []
    for record in rounds:
        metrics = record["metrics"]
        rows.append(
            {
                "round": record["round"],
                "status": record["status"],
                "rollout_success_rate": metrics.get("rollout_success_rate"),
                "rollout_fall_rate": metrics.get("rollout_fall_rate"),
                "rollout_mean_return": metrics.get("rollout_mean_return"),
                "rollout_mean_reached_riser": metrics.get(
                    "rollout_mean_reached_riser"
                ),
                "actor_loss": metrics.get("actor_loss"),
                "critic_loss": metrics.get("value"),
                "moving_forward_kl": metrics.get("moving_forward_kl"),
                "clip_fraction": metrics.get("clip_fraction"),
                "actor_gradient_norm": metrics.get(
                    "actor_gradient_norm_pre_clip_max"
                ),
                "critic_gradient_norm": metrics.get(
                    "critic_gradient_norm_pre_clip_max"
                ),
                "actor_epochs_completed": metrics.get("actor_epochs_completed"),
                "cbf_intervention_count": metrics.get("cbf_intervention_count"),
                "cbf_intervention_per_riser": metrics.get(
                    "cbf_intervention_per_riser"
                ),
                "mean_correction_norm": metrics.get("cbf_correction_mean"),
                "toe_riser_kick_count": metrics.get(
                    "toe_riser_kick_event_count"
                ),
                "teacher_eligible_count": metrics.get("teacher_eligible_count"),
                "teacher_eligible_fraction_among_interventions": metrics.get(
                    "teacher_eligible_fraction_among_interventions"
                ),
                "teacher_weighted_count": metrics.get("teacher_weighted_count"),
                "teacher_loss": metrics.get("teacher_loss"),
                "mean_policy_to_teacher_action_distance": metrics.get(
                    "mean_policy_to_teacher_action_distance"
                ),
                "mean_weighted_policy_to_teacher_action_distance": metrics.get(
                    "mean_weighted_policy_to_teacher_action_distance"
                ),
                "rollback_reason": record.get("rollback_reason"),
                "round_start_actor_sha256": record["round_start_actor_sha256"],
                "round_end_actor_sha256": record["round_end_actor_sha256"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _collect_one_round(runner) -> dict[str, Any]:
    from rsl_rl.utils import check_nan
    from src.tasks.stairs_cbf.teacher import ProximalHardRollback

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
            next_obs, rewards, dones, extras = runner.env.step(
                raw_actions.to(runner.env.device)
            )
            check_nan(next_obs, rewards, dones)
            extras = dict(extras)
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
            "normal_on_policy_initial_states_only": True,
            "performance_gate_used": False,
        }
        try:
            action_metrics = runner.alg.relabel_teacher_transitions()
        except RuntimeError as error:
            message = str(error)
            routing_tokens = (
                "PPO storage",
                "runtime action",
                "teacher",
                "swing-foot",
                "episode identity",
            )
            if any(token in message for token in routing_tokens):
                raise ProximalHardRollback(
                    "v29 action/storage/teacher routing error",
                    {**rollout, "exception": message},
                ) from error
            raise
        runner.alg.compute_returns(obs)
    try:
        update = runner.alg.update()
    except RuntimeError as error:
        if isinstance(error, ProximalHardRollback):
            raise ProximalHardRollback(error.reason, {**rollout, **error.metrics})
        message = str(error)
        if "behavior" in message or "a_policy" in message:
            raise ProximalHardRollback(
                "v29 raw-action behavior routing error",
                {**rollout, "exception": message},
            ) from error
        raise
    update.update(action_metrics)
    update.update(rollout)
    update["cbf_intervention_count"] = action_metrics[
        "cbf_intervention_fraction"
    ] * (n * runner.cfg["num_steps_per_env"])
    return update


def _smoke_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "teacher_eligible_count_nonzero": metrics["teacher_eligible_count"] > 0,
        "teacher_loss_finite": math.isfinite(float(metrics["teacher_loss"])),
        "teacher_transform_error_below_1e_minus_6": metrics[
            "teacher_reprojection_max_abs_error"
        ]
        < 1.0e-6,
        "ppo_storage_is_raw_action": metrics["policy_storage_max_abs_error"]
        < 1.0e-6,
        "executed_action_is_cbf_safe": metrics[
            "executed_action_routing_max_abs_error"
        ]
        < 1.0e-5,
        "episode_identity_routing_valid": metrics[
            "episode_identity_transition_max_abs_error"
        ]
        == 0.0,
        "actor_backward_and_step_completed": metrics[
            "actor_minibatches_completed"
        ]
        >= 1,
        "critic_backward_and_step_completed": metrics[
            "critic_minibatches_completed"
        ]
        >= 1,
    }


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v29 execution requires a clean committed worktree")
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError("v29 checkpoint or fixed config is missing")
    config_reference = _validate_config(repo, config_path, checkpoint)
    if args.smoke:
        seed, rounds, num_envs, rollout_steps = (
            SMOKE_SEED,
            1,
            SMOKE_ENVS,
            SMOKE_STEPS,
        )
        marker_name = "smoke_execution_started.json"
    else:
        seed, rounds, num_envs, rollout_steps = (
            ADAPTATION_SEED,
            ROUNDS,
            NUM_ENVS,
            ROLLOUT_STEPS,
        )
        marker_name = "formal_execution_started.json"
    marker = output_dir / marker_name
    if marker.exists():
        raise RuntimeError(f"v29 execution already started: {marker}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        marker,
        {
            "protocol_id": PROTOCOL_ID,
            "config": config_reference,
            "smoke": args.smoke,
            "seed": seed,
            "rounds": rounds,
            "num_envs": num_envs,
            "rollout_steps": rollout_steps,
            "performance_selection": False,
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
    from src.tasks.stairs_cbf.teacher import ProximalHardRollback
    from src.tasks.stairs_cbf.teacher_v26 import configure_v26_higher_riser
    from src.tasks.stairs_cbf.teacher_v29 import CbfTeacherV29Runner

    env_cfg = load_env_cfg(TASK_ID, play=True)
    shift = configure_v26_higher_riser(
        env_cfg,
        riser_height_m=RISER_HEIGHT_M,
        runtime_filter=True,
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(TASK_ID)
    agent_cfg.seed = seed
    agent_cfg.num_steps_per_env = rollout_steps
    _configure_algorithm(agent_cfg)
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = CbfTeacherV29Runner(env, asdict(agent_cfg), log_dir=None, device=args.device)
    try:
        structural_audit = _algorithm_audit(runner, env_cfg, shift)
        warm_start = runner.load_initial_checkpoint(
            str(checkpoint), map_location=args.device
        )
        initial_actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
        if not args.smoke:
            torch.save(
                {
                    "actor_state_dict": actor_state(runner.alg.actor),
                    "actor_sha256": initial_actor_hash,
                    "source_checkpoint_sha256": file_sha256(checkpoint),
                },
                output_dir / "base_actor.pt",
            )
        records = []
        for round_index in range(1, rounds + 1):
            runner.alg.freeze_round_reference()
            transaction = runner.snapshot_proximal_state()
            start_hash = actor_state_sha256(actor_state(runner.alg.actor))
            _save_checkpoint(
                runner,
                output_dir / "current_round_start.pt",
                round_index - 1,
                {"round": round_index, "boundary": "start", "actor": start_hash},
            )
            status = "updated"
            rollback_reason = None
            try:
                metrics = _collect_one_round(runner)
            except ProximalHardRollback as rollback:
                runner.restore_proximal_state(transaction)
                runner.alg.storage.clear()
                runner.alg.clear_cbf_rollout()
                runner.alg.last_update_metrics = {}
                status = "hard_rollback"
                rollback_reason = rollback.reason
                metrics = {**rollback.metrics, "hard_rollback": True}
            end_hash = actor_state_sha256(actor_state(runner.alg.actor))
            record = {
                "round": round_index,
                "status": status,
                "rollback_reason": rollback_reason,
                "round_start_actor_sha256": start_hash,
                "round_end_actor_sha256": end_hash,
                "round_reference_is_exact_pi_k": True,
                "performance_evaluation_or_gate_used": False,
                "metrics": metrics,
            }
            records.append(record)
            _atomic_json(output_dir / "round_metrics.json", records)
            _write_round_csv(output_dir / "round_metrics.csv", records)
            print(json.dumps(record, sort_keys=True), flush=True)

        final_checkpoint = output_dir / f"final_round_{rounds:02d}.pt"
        _save_checkpoint(
            runner,
            final_checkpoint,
            rounds,
            {
                "round": rounds,
                "boundary": "final",
                "selection": "unconditional final round",
            },
        )
        final_actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
        common = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "policy_method": POLICY_METHOD,
            "smoke": args.smoke,
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "config": config_reference,
            "base_checkpoint_sha256": file_sha256(checkpoint),
            "seed": seed,
            "environment": fixed_environment_parameters(),
            "training": formal_algorithm_parameters(),
            "structural_audit": structural_audit,
            "warm_start": warm_start,
            "initial_actor_sha256": initial_actor_hash,
            "final_actor_sha256": final_actor_hash,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "final_policy_rule": (
                "round 8 actor, never best checkpoint"
                if not args.smoke
                else "single smoke update only"
            ),
            "rounds": records,
            "performance_rollbacks": 0,
            "candidate_line_search_count": 0,
            "failure_replay_bank_count": 0,
            "adaptation_seed_count": 1,
        }
        if args.smoke:
            checks = _smoke_checks(records[0]["metrics"])
            summary = {**common, "checks": checks, "passed": all(checks.values())}
            _atomic_json(output_dir / "smoke_summary.json", summary)
            if not summary["passed"]:
                raise RuntimeError(f"v29 functional smoke failed: {checks}")
        else:
            summary = common
            _atomic_json(output_dir / "training_summary.json", summary)
            _atomic_json(
                output_dir / "formal_execution_completed.json",
                {
                    "protocol_id": PROTOCOL_ID,
                    "rounds_completed": rounds,
                    "final_actor_sha256": final_actor_hash,
                    "training_summary_sha256": file_sha256(
                        output_dir / "training_summary.json"
                    ),
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

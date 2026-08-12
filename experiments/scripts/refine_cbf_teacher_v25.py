"""Run the sole fixed-eight-round success-gated CBF-teacher adaptation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cbf_teacher_v25_protocol import (
    ACTOR_LEARNING_RATE,
    ADAPTATION_SEED,
    BASE_CHECKPOINT_SHA256,
    CRITIC_EPOCHS,
    CRITIC_LEARNING_RATE,
    ENTROPY_COEFFICIENT,
    GAE_LAMBDA,
    GAMMA,
    HARD_KL_CEILING,
    MAX_ACTOR_EPOCHS,
    MAX_GRAD_NORM,
    MINI_BATCHES,
    MOVING_KL_BETA,
    NUM_ENVS,
    POLICY_METHOD,
    PPO_CLIP,
    PROTOCOL_ID,
    ROLLOUT_STEPS,
    ROUNDS,
    SOURCE_FILES,
    TARGET_KL,
    TEACHER_CORRECTION_SCALE,
    TEACHER_DISTILLATION_WEIGHT,
    TEACHER_SUCCESS_HORIZON,
    formal_algorithm_parameters,
    validate_v25_calibrated_context,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=ADAPTATION_SEED)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Permit reduced sizes and omit the prospective formal protocol.",
    )
    return parser.parse_args()


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_frozen_protocol(
    repo: Path,
    protocol_path: Path,
    *,
    checkpoint: Path,
    context_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    context = validate_v25_calibrated_context(json.loads(context_path.read_text()))
    implementation = protocol.get("implementation_boundary", {})
    implementation_commit = str(implementation.get("git_commit", ""))
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
            cwd=repo,
            check=False,
        ).returncode
        == 0
    )
    checks = {
        "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
        "method": protocol.get("policy_method") == POLICY_METHOD,
        "implementation_is_ancestor": ancestor,
        "randomness_preflight": protocol.get("randomness_preflight", {}).get("passed")
        is True,
        "base_checkpoint": protocol.get("base_checkpoint", {}).get("sha256")
        == file_sha256(checkpoint)
        == BASE_CHECKPOINT_SHA256,
        "context_file": protocol.get("context", {}).get("file_sha256")
        == file_sha256(context_path),
        "context_parameters": protocol.get("context", {}).get("parameters_sha256")
        == context["parameters_sha256"],
        "algorithm": protocol.get("training") == formal_algorithm_parameters(),
        "execution_not_started_at_freeze": protocol.get(
            "prospective_execution", {}
        ).get("adapted_policy_outcomes_observed")
        is False,
        "all_bound_sources_unchanged": set(implementation.get("source_files", {}))
        == set(SOURCE_FILES)
        and all(
            (repo / relative).is_file()
            and file_sha256(repo / relative)
            == implementation.get("source_files", {}).get(relative)
            for relative in SOURCE_FILES
        ),
    }
    for name, path in (("protocol", protocol_path), ("context", context_path)):
        relative = path.relative_to(repo)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        checks[f"{name}_committed_at_head"] = committed == path.read_bytes()
    if not all(checks.values()):
        raise RuntimeError(f"v25 frozen protocol validation failed: {checks}")
    return {
        "file": str(protocol_path),
        "sha256": file_sha256(protocol_path),
        "implementation_commit": implementation_commit,
        "validation": checks,
    }


def _configure_algorithm(agent_cfg):
    from src.tasks.stairs_cbf.teacher import CbfTeacherPpoAlgorithmCfg

    agent_cfg.algorithm = CbfTeacherPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=PPO_CLIP,
        entropy_coef=ENTROPY_COEFFICIENT,
        num_learning_epochs=MAX_ACTOR_EPOCHS,
        num_mini_batches=MINI_BATCHES,
        learning_rate=ACTOR_LEARNING_RATE,
        schedule="fixed",
        gamma=GAMMA,
        lam=GAE_LAMBDA,
        desired_kl=TARGET_KL,
        max_grad_norm=MAX_GRAD_NORM,
        normalize_advantage_per_mini_batch=False,
        teacher_distillation_weight=TEACHER_DISTILLATION_WEIGHT,
        teacher_success_horizon=TEACHER_SUCCESS_HORIZON,
        teacher_correction_scale=TEACHER_CORRECTION_SCALE,
    )
    return agent_cfg.algorithm


def _algorithm_audit(runner, env_cfg, shift_metadata) -> dict[str, Any]:
    from src.tasks.stairs_cbf.teacher import (
        CbfTeacherPPO,
        SwingUnderResponseCbfActionCfg,
    )

    alg = runner.alg
    action_cfg = env_cfg.actions["joint_pos"]
    actor_optimizer_ids = {
        id(parameter)
        for group in alg.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    critic_optimizer_ids = {
        id(parameter)
        for group in alg.critic_optimizer.param_groups
        for parameter in group["params"]
    }
    checks = {
        "algorithm_class": isinstance(alg, CbfTeacherPPO),
        "action_config_class": isinstance(action_cfg, SwingUnderResponseCbfActionCfg),
        "actor_observation_dim": int(alg.actor.obs_dim),
        "critic_observation_dim": int(alg.critic.obs_dim),
        "actor_observation_groups": list(alg.actor.obs_groups),
        "critic_observation_groups": list(alg.critic.obs_groups),
        "deployable_failure_group_absent": "deployable_failure"
        not in env_cfg.observations,
        "one_actor": isinstance(alg.actor, torch.nn.Module),
        "one_privileged_critic": isinstance(alg.critic, torch.nn.Module),
        "auxiliary_critics_absent": all(
            module is None
            for module in (alg.fall_critic, alg.intervention_critic, alg.risk_head)
        ),
        "specialist_reward_absent": "specialist_failure_signal" not in env_cfg.rewards,
        "runtime_filter": action_cfg.enabled,
        "phase_selective_shift": shift_metadata["shift"]
        == "fixed_phase_selective_swing_leg_underresponse",
        "actor_critic_optimizers_disjoint": not bool(
            actor_optimizer_ids & critic_optimizer_ids
        ),
        "log_std_trainable_parameter_count": sum(
            parameter.requires_grad for parameter in alg.actor.distribution.parameters()
        ),
        "actor_learning_rate": alg.actor_learning_rate,
        "critic_learning_rate": alg.critic_learning_rate,
        "ppo_clip": alg.clip_param,
        "maximum_actor_epochs": alg.num_learning_epochs,
        "critic_epochs": alg.critic_learning_epochs,
        "mini_batches": alg.num_mini_batches,
        "moving_kl_beta": alg.moving_kl_beta,
        "target_kl": alg.desired_kl,
        "hard_kl_ceiling": alg.hard_kl_ceiling,
        "teacher_distillation_weight": alg.teacher_distillation_weight,
        "teacher_success_horizon": alg.teacher_success_horizon,
        "teacher_correction_scale": alg.teacher_correction_scale,
    }
    expected = {
        "algorithm_class": True,
        "action_config_class": True,
        "actor_observation_dim": 405,
        "critic_observation_dim": 838,
        "actor_observation_groups": ["actor"],
        "critic_observation_groups": ["actor", "critic", "online_privileged"],
        "deployable_failure_group_absent": True,
        "one_actor": True,
        "one_privileged_critic": True,
        "auxiliary_critics_absent": True,
        "specialist_reward_absent": True,
        "runtime_filter": True,
        "phase_selective_shift": True,
        "actor_critic_optimizers_disjoint": True,
        "log_std_trainable_parameter_count": 0,
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "ppo_clip": PPO_CLIP,
        "maximum_actor_epochs": MAX_ACTOR_EPOCHS,
        "critic_epochs": CRITIC_EPOCHS,
        "mini_batches": MINI_BATCHES,
        "moving_kl_beta": MOVING_KL_BETA,
        "target_kl": TARGET_KL,
        "hard_kl_ceiling": HARD_KL_CEILING,
        "teacher_distillation_weight": TEACHER_DISTILLATION_WEIGHT,
        "teacher_success_horizon": TEACHER_SUCCESS_HORIZON,
        "teacher_correction_scale": TEACHER_CORRECTION_SCALE,
    }
    mismatches = {
        key: {"actual": checks[key], "expected": expected_value}
        for key, expected_value in expected.items()
        if checks[key] != expected_value
    }
    if mismatches:
        raise RuntimeError(f"v25 structural invariant failed: {mismatches}")
    return checks


def _save_checkpoint(
    runner,
    path: Path,
    *,
    iteration: int,
    metadata: dict[str, Any],
) -> None:
    payload = runner.alg.save()
    payload["iter"] = iteration
    payload["infos"] = {"cbf_teacher_v25": metadata}
    payload["python_random_state"] = random.getstate()
    payload["numpy_random_state"] = np.random.get_state()
    payload["torch_random_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _write_round_csv(path: Path, rounds: list[dict[str, Any]]) -> None:
    fields = (
        "round",
        "status",
        "rollout_episode_count",
        "rollout_success_rate",
        "rollout_fall_rate",
        "rollout_kick_episode_rate",
        "actor_loss",
        "ppo_loss",
        "teacher_loss",
        "teacher_transition_count",
        "teacher_transition_fraction",
        "teacher_weight_mean_over_valid",
        "moving_forward_kl",
        "cbf_intervention_per_riser",
        "toe_riser_kick_event_count",
        "teacher_reprojection_max_abs_error",
        "actor_epochs_completed",
        "target_kl_early_stopped",
        "rollback_reason",
        "round_start_actor_sha256",
        "round_end_actor_sha256",
    )
    rows = []
    for record in rounds:
        metrics = record.get("metrics", {})
        rows.append(
            {
                "round": record["round"],
                "status": record["status"],
                "rollout_episode_count": metrics.get("rollout_episode_count"),
                "rollout_success_rate": metrics.get("rollout_success_rate"),
                "rollout_fall_rate": metrics.get("rollout_fall_rate"),
                "rollout_kick_episode_rate": metrics.get("rollout_kick_episode_rate"),
                "actor_loss": metrics.get("actor_loss"),
                "ppo_loss": metrics.get("surrogate"),
                "teacher_loss": metrics.get("teacher_loss"),
                "teacher_transition_count": metrics.get("teacher_transition_count"),
                "teacher_transition_fraction": metrics.get(
                    "teacher_transition_fraction"
                ),
                "teacher_weight_mean_over_valid": metrics.get(
                    "teacher_weight_mean_over_valid"
                ),
                "moving_forward_kl": metrics.get("moving_forward_kl"),
                "cbf_intervention_per_riser": metrics.get("cbf_intervention_per_riser"),
                "toe_riser_kick_event_count": metrics.get("toe_riser_kick_event_count"),
                "teacher_reprojection_max_abs_error": metrics.get(
                    "teacher_reprojection_max_abs_error"
                ),
                "actor_epochs_completed": metrics.get("actor_epochs_completed"),
                "target_kl_early_stopped": metrics.get("target_kl_early_stopped"),
                "rollback_reason": record.get("rollback_reason"),
                "round_start_actor_sha256": record["round_start_actor_sha256"],
                "round_end_actor_sha256": record["round_end_actor_sha256"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
    episode_returns = torch.zeros(runner.env.num_envs, device=runner.env.device)
    episode_had_intervention = torch.zeros(
        runner.env.num_envs, dtype=torch.bool, device=runner.env.device
    )
    episode_had_kick = torch.zeros_like(episode_had_intervention)
    completed_returns: list[float] = []
    episode_count = 0
    success_count = 0
    fall_count = 0
    timeout_count = 0
    recovery_takeover_count = 0
    kick_episode_count = 0
    reward_sum = 0.0
    with torch.no_grad():
        for _ in range(runner.cfg["num_steps_per_env"]):
            actions = runner.alg.act(obs)
            next_obs, rewards, dones, extras = runner.env.step(
                actions.to(runner.env.device)
            )
            check_nan(next_obs, rewards, dones)
            extras = dict(extras)
            episode_returns += rewards
            reward_sum += float(rewards.sum())
            intervened = extras["cbf_intervened"].bool()
            kick = extras["v25_toe_riser_kick"].bool()
            episode_had_intervention |= intervened
            episode_had_kick |= kick
            done_mask = dones.bool()
            if bool(done_mask.any()):
                fell = extras["online_fell"].bool()
                timeouts = extras.get(
                    "time_outs", torch.zeros_like(done_mask, dtype=torch.bool)
                ).bool()
                success = runner.env.unwrapped.termination_manager.get_term(
                    "reached_top"
                ).bool()
                done_ids = done_mask.nonzero(as_tuple=False).flatten()
                completed_returns.extend(
                    float(episode_returns[env_id]) for env_id in done_ids.tolist()
                )
                episode_count += len(done_ids)
                success_count += int((done_mask & success).sum())
                fall_count += int((done_mask & fell).sum())
                timeout_count += int((done_mask & timeouts).sum())
                recovery_takeover_count += int(
                    (done_mask & success & episode_had_intervention).sum()
                )
                kick_episode_count += int((done_mask & episode_had_kick).sum())
                episode_returns[done_ids] = 0.0
                episode_had_intervention[done_ids] = False
                episode_had_kick[done_ids] = False
            obs = next_obs.to(runner.device)
            runner.alg.process_env_step(
                obs,
                rewards.to(runner.device),
                dones.to(runner.device),
                extras,
            )
        rollout_metrics = {
            "rollout_episode_count": episode_count,
            "rollout_success_count": success_count,
            "rollout_fall_count": fall_count,
            "rollout_timeout_count": timeout_count,
            "rollout_success_rate": success_count / max(1, episode_count),
            "rollout_fall_rate": fall_count / max(1, episode_count),
            "rollout_kick_episode_count": kick_episode_count,
            "rollout_kick_episode_rate": kick_episode_count / max(1, episode_count),
            "rollout_mean_return": (
                sum(completed_returns) / len(completed_returns)
                if completed_returns
                else None
            ),
            "rollout_mean_reward_per_transition": reward_sum
            / (runner.env.num_envs * runner.cfg["num_steps_per_env"]),
            "recovery_takeover_count": recovery_takeover_count,
            "recovery_takeover_rate": recovery_takeover_count / max(1, episode_count),
            "ordinary_cbf_intervention_is_teacher": False,
            "normal_physical_initial_states_only": True,
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
            )
            if any(token in message for token in routing_tokens):
                raise ProximalHardRollback(
                    "v25 action/teacher routing audit failed",
                    {**rollout_metrics, "exception": message},
                ) from error
            raise
        runner.alg.compute_returns(obs)

    try:
        update_metrics = runner.alg.update()
    except RuntimeError as error:
        if isinstance(error, ProximalHardRollback):
            raise ProximalHardRollback(
                error.reason, {**rollout_metrics, **error.metrics}
            ) from error
        message = str(error)
        if "behavior" in message or "a_policy" in message:
            raise ProximalHardRollback(
                "behavior Gaussian/log-prob routing audit failed",
                {**rollout_metrics, "exception": message},
            ) from error
        raise
    update_metrics.update(action_metrics)
    update_metrics.update(rollout_metrics)
    return update_metrics


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    context_path = args.context.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file() or not context_path.is_file():
        raise FileNotFoundError("base checkpoint or v25 context does not exist")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v25 base checkpoint differs from frozen pi0")
    if not args.smoke:
        formal = {
            "seed": (args.seed, ADAPTATION_SEED),
            "rounds": (args.rounds, ROUNDS),
            "num_envs": (args.num_envs, NUM_ENVS),
            "rollout_steps": (args.rollout_steps, ROLLOUT_STEPS),
        }
        mismatches = {
            key: {"actual": actual, "required": required}
            for key, (actual, required) in formal.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(f"formal v25 execution mismatch: {mismatches}")
        if args.protocol is None:
            raise ValueError("formal v25 execution requires a frozen protocol")
        if (output_dir / "formal_execution_started.json").exists():
            raise RuntimeError("formal v25 adaptation has already been started")
    elif min(args.rounds, args.num_envs, args.rollout_steps) < 1:
        raise ValueError("smoke rollout sizes must be positive")

    context = validate_v25_calibrated_context(json.loads(context_path.read_text()))
    protocol_reference = (
        _validate_frozen_protocol(
            repo,
            args.protocol.resolve(),
            checkpoint=checkpoint,
            context_path=context_path,
        )
        if not args.smoke and args.protocol is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        _write_json(
            output_dir / "formal_execution_started.json",
            {
                "protocol": protocol_reference,
                "adapted_policy_outcomes_observed": False,
                "fresh_adaptation_count": 1,
            },
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(repo))
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import src.tasks  # noqa: F401
    from src.tasks.stairs_cbf.teacher import (
        CbfTeacherRefinementRunner,
        ProximalHardRollback,
        configure_v25_swing_underresponse,
    )

    gain = float(context["shift"]["swing_underresponse_gain"])
    env_cfg = load_env_cfg("Unitree-G1-Stairs-Online-DQHMED")
    shift_metadata = configure_v25_swing_underresponse(
        env_cfg, gain=gain, runtime_filter=True
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = load_rl_cfg("Unitree-G1-Stairs-Online-DQHMED")
    agent_cfg.seed = args.seed
    agent_cfg.num_steps_per_env = args.rollout_steps
    _configure_algorithm(agent_cfg)
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = CbfTeacherRefinementRunner(
        env, asdict(agent_cfg), log_dir=None, device=args.device
    )
    try:
        structural_audit = _algorithm_audit(runner, env_cfg, shift_metadata)
        warm_start = runner.load_initial_checkpoint(
            str(checkpoint), map_location=args.device
        )
        if (
            warm_start["actor_observation_dim"] != 405
            or warm_start["critic_observation_dim"] != 838
        ):
            raise RuntimeError("v25 warm start changed the observation interface")
        initial_actor_sha = actor_state_sha256(actor_state(runner.alg.actor))
        rounds: list[dict[str, Any]] = []
        for round_index in range(1, args.rounds + 1):
            runner.alg.freeze_round_reference()
            transaction = runner.snapshot_proximal_state()
            round_start_sha = actor_state_sha256(actor_state(runner.alg.actor))
            _save_checkpoint(
                runner,
                output_dir / "checkpoints" / f"round_{round_index:02d}_start.pt",
                iteration=round_index - 1,
                metadata={
                    "round": round_index,
                    "boundary": "start",
                    "actor_sha256": round_start_sha,
                },
            )
            status = "updated"
            rollback_reason = None
            metrics: dict[str, Any]
            try:
                metrics = _collect_one_round(runner)
            except ProximalHardRollback as rollback:
                runner.restore_proximal_state(transaction)
                runner.alg.storage.clear()
                runner.alg.clear_cbf_rollout()
                runner.alg.last_update_metrics = {}
                status = "hard_rollback"
                rollback_reason = rollback.reason
                metrics = dict(rollback.metrics)
                metrics.update(
                    {"hard_rollback": True, "hard_rollback_reason": rollback.reason}
                )
            round_end_sha = actor_state_sha256(actor_state(runner.alg.actor))
            record = {
                "round": round_index,
                "status": status,
                "rollback_reason": rollback_reason,
                "round_start_actor_sha256": round_start_sha,
                "round_end_actor_sha256": round_end_sha,
                "round_reference_is_moving_pi_k": True,
                "performance_evaluation_or_gate_used": False,
                "metrics": metrics,
            }
            rounds.append(record)
            _save_checkpoint(
                runner,
                output_dir / "checkpoints" / f"round_{round_index:02d}_end.pt",
                iteration=round_index,
                metadata=record,
            )
            _write_json(output_dir / "round_metrics.json", rounds)
            _write_round_csv(output_dir / "round_metrics.csv", rounds)
            print(json.dumps(record, sort_keys=True), flush=True)

        final_checkpoint = output_dir / f"final_round_{args.rounds:02d}.pt"
        _save_checkpoint(
            runner,
            final_checkpoint,
            iteration=args.rounds,
            metadata={
                "round": args.rounds,
                "boundary": "final",
                "selection": "fixed final round; no validation or performance selection",
            },
        )
        final_actor_sha = actor_state_sha256(actor_state(runner.alg.actor))
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "policy_method": POLICY_METHOD,
            "experiment_class": "single fixed CBF-aligned shift development test",
            "smoke": args.smoke,
            "git_commit": _git_output(repo, "rev-parse", "HEAD"),
            "protocol": protocol_reference,
            "base_checkpoint": {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
            },
            "context": {
                "path": str(context_path),
                "file_sha256": file_sha256(context_path),
                "parameters_sha256": context["parameters_sha256"],
                "selected_candidate_index": context["shift"][
                    "selected_candidate_index"
                ],
                "swing_underresponse_gain": gain,
                "base_policy_only_first_qualifier": True,
                "reused_without_reselection": True,
                "metadata": shift_metadata,
            },
            "adaptation_seed": args.seed,
            "training": formal_algorithm_parameters(),
            "warm_start": warm_start,
            "structural_audit": structural_audit,
            "initial_actor_sha256": initial_actor_sha,
            "final_actor_sha256": final_actor_sha,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "final_policy_rule": (
                "round 8 actor, never best-so-far"
                if not args.smoke
                else f"round {args.rounds} actor, smoke only"
            ),
            "rounds": rounds,
            "hard_rollback_count": sum(
                record["status"] == "hard_rollback" for record in rounds
            ),
            "performance_rollbacks": 0,
            "candidate_screen_or_confirmation_count": 0,
            "state_restart_count": 0,
            "failure_or_success_bank_count": 0,
            "adaptation_seed_count": 1,
        }
        _write_json(output_dir / "training_summary.json", summary)
        if not args.smoke:
            _write_json(
                output_dir / "formal_execution_completed.json",
                {
                    "protocol": protocol_reference,
                    "adapted_policy_outcomes_observed": True,
                    "fresh_adaptation_count": 1,
                    "final_actor_sha256": final_actor_sha,
                    "training_summary_sha256": file_sha256(
                        output_dir / "training_summary.json"
                    ),
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()

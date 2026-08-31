"""Train one v141 F2/F3 specialist from the frozen v139 checkpoint."""

from __future__ import annotations

import argparse
import json
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
    CLEARANCE_BARRIER_SLOPE,
    FILTER_ALPHA,
    RECOVERY_DISTANCE_M,
    environment_parameters,
)
from filter_free_v141_protocol import (
    BASE_ACTOR_LEARNING_RATE,
    BASE_CHECKPOINT_SHA256,
    BASE_CRITIC_LEARNING_RATE,
    DEFAULT_MOVING_KL_BETA,
    DEFAULT_TARGET_FRACTION,
    GENERATION_1_ROUNDS,
    METHOD_ID,
    NUM_ENVS,
    RETENTION_CONTEXT,
    ROLLOUT_STEPS,
    SPECIALISTS,
    TASK_ID,
    TRAINING_ACTION_STD,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--specialist", choices=SPECIALISTS, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=GENERATION_1_ROUNDS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--target-fraction", type=float, default=DEFAULT_TARGET_FRACTION)
    parser.add_argument("--intervention-ppo-eta", type=float, required=True)
    parser.add_argument(
        "--correction-weight-mode",
        choices=(
            "intervention_only",
            "positive_advantage",
            "episode_success_positive_advantage",
        ),
        required=True,
    )
    parser.add_argument(
        "--correction-loss-weight",
        type=float,
        choices=(0.05, 0.1, 0.2, 0.4),
        required=True,
    )
    parser.add_argument(
        "--dual-reward-scale", type=float, choices=(0.0, 0.25, 1.0), required=True
    )
    parser.add_argument("--actor-learning-rate", type=float, default=BASE_ACTOR_LEARNING_RATE)
    parser.add_argument("--critic-learning-rate", type=float, default=BASE_CRITIC_LEARNING_RATE)
    parser.add_argument("--moving-kl-beta", type=float, default=DEFAULT_MOVING_KL_BETA)
    parser.add_argument("--actor-epochs", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument(
        "--exploration-std", type=float, choices=(0.03, 0.05), default=0.05
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _save_checkpoint(runner, path: Path, round_index: int, metadata: dict[str, Any]) -> None:
    payload = runner.alg.save()
    payload["iter"] = round_index
    payload["infos"] = {"filter_free_v141": metadata}
    payload["python_random_state"] = random.getstate()
    payload["numpy_random_state"] = np.random.get_state()
    payload["torch_random_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    # RNG APIs require CPU ByteTensors.  Checkpoint model tensors may be mapped
    # directly to CUDA for recovery, but RNG state must never follow that map.
    torch_state = payload["torch_random_state"].detach().to(
        device="cpu", dtype=torch.uint8
    )
    torch.set_rng_state(torch_state.contiguous())
    if torch.cuda.is_available() and "torch_cuda_random_state_all" in payload:
        cuda_states = [
            state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            for state in payload["torch_cuda_random_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _configure_context_env(
    *, context: str, num_envs: int, seed: int, device: str, dual_reward_scale: float
):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
    from src.tasks.stairs_cbf.paper_clearance_margin_v138 import (
        configure_paper_clearance_margin,
    )
    from src.tasks.stairs_cbf.paper_dual_v35 import configure_paper_dual_reward
    from src.tasks.stairs_cbf.velocity_cbf_action import (
        CURRENT_CBF_MODE,
        configure_v34_cbf,
    )

    env_cfg = load_env_cfg(TASK_ID, play=True)
    shift = configure_v31_context(
        env_cfg,
        context=context,
        runtime_filter=True,
        context_spec=environment_parameters(context),
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    cbf = configure_v34_cbf(
        env_cfg,
        mode=CURRENT_CBF_MODE,
        runtime_filter=True,
        parameters=None,
        measure_compute_time=False,
    )
    reward = configure_paper_dual_reward(
        env_cfg,
        "paper_stair_sloped_unit_balanced",
        runtime_filter_during_training=True,
    )
    env_cfg.rewards["cbf_dual"].weight = float(dual_reward_scale)
    clearance = env_cfg.rewards["foot_clearance"]
    clearance.params = {
        **clearance.params,
        "reference_mode": "next_riser",
        "lookahead_distance": 0.60,
    }
    clearance_metadata = configure_paper_clearance_margin(env_cfg)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    base = ManagerBasedRlEnv(env_cfg, device=device)
    agent_cfg = load_rl_cfg(TASK_ID)
    wrapper = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    return base, wrapper, env_cfg, {
        "context": context,
        "shift": shift,
        "cbf": cbf,
        "reward": reward,
        "dual_reward_scale": dual_reward_scale,
        "clearance": clearance_metadata,
    }


def _configure_algorithm(agent_cfg, args: argparse.Namespace, actual_target_fraction: float) -> None:
    from src.tasks.stairs_cbf.filter_free_v141 import (
        InterventionAwareCbfDistillationPpoAlgorithmCfg,
    )

    agent_cfg.algorithm = InterventionAwareCbfDistillationPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=args.actor_epochs,
        num_mini_batches=4,
        learning_rate=args.actor_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        hard_kl_ceiling=0.01,
        max_grad_norm=0.5,
        normalize_advantage_per_mini_batch=False,
        std_scale_from_base=0.35,
        minimum_std=args.exploration_std,
        maximum_std=args.exploration_std,
        moving_kl_beta=args.moving_kl_beta,
        critic_learning_epochs=2,
        freeze_log_std=True,
        teacher_distillation_weight=args.correction_loss_weight,
        teacher_success_horizon=50,
        teacher_correction_scale=0.05,
        teacher_mode="residual",
        teacher_gate="all_interventions",
        teacher_eta=1.0,
        teacher_smooth_l1_beta=0.05,
        intervention_ppo_eta=args.intervention_ppo_eta,
        correction_weight_mode=args.correction_weight_mode,
        v141_dual_reward_scale=args.dual_reward_scale,
        v141_target_fraction=actual_target_fraction,
        v141_intervention_epsilon=1.0e-5,
    )


def _collect_round(runner) -> dict[str, Any]:
    runner.alg.clear_cbf_rollout()
    runner.alg.train_mode()
    obs, _ = runner.env.reset()
    obs = obs.to(runner.device)
    n = runner.env.num_envs
    target_mask = runner.env.target_environment_mask
    group_masks = {"target": target_mask, "retention_f1": ~target_mask}
    episode_returns = torch.zeros(n, device=runner.env.device)
    episode_risers = torch.zeros(n, dtype=torch.long, device=runner.env.device)
    group_stats = {
        name: {"episodes": 0, "successes": 0, "falls": 0, "returns": [], "risers": []}
        for name in group_masks
    }
    reward_sum = task_reward_sum = cbf_reward_sum = 0.0
    would_intervene_count = 0
    nominal_violation_count = executed_violation_count = 0
    transition_count = n * runner.cfg["num_steps_per_env"]

    with torch.no_grad():
        for _ in range(runner.cfg["num_steps_per_env"]):
            actions = runner.alg.act(obs)
            next_obs, rewards, dones, extras = runner.env.step(
                actions.to(runner.env.device)
            )
            extras = dict(extras)
            success = extras["v141_success_terminal"].bool()
            done = dones.bool()
            fell = extras["online_fell"].bool() & ~success
            episode_returns += rewards
            episode_risers = torch.maximum(
                episode_risers, extras["online_stair_index"].long()
            )
            reward_sum += float(rewards.sum())
            task_reward_sum += float(extras["v141_task_reward"].sum())
            cbf_reward_sum += float(extras["v141_cbf_reward"].sum())
            would_intervene_count += int(extras["cbf_would_intervene"].sum())
            nominal = extras["cbf_nominal_barrier_margin"] < -1.0e-5
            filtered = extras["cbf_filtered_barrier_margin"] < -1.0e-5
            nominal_violation_count += int(nominal.sum())
            executed_violation_count += int(filtered.sum())
            if bool(done.any()):
                for name, mask in group_masks.items():
                    selected = done & mask
                    ids = selected.nonzero(as_tuple=False).flatten()
                    stats = group_stats[name]
                    stats["episodes"] += len(ids)
                    stats["successes"] += int((selected & success).sum())
                    stats["falls"] += int((selected & fell).sum())
                    stats["returns"].extend(
                        float(episode_returns[index]) for index in ids.tolist()
                    )
                    stats["risers"].extend(
                        int(episode_risers[index]) for index in ids.tolist()
                    )
                ids = done.nonzero(as_tuple=False).flatten()
                episode_returns[ids] = 0.0
                episode_risers[ids] = 0
            obs = next_obs.to(runner.device)
            runner.alg.process_env_step(
                obs,
                rewards.to(runner.device),
                dones.to(runner.device),
                extras,
            )
        label_metrics = runner.alg.relabel_teacher_transitions()
        runner.alg.compute_returns(obs)
        weight_metrics = runner.alg.prepare_v141_advantage_weights()

    update = runner.alg.update()
    metrics: dict[str, Any] = {
        **update,
        **label_metrics,
        **weight_metrics,
        "rollout_transition_count": transition_count,
        "rollout_mean_reward": reward_sum / transition_count,
        "rollout_mean_task_reward": task_reward_sum / transition_count,
        "rollout_mean_scaled_cbf_reward": cbf_reward_sum / transition_count,
        "rollout_would_intervene_fraction": would_intervene_count / transition_count,
        "rollout_nominal_violation_fraction": nominal_violation_count / transition_count,
        "rollout_executed_violation_fraction": executed_violation_count / transition_count,
    }
    for name, stats in group_stats.items():
        count = int(stats["episodes"])
        metrics.update(
            {
                f"rollout_{name}_episode_count": count,
                f"rollout_{name}_success_count": int(stats["successes"]),
                f"rollout_{name}_fall_count": int(stats["falls"]),
                f"rollout_{name}_success_rate": (
                    float(stats["successes"]) / count if count else None
                ),
                f"rollout_{name}_mean_return": (
                    float(np.mean(stats["returns"])) if count else None
                ),
                f"rollout_{name}_mean_reached_riser": (
                    float(np.mean(stats["risers"])) if count else None
                ),
            }
        )
    return metrics


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    output = args.output_dir.resolve()
    if args.rounds < 1 or args.num_envs < 2 or args.rollout_steps < 1:
        raise ValueError("v141 training dimensions must be positive")
    if not checkpoint.is_file() or file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v141 base checkpoint is missing or differs from v139")
    target_count = round(args.num_envs * args.target_fraction)
    retention_count = args.num_envs - target_count
    if target_count <= 0 or retention_count <= 0:
        raise ValueError("v141 target/retention allocation is empty")
    actual_target_fraction = target_count / args.num_envs
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(repo))
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    import src.tasks  # noqa: F401
    from mixed_vec_env_v141 import SpecialistMixedVecEnvV141
    from src.tasks.stairs_cbf.filter_free_v141 import (
        InterventionAwareCbfDistillationRunner,
    )

    records_path = output / "round_metrics.json"
    records = json.loads(records_path.read_text()) if args.resume and records_path.exists() else []
    recovery = output / f"round_{len(records):02d}.pt" if records else None
    target_base = retention_base = env = runner = None
    started = time.monotonic()
    try:
        target_base, target_wrapper, target_cfg, target_metadata = _configure_context_env(
            context=args.specialist,
            num_envs=target_count,
            seed=args.seed,
            device=args.device,
            dual_reward_scale=args.dual_reward_scale,
        )
        retention_base, retention_wrapper, retention_cfg, retention_metadata = _configure_context_env(
            context=RETENTION_CONTEXT,
            num_envs=retention_count,
            seed=args.seed + 10_000,
            device=args.device,
            dual_reward_scale=args.dual_reward_scale,
        )
        env = SpecialistMixedVecEnvV141(
            target_wrapper,
            retention_wrapper,
            target_context=args.specialist,
            dual_reward_scale=args.dual_reward_scale,
        )
        agent_cfg = load_rl_cfg(TASK_ID)
        agent_cfg.seed = args.seed
        agent_cfg.num_steps_per_env = args.rollout_steps
        _configure_algorithm(agent_cfg, args, actual_target_fraction)
        runner = InterventionAwareCbfDistillationRunner(
            env, asdict(agent_cfg), log_dir=None, device=args.device
        )
        runner.alg.set_context_group_mask(env.target_environment_mask)
        if recovery is None:
            warm_start = runner.load_initial_checkpoint(
                str(checkpoint), map_location=args.device
            )
            _save_checkpoint(
                runner,
                output / "round_00.pt",
                0,
                {"boundary": "v139", "specialist": args.specialist},
            )
        else:
            # Load the small metadata/RNG view on CPU; the runner separately
            # restores model and optimizer tensors onto the requested device.
            payload = torch.load(recovery, map_location="cpu", weights_only=False)
            warm_start = runner.load_recovery_checkpoint(
                str(recovery), map_location=args.device
            )
            _restore_rng_state(payload)

        initial_actor_sha = actor_state_sha256(actor_state(runner.alg.actor))
        if records:
            initial_actor_sha = json.loads((output / "training_started.json").read_text())[
                "initial_actor_sha256"
            ]
        else:
            _atomic_json(
                output / "training_started.json",
                {
                    "method_id": METHOD_ID,
                    "candidate": args.candidate,
                    "specialist": args.specialist,
                    "retention_context": RETENTION_CONTEXT,
                    "seed": args.seed,
                    "rounds": args.rounds,
                    "num_envs": args.num_envs,
                    "rollout_steps": args.rollout_steps,
                    "target_count": target_count,
                    "retention_count": retention_count,
                    "actual_target_fraction": actual_target_fraction,
                    "base_checkpoint": str(checkpoint),
                    "base_checkpoint_sha256": file_sha256(checkpoint),
                    "initial_actor_sha256": initial_actor_sha,
                    "git_commit": _git(repo, "rev-parse", "HEAD"),
                    "hyperparameters": {
                        "intervention_ppo_eta": args.intervention_ppo_eta,
                        "correction_weight_mode": args.correction_weight_mode,
                        "correction_loss_weight": args.correction_loss_weight,
                        "dual_reward_scale": args.dual_reward_scale,
                        "actor_learning_rate": args.actor_learning_rate,
                        "critic_learning_rate": args.critic_learning_rate,
                        "moving_kl_beta": args.moving_kl_beta,
                        "actor_epochs": args.actor_epochs,
                        "exploration_std": args.exploration_std,
                    },
                    "contexts": {
                        args.specialist: target_metadata,
                        RETENTION_CONTEXT: retention_metadata,
                    },
                    "selection_during_training": False,
                },
            )

        for round_index in range(len(records) + 1, args.rounds + 1):
            runner.alg.freeze_round_reference()
            start_hash = actor_state_sha256(actor_state(runner.alg.actor))
            round_started = time.monotonic()
            metrics = _collect_round(runner)
            end_hash = actor_state_sha256(actor_state(runner.alg.actor))
            record = {
                "round": round_index,
                "round_start_actor_sha256": start_hash,
                "round_end_actor_sha256": end_hash,
                "elapsed_seconds": time.monotonic() - round_started,
                "metrics": metrics,
            }
            records.append(record)
            _save_checkpoint(
                runner,
                output / f"round_{round_index:02d}.pt",
                round_index,
                {
                    "boundary": "round_end",
                    "candidate": args.candidate,
                    "specialist": args.specialist,
                    "actor_sha256": end_hash,
                },
            )
            _atomic_json(records_path, records)
            print(json.dumps(record, sort_keys=True), flush=True)

        final_checkpoint = output / f"round_{args.rounds:02d}.pt"
        summary = {
            "schema_version": 1,
            "method_id": METHOD_ID,
            "candidate": args.candidate,
            "specialist": args.specialist,
            "retention_context": RETENTION_CONTEXT,
            "seed": args.seed,
            "rounds": args.rounds,
            "num_envs": args.num_envs,
            "rollout_steps": args.rollout_steps,
            "target_count": target_count,
            "retention_count": retention_count,
            "target_fraction": actual_target_fraction,
            "warm_start": warm_start,
            "base_checkpoint_sha256": file_sha256(checkpoint),
            "initial_actor_sha256": initial_actor_sha,
            "final_actor_sha256": actor_state_sha256(actor_state(runner.alg.actor)),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "fixed_final_round_checkpoint": True,
            "best_so_far_selection": False,
            "hyperparameters": json.loads((output / "training_started.json").read_text())[
                "hyperparameters"
            ],
            "round_metrics": records,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(output / "training_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

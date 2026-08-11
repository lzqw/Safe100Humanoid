"""Run the single fixed-eight-round v24 pure-contact adaptation."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from proximal_v24_protocol import (
    ADAPTATION_SEED,
    BASE_CHECKPOINT_SHA256,
    CONTEXT_ID,
    EXPERIMENT_NAME,
    POLICY_METHOD,
    PROTOCOL_ID,
    formal_algorithm_parameters,
    validate_v24_calibrated_context,
)
from refine_proximal_v23 import (
    _algorithm_audit,
    _collect_one_round,
    _configure_algorithm,
    _save_checkpoint,
    _write_json,
    _write_round_csv,
)

ROUNDS = 8
NUM_ENVS = 64
ROLLOUT_STEPS = 1024


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
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_frozen_protocol(
    repo: Path,
    protocol_path: Path,
    *,
    checkpoint: Path,
    context_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    implementation_commit = str(
        protocol.get("implementation_boundary", {}).get("git_commit", "")
    )
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
        "experiment_name": protocol.get("experiment_name") == EXPERIMENT_NAME,
        "method": protocol.get("policy_method") == POLICY_METHOD,
        "status": protocol.get("status")
        == "prospectively_frozen_after_base_only_calibration_before_adaptation",
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
        "context_id": protocol.get("context", {}).get("context_id")
        == context.get("context_id")
        == CONTEXT_ID,
        "algorithm_unchanged": protocol.get("training")
        == formal_algorithm_parameters(),
        "adaptation_not_started": protocol.get("prospective_execution", {}).get(
            "adaptation_started"
        )
        is False,
        "adapted_outcomes_absent": protocol.get("prospective_execution", {}).get(
            "adapted_policy_outcomes_observed"
        )
        is False,
        "single_planned_adaptation": protocol.get("prospective_execution", {}).get(
            "fresh_adaptation_count_planned"
        )
        == 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v24 frozen protocol validation failed: {checks}")
    return {
        "file": str(protocol_path),
        "sha256": file_sha256(protocol_path),
        "implementation_commit": implementation_commit,
        "validation": checks,
    }


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    context_path = args.context.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file() or not context_path.is_file():
        raise FileNotFoundError("base checkpoint or v24 context does not exist")
    context_raw = json.loads(context_path.read_text())
    context = validate_v24_calibrated_context(context_raw)
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
            raise ValueError(f"formal v24 execution mismatch: {mismatches}")
        if args.protocol is None:
            raise ValueError("formal v24 execution requires a frozen protocol")
        if (output_dir / "formal_execution_started.json").exists():
            raise RuntimeError("formal v24 adaptation has already been started")
    elif min(args.rounds, args.num_envs, args.rollout_steps) < 1:
        raise ValueError("smoke rollout sizes must be positive")

    protocol_reference = (
        _validate_frozen_protocol(
            repo,
            args.protocol.resolve(),
            checkpoint=checkpoint,
            context_path=context_path,
            context=context,
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
                "v23_lateral_result_modified": False,
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
    from src.tasks.stairs_cbf.proximal import (
        CbfProximalRefinementRunner,
        ProximalHardRollback,
    )
    from src.tasks.stairs_cbf.proximal_context import apply_cbf_proximal_context

    env_cfg = load_env_cfg("Unitree-G1-Stairs-Online-DQHMED")
    context_metadata = apply_cbf_proximal_context(env_cfg, context, role="target")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.actions["joint_pos"].enabled = True
    agent_cfg = load_rl_cfg("Unitree-G1-Stairs-Online-DQHMED")
    agent_cfg.seed = args.seed
    agent_cfg.num_steps_per_env = args.rollout_steps
    _configure_algorithm(agent_cfg)
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = CbfProximalRefinementRunner(
        env, asdict(agent_cfg), log_dir=None, device=args.device
    )
    try:
        structural_audit = _algorithm_audit(runner, env_cfg, context_metadata)
        warm_start = runner.load_initial_checkpoint(
            str(checkpoint), map_location=args.device
        )
        if (
            warm_start["actor_observation_dim"] != 405
            or warm_start["critic_observation_dim"] != 838
        ):
            raise RuntimeError("v24 warm start changed the original interface")
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
                    "experiment": EXPERIMENT_NAME,
                    "round": round_index,
                    "boundary": "start",
                    "actor_sha256": round_start_sha,
                },
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
                metrics = dict(rollback.metrics)
                metrics.update(
                    {
                        "hard_rollback": True,
                        "hard_rollback_reason": rollback.reason,
                    }
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
                metadata={"experiment": EXPERIMENT_NAME, **record},
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
                "experiment": EXPERIMENT_NAME,
                "round": args.rounds,
                "boundary": "final",
                "selection": "fixed final round; no validation or performance selection",
            },
        )
        final_actor_sha = actor_state_sha256(actor_state(runner.alg.actor))
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "experiment_name": EXPERIMENT_NAME,
            "policy_method": POLICY_METHOD,
            "experiment_class": "independent single pure-contact development test",
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
                "calibration_selected_candidate_parameter_seed": context["calibration"][
                    "selected_candidate_parameter_seed"
                ],
                "selected_foot_friction": context["calibration"][
                    "selected_foot_friction"
                ],
                "base_policy_only_first_qualifier": True,
                "metadata": context_metadata,
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
            "specialist_bank_count": 0,
            "dual_rollout_batch_count": 0,
            "v23_lateral_result_modified": False,
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
                    "v23_lateral_result_modified": False,
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()

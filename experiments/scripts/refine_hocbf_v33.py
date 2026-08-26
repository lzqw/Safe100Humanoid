"""Run one unconditional eight-round v33 A2 refinement from the common base."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cbf_teacher_v31_protocol import (
    BASE_CHECKPOINT_SHA256,
    CLEARANCE_BARRIER_SLOPE,
    FILTER_ALPHA,
    RECOVERY_DISTANCE_M,
    TASK_ID,
    common_training_parameters,
    environment_parameters,
)
from hocbf_v33_protocol import HOCBF_MODE, PROTOCOL_ID, TRAINING_SEEDS
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_cbf_teacher_v31 import (
    _collect_round,
    _configure_algorithm,
    _restore_rng_state,
    _write_round_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--context", choices=tuple(TRAINING_SEEDS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_checkpoint(
    runner, path: Path, round_index: int, metadata: dict[str, Any]
) -> None:
    payload = runner.alg.save()
    payload["iter"] = round_index
    payload["infos"] = {"hocbf_v33": metadata}
    payload["python_random_state"] = random.getstate()
    payload["numpy_random_state"] = np.random.get_state()
    payload["torch_random_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _resume_boundary(output: Path) -> tuple[list[dict[str, Any]], Path]:
    records_path = output / "round_metrics.json"
    if not records_path.is_file() or not (output / "execution_started.json").is_file():
        raise RuntimeError("v33 resume boundary is incomplete")
    if (output / "execution_completed.json").exists():
        raise RuntimeError("v33 training is already complete")
    records = json.loads(records_path.read_text())
    if [int(row["round"]) for row in records] != list(range(1, len(records) + 1)):
        raise RuntimeError("v33 resume rounds are not consecutive")
    if not 0 < len(records) < 8:
        raise RuntimeError("v33 resume requires one through seven rounds")
    checkpoint = output / f"round_{len(records):02d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return records, checkpoint


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    config = json.loads(args.config.resolve().read_text())
    selected = json.loads(args.selected.resolve().read_text())
    base_checkpoint = args.base_checkpoint.resolve()
    output = args.output_dir.resolve()
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or selected.get("protocol_id") != PROTOCOL_ID
    ):
        raise RuntimeError("v33 training inputs differ")
    if selected.get("status") != "globally_selected_and_frozen":
        raise RuntimeError("v33 HOCBF parameters are not frozen")
    if file_sha256(base_checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v33 training base checkpoint differs")
    seed = TRAINING_SEEDS[args.context]
    records: list[dict[str, Any]] = []
    recovery: Path | None = None
    if output.exists():
        if not args.resume:
            raise RuntimeError(f"v33 training output exists: {output}")
        records, recovery = _resume_boundary(output)
    else:
        if args.resume:
            raise RuntimeError("v33 resume output is absent")
        output.mkdir(parents=True)
        _atomic_json(
            output / "execution_started.json",
            {
                "protocol_id": PROTOCOL_ID,
                "context": args.context,
                "seed": seed,
                "rounds": 8,
                "num_envs": 64,
                "rollout_steps": 1024,
                "initial_policy": "common_base",
                "method": "v31_A2_with_selected_v33_HOCBF",
                "performance_gate": False,
                "KL_gate": False,
                "checkpoint_selection": False,
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
    from src.tasks.stairs_cbf.hocbf_action import (
        TaskConsistentHocbfActionCfg,
        configure_v33_cbf,
    )
    from src.tasks.stairs_cbf.teacher_v30 import CbfTeacherV30Runner

    env_cfg = load_env_cfg(TASK_ID, play=True)
    shift = configure_v31_context(
        env_cfg,
        context=args.context,
        runtime_filter=True,
        context_spec=environment_parameters(args.context),
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    cbf = configure_v33_cbf(
        env_cfg,
        mode=HOCBF_MODE,
        runtime_filter=True,
        omega=float(selected["omega"]),
        forward_task_weight=float(selected["lambda_x"]),
        correction_smoothness=float(selected["lambda_s"]),
        measure_compute_time=False,
    )
    env_cfg.scene.num_envs = 64
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(TASK_ID)
    agent_cfg.seed = seed
    agent_cfg.num_steps_per_env = 1024
    _configure_algorithm(agent_cfg, "A2", preflight=False)
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = CbfTeacherV30Runner(
        env, asdict(agent_cfg), log_dir=None, device=args.device
    )
    try:
        action_cfg = env_cfg.actions["joint_pos"]
        structural = {
            "action": type(action_cfg).__name__,
            "selected_HOCBF_action": isinstance(
                action_cfg, TaskConsistentHocbfActionCfg
            ),
            "runtime_filter": bool(action_cfg.enabled),
            "actor_observation_dim": int(runner.alg.actor.obs_dim),
            "critic_observation_dim": int(runner.alg.critic.obs_dim),
            "rounds": 8,
            "num_envs": 64,
            "steps_per_round": 1024,
            "actor_epochs": int(runner.alg.num_learning_epochs),
            "critic_epochs": int(runner.alg.critic_learning_epochs),
            "plant_identity": shift["plant_action_transform"] == "identity",
            "performance_gate": False,
            "KL_stop_or_rollback": False,
        }
        required = (
            structural["selected_HOCBF_action"]
            and structural["runtime_filter"]
            and structural["actor_observation_dim"] == 405
            and structural["critic_observation_dim"] == 838
            and structural["plant_identity"]
        )
        if not required:
            raise RuntimeError(f"v33 training structure differs: {structural}")
        if recovery is None:
            warm_start = runner.load_initial_checkpoint(
                str(base_checkpoint), map_location=args.device
            )
            initial_hash = actor_state_sha256(actor_state(runner.alg.actor))
            _save_checkpoint(
                runner,
                output / "round_00.pt",
                0,
                {"boundary": "common_base", "context": args.context},
            )
        else:
            payload = torch.load(recovery, map_location=args.device, weights_only=False)
            warm_start = runner.load_recovery_checkpoint(
                str(recovery), map_location=args.device
            )
            _restore_rng_state(payload)
            round_zero = torch.load(
                output / "round_00.pt", map_location="cpu", weights_only=False
            )
            initial_hash = actor_state_sha256(round_zero["actor_state_dict"])
        for round_index in range(len(records) + 1, 9):
            runner.alg.freeze_round_reference()
            start_hash = actor_state_sha256(actor_state(runner.alg.actor))
            round_started = time.monotonic()
            metrics = _collect_round(runner)
            end_hash = actor_state_sha256(actor_state(runner.alg.actor))
            record = {
                "round": round_index,
                "status": "updated_unconditionally",
                "round_start_actor_sha256": start_hash,
                "round_end_actor_sha256": end_hash,
                "performance_gate_used": False,
                "kl_gate_used": False,
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
                    "context": args.context,
                    "actor_sha256": end_hash,
                },
            )
            _atomic_json(output / "round_metrics.json", records)
            _write_round_csv(output / "round_metrics.csv", records)
            print(json.dumps(record, sort_keys=True), flush=True)
        final_checkpoint = output / "round_08.pt"
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "context": args.context,
            "seed": seed,
            "method": "A2",
            "initial_policy": "common_base",
            "selected_hocbf": cbf,
            "common_training": common_training_parameters(),
            "structural_audit": structural,
            "warm_start": warm_start,
            "base_checkpoint_sha256": file_sha256(base_checkpoint),
            "initial_actor_sha256": initial_hash,
            "final_actor_sha256": actor_state_sha256(actor_state(runner.alg.actor)),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "final_policy_rule": "unconditional round 8",
            "rounds_completed": len(records),
            "rounds": records,
            "performance_selection_count": 0,
            "kl_stop_or_rollback_count": 0,
            "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in records),
        }
        _atomic_json(output / "training_summary.json", summary)
        _atomic_json(
            output / "execution_completed.json",
            {
                "protocol_id": PROTOCOL_ID,
                "context": args.context,
                "rounds_completed": 8,
                "final_checkpoint_sha256": file_sha256(final_checkpoint),
            },
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

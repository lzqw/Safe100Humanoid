"""Run v34's only fixed 8-env by 256-step implementation smoke."""

from __future__ import annotations

import argparse
import inspect
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
    CLEARANCE_BARRIER_SLOPE,
    FILTER_ALPHA,
    RECOVERY_DISTANCE_M,
    TASK_ID,
    environment_parameters,
)
from proximal_v23_io import file_sha256
from velocity_cbf_v34_protocol import (
    CURRENT_CBF_MODE,
    OPTIMIZED_CBF_MODE,
    PROTOCOL_ID,
    SMOKE_ENVS,
    SMOKE_SEED,
    SMOKE_STEPS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _seed() -> None:
    random.seed(SMOKE_SEED)
    np.random.seed(SMOKE_SEED)
    torch.manual_seed(SMOKE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SMOKE_SEED)


def _run(
    *,
    checkpoint: Path,
    mode: str,
    parameters: dict[str, Any] | None,
    device: str,
) -> dict[str, Any]:
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import src.tasks  # noqa: F401
    from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
    from src.tasks.stairs_cbf.velocity_cbf_action import configure_v34_cbf

    _seed()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    configure_v31_context(
        env_cfg,
        context="F1",
        runtime_filter=True,
        context_spec=environment_parameters("F1"),
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    cbf = configure_v34_cbf(
        env_cfg,
        mode=mode,
        runtime_filter=True,
        parameters=parameters,
        measure_compute_time=True,
    )
    env_cfg.scene.num_envs = SMOKE_ENVS
    env_cfg.seed = SMOKE_SEED
    base_env = ManagerBasedRlEnv(env_cfg, device=device)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError("v34 smoke task has no inference runner")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=device)
    try:
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
        )
        obs, _ = env.reset()
        policy = runner.get_inference_policy(device)
        term = base_env.action_manager.get_term("joint_pos")
        violation_steps = correction_steps = 0
        minimum_margin = float("inf")
        maximum_safe_error = maximum_routing_error = 0.0
        finite = True
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device=device)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(SMOKE_STEPS):
                obs, _, _, extras = env.step(policy(obs))
                extras = dict(extras)
                violation = term.geometric_active & (term.psi_nominal < 0.0)
                violation_steps += int(violation.sum())
                correction_steps += int(
                    (extras["v34_velocity_correction_norm"] > 1.0e-5).sum()
                )
                if bool(violation.any()):
                    minimum_margin = min(
                        minimum_margin,
                        float(extras["v34_filtered_margin"][violation].amin()),
                    )
                maximum_safe_error = max(
                    maximum_safe_error,
                    float(extras["v34_nominal_safe_target_error"].amax()),
                )
                maximum_routing_error = max(
                    maximum_routing_error,
                    float(
                        torch.amax(
                            torch.abs(term._processed_actions - term.safe_target)
                        )
                    ),
                )
                finite = finite and bool(
                    torch.isfinite(
                        torch.stack(
                            (
                                extras["v34_nominal_margin"],
                                extras["v34_filtered_margin"],
                                extras["v34_velocity_correction_norm"],
                                extras["v34_velocity_correction_jerk"],
                            ),
                            dim=-1,
                        )
                    ).all()
                )
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device=device)
        elapsed = time.perf_counter() - started
        return {
            "mode": mode,
            "cbf": cbf,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": SMOKE_ENVS * SMOKE_STEPS / elapsed,
            "mean_cbf_compute_time_ms": term.mean_compute_time_ms(),
            "violation_steps": violation_steps,
            "nonzero_correction_steps": correction_steps,
            "minimum_filtered_margin": (
                0.0 if minimum_margin == float("inf") else minimum_margin
            ),
            "maximum_nominal_safe_target_error": maximum_safe_error,
            "maximum_executed_safe_target_routing_error": maximum_routing_error,
            "all_finite": finite,
            "device": str(term.nominal_target.device),
        }
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    config = json.loads(args.search_config.resolve().read_text())
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("v34 smoke config differs")
    candidate = config["candidate_generator"]["candidates"][17]
    parameters = {
        key: value
        for key, value in candidate.items()
        if key not in ("candidate", "candidate_index", "mode")
    }
    sys.path.insert(0, str(args.repo.resolve()))
    from src.tasks.stairs_cbf.velocity_cbf_action import TaskMetricVelocityCbfAction

    current = _run(
        checkpoint=args.checkpoint.resolve(),
        mode=CURRENT_CBF_MODE,
        parameters=None,
        device=args.device,
    )
    optimized = _run(
        checkpoint=args.checkpoint.resolve(),
        mode=OPTIMIZED_CBF_MODE,
        parameters=parameters,
        device=args.device,
    )
    hot_path = inspect.getsource(TaskMetricVelocityCbfAction.process_actions)
    forbidden = [
        token
        for token in (".cpu(", ".numpy(", ".item(", ".tolist(", "bool(")
        if token in hot_path
    ]
    checks = {
        "single_smoke_invocation": True,
        "exact_nominal_safe_identity": optimized["maximum_nominal_safe_target_error"]
        == 0.0,
        "violation_observed": optimized["violation_steps"] > 0,
        "violation_correction_nonzero": optimized["nonzero_correction_steps"] > 0,
        "hard_margin_within_tolerance": optimized["minimum_filtered_margin"] >= -1.0e-5,
        "safe_action_routing_exact": optimized[
            "maximum_executed_safe_target_routing_error"
        ]
        == 0.0,
        "finite": optimized["all_finite"],
        "gpu_vectorized": optimized["device"].startswith("cuda"),
        "no_cpu_sync_in_action_hot_path": not forbidden,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "seed": SMOKE_SEED,
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "candidate": candidate["candidate"],
        "parameters": parameters,
        "current_CBF0": current,
        "task_metric_velocity_CBF": optimized,
        "throughput_ratio": optimized["environment_steps_per_second"]
        / current["environment_steps_per_second"],
        "forbidden_hot_path_tokens": forbidden,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.resolve().with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError(f"v34 smoke failed: {checks}")


if __name__ == "__main__":
    main()

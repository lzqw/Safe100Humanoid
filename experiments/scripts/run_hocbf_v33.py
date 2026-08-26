"""Run the single fixed v33 implementation smoke comparison."""

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
from hocbf_v33_protocol import (
    CURRENT_CBF_MODE,
    HOCBF_MODE,
    PROTOCOL_ID,
    SMOKE_ENVS,
    SMOKE_SEED,
    SMOKE_STEPS,
)
from proximal_v23_io import file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_mode(
    *,
    repo: Path,
    checkpoint: Path,
    mode: str,
    parameters: dict[str, float] | None,
    device: str,
) -> dict[str, Any]:
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import src.tasks  # noqa: F401
    from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
    from src.tasks.stairs_cbf.hocbf_action import configure_v33_cbf

    _seed(SMOKE_SEED)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    shift = configure_v31_context(
        env_cfg,
        context="F1",
        runtime_filter=True,
        context_spec=environment_parameters("F1"),
        clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
        recovery_distance_m=RECOVERY_DISTANCE_M,
        filter_alpha=FILTER_ALPHA,
    )
    kwargs: dict[str, Any] = {}
    if parameters is not None:
        kwargs = {
            "omega": parameters["omega"],
            "forward_task_weight": parameters["lambda_x"],
            "correction_smoothness": parameters["lambda_s"],
        }
    cbf = configure_v33_cbf(
        env_cfg,
        mode=mode,
        runtime_filter=True,
        measure_compute_time=True,
        **kwargs,
    )
    env_cfg.scene.num_envs = SMOKE_ENVS
    env_cfg.seed = SMOKE_SEED
    base_env = ManagerBasedRlEnv(env_cfg, device=device)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError("v33 smoke task has no inference runner")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=device)
    try:
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
        )
        policy = runner.get_inference_policy(device)
        obs, _ = env.reset()
        term = base_env.action_manager.get_term("joint_pos")
        correction_steps = violation_steps = 0
        safe_error = 0.0
        minimum_projected_margin = float("inf")
        finite = True
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device=device)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(SMOKE_STEPS):
                obs, _, _, extras = env.step(policy(obs))
                extras = dict(extras)
                active = term.geometric_active
                violation = active & (term.psi_nominal < 0.0)
                correction = extras["v33_qddot_correction_norm"]
                correction_steps += int((correction > 1.0e-5).sum())
                violation_steps += int(violation.sum())
                safe_error = max(
                    safe_error,
                    float(extras["v33_nominal_safe_target_error"].amax()),
                )
                if bool(violation.any()):
                    minimum_projected_margin = min(
                        minimum_projected_margin,
                        float(extras["v33_projected_hocbf_margin"][violation].amin()),
                    )
                finite = finite and bool(
                    torch.isfinite(
                        torch.stack(
                            (
                                extras["v33_h_dot"],
                                extras["v33_estimated_drift"],
                                extras["v33_projected_hocbf_margin"],
                                correction,
                            ),
                            dim=-1,
                        )
                    ).all()
                )
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device=device)
        elapsed = time.perf_counter() - started
        return {
            "mode": mode,
            "cbf": cbf,
            "shift": shift,
            "steps": SMOKE_STEPS,
            "num_envs": SMOKE_ENVS,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": SMOKE_ENVS * SMOKE_STEPS / elapsed,
            "mean_cbf_compute_time_ms": term.mean_compute_time_ms(),
            "violation_steps": violation_steps,
            "nonzero_correction_steps": correction_steps,
            "maximum_nominal_safe_target_error": safe_error,
            "minimum_projected_margin_on_violation": (
                0.0
                if minimum_projected_margin == float("inf")
                else minimum_projected_margin
            ),
            "all_finite": finite,
            "action_device": str(term.nominal_target.device),
        }
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("v33 smoke config differs")
    sys.path.insert(0, str(repo))
    from src.tasks.stairs_cbf.hocbf_action import TaskConsistentHocbfAction

    parameters = {"omega": 8.0, "lambda_x": 8.0, "lambda_s": 0.1}
    current = _run_mode(
        repo=repo,
        checkpoint=checkpoint,
        mode=CURRENT_CBF_MODE,
        parameters=None,
        device=args.device,
    )
    hocbf = _run_mode(
        repo=repo,
        checkpoint=checkpoint,
        mode=HOCBF_MODE,
        parameters=parameters,
        device=args.device,
    )
    source = inspect.getsource(TaskConsistentHocbfAction.process_actions)
    forbidden = [
        token
        for token in (".cpu(", ".numpy(", ".item(", ".tolist(", "bool(")
        if token in source
    ]
    throughput_ratio = (
        hocbf["environment_steps_per_second"] / current["environment_steps_per_second"]
    )
    checks = {
        "single_fixed_invocation": True,
        "exact_safe_nominal_identity": hocbf["maximum_nominal_safe_target_error"]
        == 0.0,
        "violation_observed": hocbf["violation_steps"] > 0,
        "violation_has_nonzero_correction": hocbf["nonzero_correction_steps"] > 0,
        "filtered_margin_within_tolerance": hocbf[
            "minimum_projected_margin_on_violation"
        ]
        >= -1.0e-4,
        "no_nan_or_inf": hocbf["all_finite"],
        "gpu_vectorized": hocbf["action_device"].startswith("cuda"),
        "no_cpu_sync_in_hocbf_hot_path": not forbidden,
        "throughput_at_least_70_percent_of_CBF0": throughput_ratio >= 0.70,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": file_sha256(checkpoint),
        "seed": SMOKE_SEED,
        "parameters": parameters,
        "current_CBF0": current,
        "new_HOCBF": hocbf,
        "throughput_ratio": throughput_ratio,
        "forbidden_hot_path_tokens": forbidden,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError(f"v33 single smoke failed: {checks}")


if __name__ == "__main__":
    main()

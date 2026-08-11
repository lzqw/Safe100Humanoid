"""Fresh paired target/D0 audit for fixed-round v24 Contact Completion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from proximal_v24_protocol import (
    BASE_CHECKPOINT_SHA256,
    EVAL_BATCH_SIZE,
    EXPERIMENT_NAME,
    FINAL_D0_EPISODES,
    FINAL_D0_SEED,
    FINAL_TARGET_EPISODES,
    FINAL_TARGET_SEED,
    POLICY_METHOD,
    PROTOCOL_ID,
    REPORT_BOOTSTRAP_SAMPLES,
    REPORT_BOOTSTRAP_SEEDS,
    development_gate,
    paired_repair_regression_counts,
    pure_contact_context_audit,
    validate_v24_calibrated_context,
)
from refine_proximal_v23 import _configure_algorithm, _write_json

METRIC_FIELDS = (
    "success_rate",
    "fall_rate",
    "timeout_rate",
    "mean_return",
    "mean_reached_riser",
    "mean_slip_signal",
    "mean_contact_mismatch",
    "intervention_per_riser",
    "mean_correction_norm",
    "recovery_takeover_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _evaluate_state(
    runner,
    state: dict[str, torch.Tensor],
    *,
    task: str,
    num_envs: int,
    repeats: int,
    seed: int,
    device: str,
    artifact_dir: Path,
    repo: Path,
    context: Path | None,
    resume: bool,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    actor_hash = actor_state_sha256(state)
    checkpoint_payload = runner.alg.save()
    checkpoint_payload["actor_state_dict"] = {
        key: value.detach().cpu() for key, value in state.items()
    }
    checkpoint_payload.setdefault("iter", 0)
    checkpoint_payload.setdefault("infos", {})
    checkpoint = artifact_dir / "actor.pt"
    torch.save(checkpoint_payload, checkpoint)
    summaries: list[dict[str, Any]] = []
    for repeat in range(repeats):
        evaluation_seed = seed + repeat
        stem = f"{task.rsplit('-', 1)[-1]}-seed{evaluation_seed}"
        output_json = artifact_dir / f"{stem}.json"
        output_csv = artifact_dir / f"{stem}.csv"
        summary = None
        if resume and output_json.is_file() and output_csv.is_file():
            try:
                candidate = json.loads(output_json.read_text())
            except (json.JSONDecodeError, OSError):
                candidate = None
            if (
                isinstance(candidate, dict)
                and candidate.get("task") == task
                and candidate.get("seed") == evaluation_seed
                and candidate.get("num_episodes") == num_envs
                and candidate.get("actor_state_sha256") == actor_hash
            ):
                summary = candidate
        if summary is None:
            command = [
                sys.executable,
                str(repo / "experiments/scripts/evaluate_proximal_v24.py"),
                "--repo",
                str(repo),
                "--task",
                task,
                "--checkpoint",
                str(checkpoint),
                "--num-envs",
                str(num_envs),
                "--num-episodes",
                str(num_envs),
                "--seed",
                str(evaluation_seed),
                "--device",
                device,
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
            if context is not None:
                command.extend(("--deployment-context", str(context)))
            completed = subprocess.run(
                command, cwd=repo, check=False, capture_output=True, text=True
            )
            if completed.returncode != 0:
                diagnostic = "\n".join(
                    (completed.stdout + "\n" + completed.stderr).splitlines()[-160:]
                )
                raise RuntimeError(
                    f"isolated v24 evaluation failed for {stem}:\n{diagnostic}"
                )
            summary = json.loads(output_json.read_text())
        if summary.get("actor_state_sha256") != actor_hash:
            raise RuntimeError("isolated v24 evaluator loaded a different actor")
        summaries.append(summary)
    aggregate: dict[str, Any] = {
        "task": task,
        "num_episodes": repeats * num_envs,
        "repeats": repeats,
        "seeds": [seed + index for index in range(repeats)],
        "replicates": summaries,
        "runtime_filter": True,
        "original_observation_interface": True,
        "paired_one_initial_episode_per_env": True,
        "initial_state_signatures": [
            summary["initial_state_signature"] for summary in summaries
        ],
        "actor_state_sha256": actor_hash,
        "recovery_takeover_count": sum(
            int(summary["recovery_takeover_count"]) for summary in summaries
        ),
    }
    for key in METRIC_FIELDS:
        values = [float(summary[key]) for summary in summaries]
        aggregate[key] = sum(values) / len(values)
        aggregate[f"{key}_std"] = (
            math.sqrt(
                sum((value - aggregate[key]) ** 2 for value in values)
                / (len(values) - 1)
            )
            if len(values) > 1
            else 0.0
        )
    failure_counts: dict[str, int] = {}
    for summary in summaries:
        for failure_type, count in summary["failure_type_counts"].items():
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + int(
                count
            )
    aggregate["failure_type_counts"] = failure_counts
    return aggregate


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"invalid CSV boolean: {value!r}")
    return normalized == "true"


def _read_episode_rows(
    root: Path, domain: str, seeds: list[int]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in seeds:
        path = root / f"{domain}-seed{seed}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                output.append(
                    {
                        "seed": int(row["evaluation_seed"]),
                        "environment_id": int(row["environment_id"]),
                        "success": _bool(row["success"]),
                        "fell": _bool(row["fell"]),
                        "timed_out": _bool(row["timed_out"]),
                        "failure_type": row["failure_type"],
                        "return": float(row["return"]),
                        "max_riser": int(row["max_riser"]),
                        "mean_slip_signal": float(row["mean_slip_signal"]),
                        "maximum_left_contact_slip_speed": float(
                            row["maximum_left_contact_slip_speed"]
                        ),
                        "maximum_right_contact_slip_speed": float(
                            row["maximum_right_contact_slip_speed"]
                        ),
                        "mean_contact_mismatch": float(row["mean_contact_mismatch"]),
                        "intervention_per_riser": float(row["intervention_per_riser"]),
                        "mean_correction_norm": float(row["mean_correction_norm"]),
                        "recovery_takeover": _bool(row["recovery_takeover"]),
                    }
                )
    output.sort(key=lambda row: (row["seed"], row["environment_id"]))
    return output


def _paired_rows(
    domain: str,
    baseline: list[dict[str, Any]],
    final: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(baseline) != len(final):
        raise RuntimeError("v24 paired row counts differ")
    rows: list[dict[str, Any]] = []
    for index, (old, new) in enumerate(zip(baseline, final, strict=True)):
        identity = (old["seed"], old["environment_id"])
        if identity != (new["seed"], new["environment_id"]):
            raise RuntimeError("v24 paired identities differ")
        row: dict[str, Any] = {
            "domain": domain,
            "pair_index": index,
            "evaluation_seed": identity[0],
            "environment_id": identity[1],
        }
        for key in (
            "success",
            "fell",
            "timed_out",
            "failure_type",
            "return",
            "max_riser",
            "mean_slip_signal",
            "maximum_left_contact_slip_speed",
            "maximum_right_contact_slip_speed",
            "mean_contact_mismatch",
            "intervention_per_riser",
            "mean_correction_norm",
            "recovery_takeover",
        ):
            row[f"baseline_{key}"] = old[key]
            row[f"final_{key}"] = new[key]
        rows.append(row)
    return rows


def _paired_interval(
    baseline: list[float],
    final: list[float],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    old = np.asarray(baseline, dtype=np.float64)
    new = np.asarray(final, dtype=np.float64)
    if old.shape != new.shape or old.ndim != 1 or old.size < 1:
        raise ValueError("v24 paired vectors must be equal and non-empty")
    delta = new - old
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    means = delta[indices].mean(axis=1)
    return {
        "baseline_mean": float(old.mean()),
        "final_mean": float(new.mean()),
        "delta": float(delta.mean()),
        "paired_bootstrap_ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "confidence_interval_is_gate": False,
    }


def _domain_report(
    rows: list[dict[str, Any]], *, bootstrap_seed: int, bootstrap_samples: int
) -> dict[str, Any]:
    metrics = (
        ("success", "success"),
        ("fall", "fell"),
        ("return", "return"),
        ("reached_riser", "max_riser"),
        ("slip_signal", "mean_slip_signal"),
        ("contact_mismatch", "mean_contact_mismatch"),
        ("intervention_per_riser", "intervention_per_riser"),
        ("correction_norm", "mean_correction_norm"),
        ("recovery_takeover", "recovery_takeover"),
    )
    report: dict[str, Any] = {"paired_conditions": len(rows)}
    for offset, (report_name, row_name) in enumerate(metrics):
        report[report_name] = _paired_interval(
            [float(row[f"baseline_{row_name}"]) for row in rows],
            [float(row[f"final_{row_name}"]) for row in rows],
            seed=bootstrap_seed + offset,
            samples=bootstrap_samples,
        )
    baseline_success = [bool(row["baseline_success"]) for row in rows]
    final_success = [bool(row["final_success"]) for row in rows]
    report["repairs_regressions"] = paired_repair_regression_counts(
        baseline_success, final_success
    )
    report["recovery_takeover_counts"] = {
        "baseline": sum(bool(row["baseline_recovery_takeover"]) for row in rows),
        "final": sum(bool(row["final_recovery_takeover"]) for row in rows),
    }
    report["failure_type_counts"] = {}
    for role in ("baseline", "final"):
        counts: dict[str, int] = {}
        for row in rows:
            if bool(row[f"{role}_fell"]):
                failure_type = str(row[f"{role}_failure_type"])
                counts[failure_type] = counts.get(failure_type, 0) + 1
        report["failure_type_counts"][role] = counts
    return report


def _write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("v24 paired CSV cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    final_checkpoint = args.final_checkpoint.resolve()
    training_path = args.training_summary.resolve()
    context_path = args.context.resolve()
    protocol_path = None if args.protocol is None else args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    for path in (checkpoint, final_checkpoint, training_path, context_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    training = json.loads(training_path.read_text())
    context = validate_v24_calibrated_context(json.loads(context_path.read_text()))
    context_audit = pure_contact_context_audit(context)
    if not args.smoke and protocol_path is None:
        raise ValueError("formal v24 audit requires the frozen protocol")
    protocol = json.loads(protocol_path.read_text()) if protocol_path else {}
    input_checks = {
        "training_protocol_id": training.get("protocol_id") == PROTOCOL_ID,
        "training_experiment_name": training.get("experiment_name") == EXPERIMENT_NAME,
        "training_method": training.get("policy_method") == POLICY_METHOD,
        "protocol_id": args.smoke or protocol.get("protocol_id") == PROTOCOL_ID,
        "base_checkpoint": training.get("base_checkpoint", {}).get("sha256")
        == file_sha256(checkpoint)
        == BASE_CHECKPOINT_SHA256,
        "context_file": training.get("context", {}).get("file_sha256")
        == file_sha256(context_path),
        "context_parameters": training.get("context", {}).get("parameters_sha256")
        == context["parameters_sha256"],
        "final_checkpoint": training.get("final_checkpoint_sha256")
        == file_sha256(final_checkpoint),
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"v24 final audit inputs differ: {input_checks}")
    if not args.smoke and (
        len(training.get("rounds", [])) != 8
        or training.get("final_policy_rule") != "round 8 actor, never best-so-far"
        or training.get("candidate_screen_or_confirmation_count") != 0
        or training.get("performance_rollbacks") != 0
    ):
        raise RuntimeError("v24 final checkpoint is not the fixed round-8 actor")

    batch_size = 4 if args.smoke else EVAL_BATCH_SIZE
    target_episodes = 4 if args.smoke else FINAL_TARGET_EPISODES
    d0_episodes = 4 if args.smoke else FINAL_D0_EPISODES
    target_repeats = target_episodes // batch_size
    d0_repeats = d0_episodes // batch_size
    target_seed = 199_250_001 if args.smoke else FINAL_TARGET_SEED
    d0_seed = 199_250_101 if args.smoke else FINAL_D0_SEED
    bootstrap_samples = 100 if args.smoke else REPORT_BOOTSTRAP_SAMPLES

    sys.path.insert(0, str(repo))
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import src.tasks  # noqa: F401
    from src.tasks.stairs_cbf.proximal import CbfProximalRefinementRunner
    from src.tasks.stairs_cbf.proximal_context import apply_cbf_proximal_context

    env_cfg = load_env_cfg("Unitree-G1-Stairs-Online-DQHMED")
    apply_cbf_proximal_context(env_cfg, context, role="target")
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg("Unitree-G1-Stairs-Online-DQHMED")
    agent_cfg.num_steps_per_env = 8
    _configure_algorithm(agent_cfg)
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = CbfProximalRefinementRunner(
        env, asdict(agent_cfg), log_dir=None, device=args.device
    )
    try:
        runner.load_initial_checkpoint(str(checkpoint), map_location=args.device)
        actors = {"baseline": actor_state(runner.alg.actor)}
        final_payload = torch.load(
            final_checkpoint, map_location=args.device, weights_only=False
        )
        runner.alg.actor.load_state_dict(final_payload["actor_state_dict"], strict=True)
        actors["final"] = actor_state(runner.alg.actor)
        if actor_state_sha256(actors["baseline"]) != training["initial_actor_sha256"]:
            raise RuntimeError("v24 audit baseline differs from training pi0")
        if actor_state_sha256(actors["final"]) != training["final_actor_sha256"]:
            raise RuntimeError("v24 audit actor differs from fixed round 8")
        evaluations: dict[str, dict[str, Any]] = {}
        for role, actor in actors.items():
            evaluations[f"{role}_target"] = _evaluate_state(
                runner,
                actor,
                task="Unitree-G1-Stairs-Online-DQHMED",
                num_envs=batch_size,
                repeats=target_repeats,
                seed=target_seed,
                device=args.device,
                artifact_dir=output_dir / "raw" / f"{role}_target",
                repo=repo,
                context=context_path,
                resume=args.resume,
            )
            evaluations[f"{role}_D0"] = _evaluate_state(
                runner,
                actor,
                task="Unitree-G1-Stairs-Online-D0",
                num_envs=batch_size,
                repeats=d0_repeats,
                seed=d0_seed,
                device=args.device,
                artifact_dir=output_dir / "raw" / f"{role}_D0",
                repo=repo,
                context=None,
                resume=args.resume,
            )
    finally:
        env.close()

    for domain in ("target", "D0"):
        old = evaluations[f"baseline_{domain}"]
        new = evaluations[f"final_{domain}"]
        if old["initial_state_signatures"] != new["initial_state_signatures"]:
            raise RuntimeError(f"v24 {domain} initial conditions are not paired")
        if not (
            old["runtime_filter"]
            and new["runtime_filter"]
            and old["original_observation_interface"]
            and new["original_observation_interface"]
        ):
            raise RuntimeError(f"v24 {domain} changed CBF or actor interface")

    target_seeds = [target_seed + index for index in range(target_repeats)]
    d0_seeds = [d0_seed + index for index in range(d0_repeats)]
    paired_target = _paired_rows(
        "target",
        _read_episode_rows(output_dir / "raw/baseline_target", "DQHMED", target_seeds),
        _read_episode_rows(output_dir / "raw/final_target", "DQHMED", target_seeds),
    )
    paired_d0 = _paired_rows(
        "D0",
        _read_episode_rows(output_dir / "raw/baseline_D0", "D0", d0_seeds),
        _read_episode_rows(output_dir / "raw/final_D0", "D0", d0_seeds),
    )
    paired = paired_target + paired_d0
    _write_paired_csv(output_dir / "paired_episode_metrics.csv", paired)
    target_report = _domain_report(
        paired_target,
        bootstrap_seed=REPORT_BOOTSTRAP_SEEDS["target"],
        bootstrap_samples=bootstrap_samples,
    )
    d0_report = _domain_report(
        paired_d0,
        bootstrap_seed=REPORT_BOOTSTRAP_SEEDS["D0"],
        bootstrap_samples=bootstrap_samples,
    )
    gate = development_gate(
        target_success_delta=target_report["success"]["delta"],
        target_fall_delta=target_report["fall"]["delta"],
        d0_success_delta=d0_report["success"]["delta"],
    )
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": EXPERIMENT_NAME,
        "policy_method": POLICY_METHOD,
        "smoke": args.smoke,
        "input_checks": input_checks,
        "context": {
            "file": str(context_path),
            "file_sha256": file_sha256(context_path),
            "parameters_sha256": context["parameters_sha256"],
            "selected_foot_friction": context["calibration"]["selected_foot_friction"],
            "pure_contact_audit": context_audit,
        },
        "checkpoint": {
            "path": str(final_checkpoint),
            "sha256": file_sha256(final_checkpoint),
            "actor_sha256": training["final_actor_sha256"],
            "fixed_round": 8 if not args.smoke else len(training["rounds"]),
            "performance_selected": False,
        },
        "paired_evaluation": {
            "runtime_cbf": True,
            "deterministic_policy_mean": True,
            "original_actor_observation_interface": True,
            "target_episodes": len(paired_target),
            "D0_episodes": len(paired_d0),
            "target_seed_start": target_seed,
            "D0_seed_start": d0_seed,
            "base_and_final_initial_conditions_identical": True,
            "confidence_intervals_are_report_only": True,
        },
        "target": target_report,
        "D0": d0_report,
        "development_gate": gate,
        "independent_contact_context_run": True,
        "v23_lateral_result_modified_or_recomputed": False,
        "joint_lateral_contact_gate": False,
        "raw_evaluations": evaluations,
    }
    _write_json(output_dir / "final_test.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

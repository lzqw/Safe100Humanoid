"""Run the frozen four-condition paired v25 final evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from cbf_teacher_v25_protocol import (
    BASE_CHECKPOINT_SHA256,
    ENVIRONMENT_VARIANT,
    EVAL_BATCH_SIZE,
    FINAL_EPISODES,
    FINAL_REPEATS,
    POLICY_METHOD,
    PROTOCOL_ID,
    SOURCE_FILES,
    TASK_ID,
    development_gate,
    final_evaluation_seed,
    paired_repair_regression_counts,
    validate_v25_calibrated_context,
)
from proximal_v23_io import file_sha256

CONDITIONS = (
    ("pi0_off", "base", "off"),
    ("pi0_on", "base", "on"),
    ("pi8_on", "final", "on"),
    ("pi8_off", "final", "off"),
)
METRICS = (
    "success_rate",
    "fall_rate",
    "timeout_rate",
    "kick_rate",
    "mean_kick_count",
    "mean_return",
    "mean_reached_riser",
    "mean_correction_norm",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"invalid CSV boolean: {value!r}")
    return normalized == "true"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _evaluate(
    *,
    repo: Path,
    checkpoint: Path,
    gain: float,
    filter_mode: str,
    condition: str,
    output_dir: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries = []
    rows = []
    for repeat in range(FINAL_REPEATS):
        seed = final_evaluation_seed(repeat)
        run_dir = output_dir / "raw" / condition / f"seed_{seed}"
        output_json = run_dir / "summary.json"
        output_csv = run_dir / "episodes.csv"
        summary = None
        if resume and output_json.is_file() and output_csv.is_file():
            candidate = json.loads(output_json.read_text())
            if (
                candidate.get("seed") == seed
                and candidate.get("task") == TASK_ID
                and candidate.get("environment_variant") == ENVIRONMENT_VARIANT
                and candidate.get("num_envs") == EVAL_BATCH_SIZE
                and candidate.get("num_episodes") == EVAL_BATCH_SIZE
                and candidate.get("runtime_filter") == (filter_mode == "on")
                and candidate.get("swing_underresponse_gain") == gain
                and candidate.get("checkpoint_sha256") == file_sha256(checkpoint)
            ):
                summary = candidate
        if summary is None:
            command = [
                sys.executable,
                str(repo / "experiments/scripts/evaluate_cbf_teacher_v25.py"),
                "--repo",
                str(repo),
                "--checkpoint",
                str(checkpoint),
                "--gain",
                str(gain),
                "--runtime-filter",
                filter_mode,
                "--num-envs",
                str(EVAL_BATCH_SIZE),
                "--num-episodes",
                str(EVAL_BATCH_SIZE),
                "--seed",
                str(seed),
                "--device",
                device,
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
            completed = subprocess.run(
                command, cwd=repo, check=False, capture_output=True, text=True
            )
            if completed.returncode:
                diagnostic = "\n".join(
                    (completed.stdout + "\n" + completed.stderr).splitlines()[-160:]
                )
                raise RuntimeError(
                    f"v25 final condition {condition} failed:\n{diagnostic}"
                )
            summary = json.loads(output_json.read_text())
        with output_csv.open(newline="") as handle:
            batch_rows = list(csv.DictReader(handle))
        if len(batch_rows) != EVAL_BATCH_SIZE:
            raise RuntimeError(f"v25 condition {condition} has incomplete episode rows")
        summaries.append(summary)
        rows.extend(batch_rows)

    aggregate: dict[str, Any] = {
        "condition": condition,
        "policy": condition.split("_")[0],
        "runtime_filter": filter_mode == "on",
        "num_episodes": len(rows),
        "seeds": [final_evaluation_seed(repeat) for repeat in range(FINAL_REPEATS)],
        "initial_state_signatures": [
            summary["initial_state_signature"] for summary in summaries
        ],
        "actor_state_sha256": summaries[0]["actor_state_sha256"],
        "checkpoint_sha256": summaries[0]["checkpoint_sha256"],
        "deterministic_policy_mean": all(
            summary.get("deterministic_policy_mean") is True for summary in summaries
        ),
        "one_initial_episode_per_env": all(
            summary.get("one_initial_episode_per_env") is True for summary in summaries
        ),
        "original_observation_interface": all(
            summary.get("original_observation_interface") is True
            for summary in summaries
        ),
        "actor_observation_dim": summaries[0]["actor_observation_dim"],
        "teacher_reprojection_max_abs_error": max(
            float(summary["teacher_reprojection_max_abs_error"])
            for summary in summaries
        ),
        "swing_selection_mismatch_count": sum(
            int(summary["swing_selection_mismatch_count"]) for summary in summaries
        ),
    }
    if any(
        summary["actor_state_sha256"] != aggregate["actor_state_sha256"]
        for summary in summaries
    ):
        raise RuntimeError(f"v25 condition {condition} changed actors between batches")
    if any(
        summary["checkpoint_sha256"] != aggregate["checkpoint_sha256"]
        or summary["actor_observation_dim"] != aggregate["actor_observation_dim"]
        for summary in summaries
    ):
        raise RuntimeError(
            f"v25 condition {condition} changed checkpoint or actor interface"
        )
    for metric in METRICS:
        values = [float(summary[metric]) for summary in summaries]
        aggregate[metric] = sum(values) / len(values)
        aggregate[f"{metric}_std_across_batches"] = (
            math.sqrt(
                sum((value - aggregate[metric]) ** 2 for value in values)
                / (len(values) - 1)
            )
            if len(values) > 1
            else 0.0
        )
    aggregate["total_reached_risers"] = sum(
        int(summary["total_reached_risers"]) for summary in summaries
    )
    aggregate["total_intervention_count"] = sum(
        int(summary["total_intervention_count"]) for summary in summaries
    )
    aggregate["total_would_intervene_count"] = sum(
        int(summary["total_would_intervene_count"]) for summary in summaries
    )
    aggregate["intervention_per_riser"] = aggregate["total_intervention_count"] / max(
        1, aggregate["total_reached_risers"]
    )
    aggregate["would_intervene_per_riser"] = aggregate[
        "total_would_intervene_count"
    ] / max(1, aggregate["total_reached_risers"])
    aggregate["success_count"] = sum(_bool(row["success"]) for row in rows)
    aggregate["failure_count"] = len(rows) - aggregate["success_count"]
    aggregate["fall_count"] = sum(_bool(row["fell"]) for row in rows)
    aggregate["timeout_count"] = sum(_bool(row["timed_out"]) for row in rows)
    aggregate["kick_episode_count"] = sum(_bool(row["toe_riser_kick"]) for row in rows)
    aggregate["toe_riser_failure_count"] = sum(
        (not _bool(row["success"])) and _bool(row["toe_riser_kick"]) for row in rows
    )
    aggregate["alignment_coverage"] = aggregate["toe_riser_failure_count"] / max(
        1, aggregate["failure_count"]
    )
    return aggregate, rows


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    final_checkpoint = args.final_checkpoint.resolve()
    training_path = args.training_summary.resolve()
    context_path = args.context.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout:
        raise RuntimeError("v25 final audit requires a clean committed worktree")
    for path in (
        base_checkpoint,
        final_checkpoint,
        training_path,
        context_path,
        protocol_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(base_checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v25 final audit base checkpoint differs from pi0")
    context = validate_v25_calibrated_context(json.loads(context_path.read_text()))
    training = json.loads(training_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if (
        training.get("protocol_id") != PROTOCOL_ID
        or protocol.get("protocol_id") != PROTOCOL_ID
    ):
        raise RuntimeError("v25 final audit inputs have the wrong protocol")
    if training.get("final_checkpoint_sha256") != file_sha256(final_checkpoint):
        raise RuntimeError("v25 final audit checkpoint differs from fixed round 8")
    if training.get("final_policy_rule") != "round 8 actor, never best-so-far":
        raise RuntimeError(
            "v25 final audit was not given the unconditional round-8 actor"
        )
    if training.get("protocol", {}).get("sha256") != file_sha256(protocol_path):
        raise RuntimeError("v25 final audit protocol differs from the training freeze")
    if protocol.get("implementation_boundary", {}).get("git_commit") != training.get(
        "protocol", {}
    ).get("implementation_commit"):
        raise RuntimeError(
            "v25 final audit implementation boundary differs from training"
        )
    bound_sources = protocol.get("implementation_boundary", {}).get("source_files", {})
    if set(bound_sources) != set(SOURCE_FILES) or any(
        not (repo / relative).is_file()
        or file_sha256(repo / relative) != bound_sources.get(relative)
        for relative in SOURCE_FILES
    ):
        raise RuntimeError("v25 final evaluation source differs from formal freeze")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = output_dir / "final_evaluation_started.json"
    if started.exists() and not args.resume:
        raise RuntimeError("v25 final evaluation already started; use --resume")
    if not started.exists():
        _atomic_json(
            started,
            {
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": file_sha256(protocol_path),
                "training_summary_sha256": file_sha256(training_path),
                "base_checkpoint_sha256": file_sha256(base_checkpoint),
                "final_checkpoint_sha256": file_sha256(final_checkpoint),
                "condition_order": [condition[0] for condition in CONDITIONS],
                "fresh_condition_count": FINAL_EPISODES,
            },
        )

    checkpoints = {"base": base_checkpoint, "final": final_checkpoint}
    aggregates: dict[str, dict[str, Any]] = {}
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    gain = float(context["shift"]["swing_underresponse_gain"])
    for condition, policy, filter_mode in CONDITIONS:
        aggregates[condition], condition_rows[condition] = _evaluate(
            repo=repo,
            checkpoint=checkpoints[policy],
            gain=gain,
            filter_mode=filter_mode,
            condition=condition,
            output_dir=output_dir,
            device=args.device,
            resume=args.resume,
        )
        print(json.dumps(aggregates[condition], sort_keys=True), flush=True)

    signatures = [
        aggregates[condition]["initial_state_signatures"]
        for condition, _, _ in CONDITIONS
    ]
    if any(value != signatures[0] for value in signatures[1:]):
        raise RuntimeError("v25 four final conditions do not share initial states")

    def identity(row: dict[str, str]) -> tuple[int, int]:
        return int(row["evaluation_seed"]), int(row["environment_id"])

    for rows in condition_rows.values():
        rows.sort(key=identity)
    identities = [
        [identity(row) for row in condition_rows[name]] for name, _, _ in CONDITIONS
    ]
    if any(value != identities[0] for value in identities[1:]):
        raise RuntimeError("v25 four-condition episode identities differ")
    if (
        len(identities[0]) != FINAL_EPISODES
        or len(set(identities[0])) != FINAL_EPISODES
    ):
        raise RuntimeError("v25 final episode identities are incomplete or duplicated")

    paired_rows = []
    for index in range(FINAL_EPISODES):
        source = condition_rows["pi0_off"][index]
        row: dict[str, Any] = {
            "pair_index": index,
            "evaluation_seed": source["evaluation_seed"],
            "environment_id": source["environment_id"],
        }
        for condition, _, _ in CONDITIONS:
            episode = condition_rows[condition][index]
            for field in (
                "success",
                "fell",
                "timed_out",
                "failure_type",
                "toe_riser_kick",
                "toe_riser_kick_count",
                "return",
                "max_riser",
                "intervention_count",
                "intervention_per_riser",
                "would_intervene_count",
                "would_intervene_per_riser",
                "mean_correction_norm",
            ):
                row[f"{condition}_{field}"] = episode[field]
        paired_rows.append(row)
    paired_path = output_dir / "paired_episode_metrics.csv"
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    base_off_success = [_bool(row["pi0_off_success"]) for row in paired_rows]
    base_on_success = [_bool(row["pi0_on_success"]) for row in paired_rows]
    final_on_success = [_bool(row["pi8_on_success"]) for row in paired_rows]
    final_off_success = [_bool(row["pi8_off_success"]) for row in paired_rows]
    off_success_delta = (
        aggregates["pi8_off"]["success_rate"] - aggregates["pi0_off"]["success_rate"]
    )
    on_success_delta = (
        aggregates["pi8_on"]["success_rate"] - aggregates["pi0_on"]["success_rate"]
    )
    gate = development_gate(
        off_success_delta=off_success_delta,
        on_success_delta=on_success_delta,
        base_off_kick_rate=aggregates["pi0_off"]["kick_rate"],
        final_off_kick_rate=aggregates["pi8_off"]["kick_rate"],
        base_on_intervention_per_riser=aggregates["pi0_on"]["intervention_per_riser"],
        final_on_intervention_per_riser=aggregates["pi8_on"]["intervention_per_riser"],
    )
    base_off_failure_count = sum(not success for success in base_off_success)
    fresh_rescue_count = sum(
        (not off) and on
        for off, on in zip(base_off_success, base_on_success, strict=True)
    )
    final = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "policy_method": POLICY_METHOD,
        "context": {
            "file": str(context_path),
            "file_sha256": file_sha256(context_path),
            "parameters_sha256": context["parameters_sha256"],
            "swing_underresponse_gain": gain,
        },
        "checkpoints": {
            "pi0_sha256": file_sha256(base_checkpoint),
            "pi8_sha256": file_sha256(final_checkpoint),
            "pi0_actor_sha256": aggregates["pi0_off"]["actor_state_sha256"],
            "pi8_actor_sha256": aggregates["pi8_off"]["actor_state_sha256"],
            "pi0_actor_same_off_on": aggregates["pi0_off"]["actor_state_sha256"]
            == aggregates["pi0_on"]["actor_state_sha256"],
            "pi8_actor_same_off_on": aggregates["pi8_off"]["actor_state_sha256"]
            == aggregates["pi8_on"]["actor_state_sha256"],
        },
        "paired_evaluation": {
            "conditions": [condition[0] for condition in CONDITIONS],
            "conditions_per_arm": FINAL_EPISODES,
            "same_initial_conditions_all_four_arms": True,
            "initial_state_signatures": signatures[0],
            "deterministic_policy_mean": True,
            "original_actor_observation_interface": True,
            "confidence_intervals_are_gates": False,
        },
        "conditions": aggregates,
        "primary_outcomes": {
            "shielded_task_delta": on_success_delta,
            "internalization_delta": off_success_delta,
            "off_kick_rate_delta": aggregates["pi8_off"]["kick_rate"]
            - aggregates["pi0_off"]["kick_rate"],
            "on_intervention_per_riser_delta": aggregates["pi8_on"][
                "intervention_per_riser"
            ]
            - aggregates["pi0_on"]["intervention_per_riser"],
            "on_intervention_per_riser_relative_reduction": gate[
                "intervention_per_riser_relative_reduction"
            ],
        },
        "fresh_pi0_rescue_audit": {
            "base_off_failure_count": base_off_failure_count,
            "off_failure_to_on_success_count": fresh_rescue_count,
            "shield_rescue_rate": fresh_rescue_count / max(1, base_off_failure_count),
            "alignment_coverage": aggregates["pi0_off"]["alignment_coverage"],
        },
        "paired_changes": {
            "off": paired_repair_regression_counts(base_off_success, final_off_success),
            "on": paired_repair_regression_counts(base_on_success, final_on_success),
        },
        "development_gate": gate,
        "interpretation": (
            "development success" if gate["passed"] else "development gate not met"
        ),
        "final_policy_rule": "round 8 actor, never best-so-far",
        "training_summary_sha256": file_sha256(training_path),
        "protocol_sha256": file_sha256(protocol_path),
        "paired_episode_metrics_sha256": file_sha256(paired_path),
    }
    _atomic_json(output_dir / "final_test.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

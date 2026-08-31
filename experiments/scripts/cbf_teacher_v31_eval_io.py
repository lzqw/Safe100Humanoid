"""Shared sequential evaluation and paired aggregation helpers for v31."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from proximal_v23_io import file_sha256

EPISODE_FIELDS = (
    "success",
    "fell",
    "timed_out",
    "failure_type",
    "toe_riser_kick",
    "toe_riser_kick_count",
    "unsafe_overlap_steps",
    "return",
    "steps",
    "completion_time_s",
    "max_riser",
    "completion_fraction",
    "intervention_count",
    "would_intervene_count",
    "nominal_barrier_violation_steps",
    "mean_correction_norm",
    "mean_counterfactual_correction_norm",
    "minimum_nominal_barrier_margin",
)


def bool_value(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"invalid CSV boolean {value!r}")
    return normalized == "true"


def identity(row: dict[str, str]) -> tuple[int, int]:
    return int(row["evaluation_seed"]), int(row["environment_id"])


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty v31 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_condition(
    *,
    repo: Path,
    protocol: Path,
    checkpoint: Path,
    context: str,
    condition: str,
    runtime_filter: str,
    episodes: int,
    batch_size: int,
    seed_base: int,
    output_root: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if episodes < 1 or episodes % batch_size:
        raise ValueError("v31 episodes must divide exactly by evaluation batch")
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    checkpoint_hash = file_sha256(checkpoint)
    for repeat in range(episodes // batch_size):
        seed = seed_base + repeat
        run_dir = output_root / "raw" / context / condition / f"seed_{seed}"
        output_json = run_dir / "summary.json"
        output_csv = run_dir / "episodes.csv"
        summary = None
        if resume and output_json.is_file() and output_csv.is_file():
            candidate = json.loads(output_json.read_text())
            if (
                candidate.get("context") == context
                and candidate.get("seed") == seed
                and candidate.get("num_envs") == batch_size
                and candidate.get("num_episodes") == batch_size
                and candidate.get("runtime_filter") == (runtime_filter == "on")
                and candidate.get("checkpoint_sha256") == checkpoint_hash
            ):
                summary = candidate
        if summary is None:
            command = [
                sys.executable,
                str(repo / "experiments/scripts/evaluate_cbf_teacher_v31.py"),
                "--repo",
                str(repo),
                "--protocol",
                str(protocol),
                "--checkpoint",
                str(checkpoint),
                "--context",
                context,
                "--runtime-filter",
                runtime_filter,
                "--num-envs",
                str(batch_size),
                "--num-episodes",
                str(batch_size),
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
                command, cwd=repo, capture_output=True, text=True, check=False
            )
            if completed.returncode:
                diagnostic = "\n".join(
                    (completed.stdout + "\n" + completed.stderr).splitlines()[-160:]
                )
                raise RuntimeError(
                    f"v31 evaluation failed for {context}/{condition}:\n{diagnostic}"
                )
            summary = json.loads(output_json.read_text())
        with output_csv.open(newline="") as handle:
            batch_rows = list(csv.DictReader(handle))
        if len(batch_rows) != batch_size:
            raise RuntimeError(f"incomplete v31 rows for {context}/{condition}")
        summaries.append(summary)
        rows.extend(batch_rows)
    rows.sort(key=identity)
    if len({identity(row) for row in rows}) != len(rows):
        raise RuntimeError(f"duplicate v31 identity in {context}/{condition}")
    aggregate = aggregate_condition(
        context=context,
        condition=condition,
        runtime_filter=runtime_filter == "on",
        summaries=summaries,
        rows=rows,
    )
    return aggregate, rows


def aggregate_condition(
    *,
    context: str,
    condition: str,
    runtime_filter: bool,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not summaries or not rows:
        raise ValueError("v31 aggregate inputs must be non-empty")
    total_risers = sum(int(row["max_riser"]) for row in rows)
    total_steps = sum(int(row["steps"]) for row in rows)
    total_interventions = sum(int(row["intervention_count"]) for row in rows)
    total_would = sum(int(row["would_intervene_count"]) for row in rows)
    total_kicks = sum(int(row["toe_riser_kick_count"]) for row in rows)
    total_overlap = sum(int(row["unsafe_overlap_steps"]) for row in rows)
    total_violations = sum(int(row["nominal_barrier_violation_steps"]) for row in rows)
    successful_times = [
        float(row["completion_time_s"]) for row in rows if bool_value(row["success"])
    ]
    aggregate = {
        "context": context,
        "condition": condition,
        "runtime_filter": runtime_filter,
        "num_episodes": len(rows),
        "seeds": [int(item["seed"]) for item in summaries],
        "initial_state_signatures": [
            item["initial_state_signature"] for item in summaries
        ],
        "checkpoint_sha256": summaries[0]["checkpoint_sha256"],
        "actor_state_sha256": summaries[0]["actor_state_sha256"],
        "actor_deterministic_state_sha256": summaries[0][
            "actor_deterministic_state_sha256"
        ],
        "success_count": sum(bool_value(row["success"]) for row in rows),
        "success_rate": sum(bool_value(row["success"]) for row in rows) / len(rows),
        "fall_count": sum(bool_value(row["fell"]) for row in rows),
        "fall_rate": sum(bool_value(row["fell"]) for row in rows) / len(rows),
        "kick_episode_count": sum(bool_value(row["toe_riser_kick"]) for row in rows),
        "kick_episode_rate": sum(bool_value(row["toe_riser_kick"]) for row in rows)
        / len(rows),
        "mean_return": float(np.mean([float(row["return"]) for row in rows])),
        "mean_reached_riser": float(np.mean([float(row["max_riser"]) for row in rows])),
        "mean_completion_time_s": float(
            np.mean([float(row["completion_time_s"]) for row in rows])
        ),
        "mean_success_completion_time_s": (
            float(np.mean(successful_times)) if successful_times else None
        ),
        "total_steps": total_steps,
        "total_reached_risers": total_risers,
        "intervention_steps_per_riser": total_interventions / max(1, total_risers),
        "counterfactual_would_intervene_fraction": total_would / max(1, total_steps),
        "mean_correction_norm": sum(
            float(row["mean_correction_norm"]) * int(row["steps"]) for row in rows
        )
        / max(1, total_steps),
        "mean_counterfactual_correction_norm": sum(
            float(row["mean_counterfactual_correction_norm"]) * int(row["steps"])
            for row in rows
        )
        / max(1, total_steps),
        "toe_riser_kick_events_per_riser": total_kicks / max(1, total_risers),
        "unsafe_overlap_steps_per_riser": total_overlap / max(1, total_risers),
        "nominal_barrier_violation_steps_per_riser": total_violations
        / max(1, total_risers),
        "mean_episode_minimum_nominal_barrier_margin": float(
            np.mean([float(row["minimum_nominal_barrier_margin"]) for row in rows])
        ),
        "global_minimum_nominal_barrier_margin": min(
            float(row["minimum_nominal_barrier_margin"]) for row in rows
        ),
        "teacher_reprojection_max_abs_error": max(
            float(item["teacher_reprojection_max_abs_error"]) for item in summaries
        ),
        "swing_selection_mismatch_count": sum(
            int(item["swing_selection_mismatch_count"]) for item in summaries
        ),
    }
    if any(
        item["checkpoint_sha256"] != aggregate["checkpoint_sha256"]
        or item["actor_state_sha256"] != aggregate["actor_state_sha256"]
        for item in summaries
    ):
        raise RuntimeError(f"v31 actor changed inside {context}/{condition}")
    return aggregate


def assert_paired(
    aggregates: dict[str, dict[str, Any]],
    rows: dict[str, list[dict[str, str]]],
    *,
    label: str,
) -> None:
    identity_lists = [[identity(row) for row in value] for value in rows.values()]
    if any(value != identity_lists[0] for value in identity_lists[1:]):
        raise RuntimeError(f"v31 paired identities differ for {label}")
    signatures = [item["initial_state_signatures"] for item in aggregates.values()]
    if any(value != signatures[0] for value in signatures[1:]):
        raise RuntimeError(f"v31 initial state signatures differ for {label}")


def numeric_column(rows: list[dict[str, str]], field: str) -> np.ndarray:
    if field in ("success", "fell", "timed_out", "toe_riser_kick"):
        return np.asarray([float(bool_value(row[field])) for row in rows])
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def paired_ci(
    baseline: list[dict[str, str]],
    final: list[dict[str, str]],
    *,
    field: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    if len(baseline) != len(final) or not baseline:
        raise ValueError("v31 paired CI requires equal non-empty rows")
    delta = numeric_column(final, field) - numeric_column(baseline, field)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "field": field,
        "paired_count": len(delta),
        "point_estimate": float(delta.mean()),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "used_as_selection_or_stopping_gate": False,
    }


def paired_repairs_regressions(
    baseline: list[dict[str, str]], final: list[dict[str, str]]
) -> dict[str, int]:
    if len(baseline) != len(final):
        raise ValueError("v31 repair counts require paired rows")
    base_success = [bool_value(row["success"]) for row in baseline]
    final_success = [bool_value(row["success"]) for row in final]
    return {
        "repairs": sum(
            (not before) and after
            for before, after in zip(base_success, final_success, strict=True)
        ),
        "regressions": sum(
            before and (not after)
            for before, after in zip(base_success, final_success, strict=True)
        ),
    }


def paired_wide_rows(
    *,
    domain: str,
    conditions: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    first = next(iter(conditions.values()))
    output_rows = []
    for index, source in enumerate(first):
        output: dict[str, Any] = {
            "domain": domain,
            "pair_index": index,
            "evaluation_seed": source["evaluation_seed"],
            "environment_id": source["environment_id"],
        }
        for condition, condition_rows in conditions.items():
            episode = condition_rows[index]
            for field in EPISODE_FIELDS:
                output[f"{condition}_{field}"] = episode[field]
        output_rows.append(output)
    return output_rows

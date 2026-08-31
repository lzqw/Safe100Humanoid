"""Run the frozen two-stage v34 CBF outcome search."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from proximal_v23_io import file_sha256
from velocity_cbf_v34_protocol import (
    FORMAL_CONTEXTS,
    OPTIMIZED_CBF_MODE,
    PARAMETER_RANGES,
    PROTOCOL_ID,
    STAGE1_EPISODES,
    STAGE2_EPISODES,
    STAGE2_TOP_K,
    TRAIN_TOP_K,
    V31_CHECKPOINT_SHA256,
    candidate_grid,
    stage1_seed,
    stage2_seed,
)

PARAMETER_NAMES = tuple(PARAMETER_RANGES)
AUXILIARY_FIELDS = (
    "success_rate",
    "fall_rate",
    "mean_return",
    "mean_reached_riser",
    "mean_completion_time_s",
    "intervention_steps_per_riser",
    "mean_velocity_correction_norm",
    "mean_velocity_correction_jerk",
    "mean_toe_riser_contact_impulse",
    "unsafe_overlap_steps_per_riser",
    "post_intervention_fall_rate",
    "post_intervention_mean_support_foot_slip",
    "mean_cbf_compute_time_ms",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--v31-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("v34 search CSV cannot be empty")
    fields = sorted({field for row in rows for field in row})
    leading = [
        field
        for field in ("rank", "candidate", "candidate_index", "status", "mean_success")
        if field in fields
    ]
    fields = leading + [field for field in fields if field not in leading]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _checkpoint(root: Path, context: str) -> Path:
    checkpoint = root / context / "A2" / "round_08.pt"
    if file_sha256(checkpoint) != V31_CHECKPOINT_SHA256[context]["A2"]:
        raise RuntimeError(f"v34 v31 A2 checkpoint differs for {context}")
    return checkpoint


def _parameters(candidate: dict[str, Any]) -> dict[str, float]:
    return {name: float(candidate[name]) for name in PARAMETER_NAMES}


def _evaluate(
    *,
    repo: Path,
    config: Path,
    checkpoint: Path,
    context: str,
    candidate: dict[str, Any],
    episodes: int,
    seed: int,
    run_dir: Path,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "episodes.csv"
    valid = False
    if resume and summary_path.is_file() and rows_path.is_file():
        prior = json.loads(summary_path.read_text())
        valid = (
            prior.get("protocol_id") == PROTOCOL_ID
            and prior.get("candidate") == candidate["candidate"]
            and prior.get("seed") == seed
            and prior.get("num_episodes") == episodes
            and prior.get("checkpoint_sha256") == file_sha256(checkpoint)
            and prior.get("runtime_filter") is True
        )
    if not valid:
        command = [
            sys.executable,
            str(repo / "experiments/scripts/evaluate_velocity_cbf_v34.py"),
            "--repo",
            str(repo),
            "--search-config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--context",
            context,
            "--cbf-mode",
            str(candidate["mode"]),
            "--runtime-filter",
            "on",
            "--num-envs",
            str(episodes),
            "--num-episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--policy-label",
            "v31_A2_round_8",
            "--candidate",
            str(candidate["candidate"]),
            "--device",
            device,
            "--output-json",
            str(summary_path),
            "--output-csv",
            str(rows_path),
        ]
        if candidate["mode"] == OPTIMIZED_CBF_MODE:
            command.extend(["--parameters-json", json.dumps(_parameters(candidate))])
        completed = subprocess.run(
            command, cwd=repo, capture_output=True, text=True, check=False
        )
        if completed.returncode:
            diagnostic = "\n".join(
                (completed.stdout + completed.stderr).splitlines()[-100:]
            )
            raise RuntimeError(diagnostic)
    summary = json.loads(summary_path.read_text())
    if summary.get("num_episodes") != episodes:
        raise RuntimeError("v34 search evaluation is incomplete")
    return summary


def _candidate_row(
    candidate: dict[str, Any], summaries: dict[str, dict[str, Any]], elapsed: float
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": candidate["candidate"],
        "candidate_index": candidate["candidate_index"],
        "mode": candidate["mode"],
        "status": "completed",
        "elapsed_seconds": elapsed,
        **_parameters(candidate),
    }
    for context, summary in summaries.items():
        for field in AUXILIARY_FIELDS:
            row[f"{context}_{field}"] = summary.get(field)
    row["mean_success"] = sum(
        float(summaries[context]["success_rate"]) for context in FORMAL_CONTEXTS
    ) / len(FORMAL_CONTEXTS)
    return row


def _run_stage(
    *,
    name: str,
    candidates: list[dict[str, Any]],
    episodes: int,
    seed_function,
    repo: Path,
    config: Path,
    v31_root: Path,
    output_root: Path,
    device: str,
    resume: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, 1):
        started = time.monotonic()
        summaries: dict[str, dict[str, Any]] = {}
        try:
            for context in FORMAL_CONTEXTS:
                summaries[context] = _evaluate(
                    repo=repo,
                    config=config,
                    checkpoint=_checkpoint(v31_root, context),
                    context=context,
                    candidate=candidate,
                    episodes=episodes,
                    seed=seed_function(context),
                    run_dir=output_root
                    / "raw"
                    / name
                    / candidate["candidate"]
                    / context,
                    device=device,
                    resume=resume,
                )
            row = _candidate_row(candidate, summaries, time.monotonic() - started)
        except Exception as error:  # noqa: BLE001 - protocol records and skips failures
            row = {
                "candidate": candidate["candidate"],
                "candidate_index": candidate["candidate_index"],
                "mode": candidate["mode"],
                "status": "failed",
                "error": str(error)[-4000:],
                "elapsed_seconds": time.monotonic() - started,
                "mean_success": -1.0,
                **_parameters(candidate),
            }
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": name,
                    "progress": f"{position}/{len(candidates)}",
                    "candidate": candidate["candidate"],
                    "status": row["status"],
                    "mean_success": row["mean_success"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    successful = [row for row in rows if row["status"] == "completed"]
    successful.sort(
        key=lambda row: (-float(row["mean_success"]), int(row["candidate_index"]))
    )
    ranks = {row["candidate"]: rank for rank, row in enumerate(successful, 1)}
    for row in rows:
        row["rank"] = ranks.get(row["candidate"], "")
    return rows


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    config_path = args.search_config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("v34 search config differs")
    candidates = candidate_grid()
    if config["candidate_generator"]["candidates"] != candidates:
        raise RuntimeError("v34 frozen candidates differ from implementation")
    output = args.output_root.resolve()
    if output.exists() and not args.resume:
        raise RuntimeError(
            "v34 search output exists; use --resume only for infrastructure recovery"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage1 = _run_stage(
        name="stage1",
        candidates=candidates,
        episodes=STAGE1_EPISODES,
        seed_function=stage1_seed,
        repo=repo,
        config=config_path,
        v31_root=args.v31_root.resolve(),
        output_root=output,
        device=args.device,
        resume=args.resume,
    )
    _write_csv(output / "all_candidates.csv", stage1)
    ranked1 = sorted(
        (row for row in stage1 if row["status"] == "completed"),
        key=lambda row: int(row["rank"]),
    )
    if len(ranked1) < STAGE2_TOP_K:
        raise RuntimeError("fewer than eight v34 candidates completed stage 1")
    by_name = {candidate["candidate"]: candidate for candidate in candidates}
    top8 = [by_name[row["candidate"]] for row in ranked1[:STAGE2_TOP_K]]
    stage2 = _run_stage(
        name="stage2",
        candidates=top8,
        episodes=STAGE2_EPISODES,
        seed_function=stage2_seed,
        repo=repo,
        config=config_path,
        v31_root=args.v31_root.resolve(),
        output_root=output,
        device=args.device,
        resume=args.resume,
    )
    _write_csv(output / "top8_results.csv", stage2)
    ranked2 = sorted(
        (row for row in stage2 if row["status"] == "completed"),
        key=lambda row: int(row["rank"]),
    )
    if len(ranked2) < TRAIN_TOP_K:
        raise RuntimeError("fewer than two v34 candidates completed stage 2")
    top2 = [by_name[row["candidate"]] for row in ranked2[:TRAIN_TOP_K]]
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "top_two_frozen_for_six_training_runs",
        "selection_objective": "stage2 mean F1/F2/F3 CBF-on success only",
        "top2": top2,
        "stage2_rows": ranked2[:TRAIN_TOP_K],
        "search_elapsed_seconds": time.monotonic() - started,
        "failed_stage1_candidates": sum(row["status"] != "completed" for row in stage1),
        "failed_stage2_candidates": sum(row["status"] != "completed" for row in stage2),
        "final_seeds_created": False,
    }
    _atomic_json(output / "top2_candidates.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Run the frozen v33 parameter screen, top-four confirmation, and selection."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hocbf_v33_protocol import (
    CONFIRM_EPISODES,
    CONFIRM_POLICIES,
    FORMAL_CONTEXTS,
    HOCBF_MODE,
    INTERFERENCE_TIE_TOLERANCE,
    PRIMARY_TIE_TOLERANCE,
    PROTOCOL_ID,
    SCREEN_CONTEXTS,
    SCREEN_EPISODES,
    SCREEN_POLICIES,
    TOP_K,
    V31_CHECKPOINT_SHA256,
    candidate_grid,
    confirmation_seed,
    screen_seed,
)
from proximal_v23_io import file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
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
        raise ValueError("v33 development CSV cannot be empty")
    fields = sorted({key for row in rows for key in row})
    leading = [
        key
        for key in ("rank", "candidate", "omega", "lambda_x", "lambda_s")
        if key in fields
    ]
    fields = leading + [key for key in fields if key not in leading]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint(v31_root: Path, context: str, policy: str) -> Path:
    path = v31_root / context / policy / "round_08.pt"
    if (
        not path.is_file()
        or file_sha256(path) != V31_CHECKPOINT_SHA256[context][policy]
    ):
        raise RuntimeError(f"v33 input checkpoint differs for {context}/{policy}")
    return path


def _evaluate(
    *,
    repo: Path,
    config: Path,
    checkpoint: Path,
    context: str,
    policy: str,
    candidate: dict[str, Any],
    runtime_filter: str,
    episodes: int,
    seed: int,
    run_dir: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "episodes.csv"
    valid = False
    if resume and summary_path.is_file() and rows_path.is_file():
        prior = json.loads(summary_path.read_text())
        valid = (
            prior.get("seed") == seed
            and prior.get("num_episodes") == episodes
            and prior.get("checkpoint_sha256") == file_sha256(checkpoint)
            and prior.get("runtime_filter") == (runtime_filter == "on")
            and prior.get("cbf", {}).get("mode") == HOCBF_MODE
        )
    if not valid:
        command = [
            sys.executable,
            str(repo / "experiments/scripts/evaluate_hocbf_v33.py"),
            "--repo",
            str(repo),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--context",
            context,
            "--policy-label",
            policy,
            "--cbf-mode",
            HOCBF_MODE,
            "--runtime-filter",
            runtime_filter,
            "--omega",
            str(candidate["omega"]),
            "--lambda-x",
            str(candidate["lambda_x"]),
            "--lambda-s",
            str(candidate["lambda_s"]),
            "--num-envs",
            str(episodes),
            "--num-episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--device",
            device,
            "--output-json",
            str(summary_path),
            "--output-csv",
            str(rows_path),
        ]
        completed = subprocess.run(
            command, cwd=repo, capture_output=True, text=True, check=False
        )
        if completed.returncode:
            diagnostic = "\n".join(
                (completed.stdout + completed.stderr).splitlines()[-120:]
            )
            raise RuntimeError(
                f"v33 evaluation failed for {context}/{policy}:\n{diagnostic}"
            )
    summary = json.loads(summary_path.read_text())
    rows = _read_rows(rows_path)
    if len(rows) != episodes:
        raise RuntimeError("v33 development evaluation has incomplete episodes")
    return summary, rows


def _paired(
    on_rows: list[dict[str, str]], off_rows: list[dict[str, str]]
) -> dict[str, int | float]:
    def identity(row: dict[str, str]) -> tuple[int, int]:
        return int(row["evaluation_seed"]), int(row["environment_id"])

    if [identity(row) for row in on_rows] != [identity(row) for row in off_rows]:
        raise RuntimeError("v33 development identities are not paired")
    on_success = [row["success"].lower() == "true" for row in on_rows]
    off_success = [row["success"].lower() == "true" for row in off_rows]
    rescue = sum(
        (not before) and after
        for before, after in zip(off_success, on_success, strict=True)
    )
    interference = sum(
        before and (not after)
        for before, after in zip(off_success, on_success, strict=True)
    )
    return {
        "rescue_count": rescue,
        "interference_count": interference,
        "rescue_rate": rescue / len(on_rows),
        "interference_rate": interference / len(on_rows),
    }


def _run_matrix(
    *,
    phase: str,
    candidates: list[dict[str, Any]],
    policies: tuple[str, ...],
    contexts: tuple[str, ...],
    episodes: int,
    repo: Path,
    config: Path,
    v31_root: Path,
    output_root: Path,
    device: str,
    resume: bool,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    off_cache: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, str]]]] = {}
    default = candidate_grid()[0]
    total = len(candidates) * len(policies) * len(contexts)
    completed_count = 0
    for policy in policies:
        for context in contexts:
            seed = (
                screen_seed(policy, context)
                if phase == "screening"
                else confirmation_seed(policy, context)
            )
            checkpoint = _checkpoint(v31_root, context, policy)
            off_cache[(policy, context)] = _evaluate(
                repo=repo,
                config=config,
                checkpoint=checkpoint,
                context=context,
                policy=policy,
                candidate=default,
                runtime_filter="off",
                episodes=episodes,
                seed=seed,
                run_dir=output_root / "raw" / phase / "off" / policy / context,
                device=device,
                resume=resume,
            )
    for candidate in candidates:
        for policy in policies:
            for context in contexts:
                seed = (
                    screen_seed(policy, context)
                    if phase == "screening"
                    else confirmation_seed(policy, context)
                )
                checkpoint = _checkpoint(v31_root, context, policy)
                on_summary, on_rows = _evaluate(
                    repo=repo,
                    config=config,
                    checkpoint=checkpoint,
                    context=context,
                    policy=policy,
                    candidate=candidate,
                    runtime_filter="on",
                    episodes=episodes,
                    seed=seed,
                    run_dir=output_root
                    / "raw"
                    / phase
                    / candidate["candidate"]
                    / policy
                    / context,
                    device=device,
                    resume=resume,
                )
                off_summary, off_rows = off_cache[(policy, context)]
                if (
                    on_summary["initial_state_signature"]
                    != off_summary["initial_state_signature"]
                ):
                    raise RuntimeError("v33 on/off initial states are not paired")
                paired = _paired(on_rows, off_rows)
                metrics.append(
                    {
                        **candidate,
                        "phase": phase,
                        "policy": policy,
                        "context": context,
                        "episodes": episodes,
                        "on_success_rate": on_summary["success_rate"],
                        "off_success_rate": off_summary["success_rate"],
                        **paired,
                        "mean_qddot_correction_jerk": on_summary[
                            "mean_qddot_correction_jerk"
                        ],
                        "mean_qddot_correction_norm": on_summary[
                            "mean_qddot_correction_norm"
                        ],
                        "intervention_events_per_riser": on_summary[
                            "intervention_events_per_riser"
                        ],
                        "intervention_steps_per_riser": on_summary[
                            "intervention_steps_per_riser"
                        ],
                        "unsafe_overlap_steps_per_riser": on_summary[
                            "unsafe_overlap_steps_per_riser"
                        ],
                        "mean_cbf_compute_time_ms": on_summary[
                            "mean_cbf_compute_time_ms"
                        ],
                    }
                )
                completed_count += 1
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "progress": f"{completed_count}/{total}",
                            "candidate": candidate["candidate"],
                            "policy": policy,
                            "context": context,
                            "success": on_summary["success_rate"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return metrics


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = sorted({str(row["candidate"]) for row in rows})
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        subset = [row for row in rows if row["candidate"] == candidate]
        parameters = subset[0]
        a2 = [row for row in subset if row["policy"] == "A2"]
        a1 = [row for row in subset if row["policy"] == "A1"]
        ranked.append(
            {
                "candidate": candidate,
                "omega": parameters["omega"],
                "lambda_x": parameters["lambda_x"],
                "lambda_s": parameters["lambda_s"],
                "A2_mean_on_success": sum(float(row["on_success_rate"]) for row in a2)
                / len(a2),
                "A1_mean_interference": sum(
                    float(row["interference_rate"]) for row in a1
                )
                / len(a1),
                "mean_correction_jerk": sum(
                    float(row["mean_qddot_correction_jerk"]) for row in subset
                )
                / len(subset),
                "mean_intervention_events_per_riser": sum(
                    float(row["intervention_events_per_riser"]) for row in subset
                )
                / len(subset),
                "mean_unsafe_overlap_steps_per_riser": sum(
                    float(row["unsafe_overlap_steps_per_riser"]) for row in subset
                )
                / len(subset),
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["A2_mean_on_success"]),
            float(row["A1_mean_interference"]),
            float(row["mean_correction_jerk"]),
            float(row["mean_intervention_events_per_riser"]),
            str(row["candidate"]),
        )
    )
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    return ranked


def _select(
    confirmation: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    best_success = max(float(row["A2_mean_on_success"]) for row in confirmation)
    primary = [
        row
        for row in confirmation
        if best_success - float(row["A2_mean_on_success"]) < PRIMARY_TIE_TOLERANCE
    ]
    best_interference = min(float(row["A1_mean_interference"]) for row in primary)
    secondary = [
        row
        for row in primary
        if float(row["A1_mean_interference"]) - best_interference
        < INTERFERENCE_TIE_TOLERANCE
    ]
    selected = min(
        secondary,
        key=lambda row: (
            float(row["mean_correction_jerk"]),
            float(row["mean_intervention_events_per_riser"]),
            str(row["candidate"]),
        ),
    )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "globally_selected_and_frozen",
        "source_commit": config["source_boundary"]["git_commit"],
        "candidate": selected["candidate"],
        "omega": selected["omega"],
        "lambda_x": selected["lambda_x"],
        "lambda_s": selected["lambda_s"],
        "zeta": 1.0,
        "drift_ema_previous": 0.8,
        "drift_clip_m_per_s2": 20.0,
        "selection_metrics": selected,
        "primary_tie_pool": [row["candidate"] for row in primary],
        "secondary_tie_pool": [row["candidate"] for row in secondary],
        "selection_order": [
            "A2 three-context mean HOCBF-on success (0.5 pp tie tolerance)",
            "A1 paired interference (lower)",
            "qddot correction jerk (lower)",
            "intervention events per riser (lower)",
        ],
        "final_seeds": {
            "frozen_policy": "204_3xx_xxx",
            "training": "205_3xx_xxx",
            "target": "206_3xx_xxx",
            "D0": "207_3xx_xxx",
            "bootstrap": "208_3xx_xxx",
        },
        "per_context_parameters_forbidden": True,
        "additional_parameter_search": False,
    }


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    v31_root = args.v31_root.resolve()
    output = args.output_root.resolve()
    config = json.loads(config_path.read_text())
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("v33 development config differs")
    candidates = candidate_grid()
    screen_metrics = _run_matrix(
        phase="screening",
        candidates=candidates,
        policies=SCREEN_POLICIES,
        contexts=SCREEN_CONTEXTS,
        episodes=SCREEN_EPISODES,
        repo=repo,
        config=config_path,
        v31_root=v31_root,
        output_root=output,
        device=args.device,
        resume=args.resume,
    )
    screen_ranked = _rank(screen_metrics)
    _write_csv(output / "screening.csv", screen_ranked)
    _write_csv(output / "screening_by_condition.csv", screen_metrics)
    top_ids = {row["candidate"] for row in screen_ranked[:TOP_K]}
    top_candidates = [
        candidate for candidate in candidates if candidate["candidate"] in top_ids
    ]
    confirmation_metrics = _run_matrix(
        phase="confirmation",
        candidates=top_candidates,
        policies=CONFIRM_POLICIES,
        contexts=FORMAL_CONTEXTS,
        episodes=CONFIRM_EPISODES,
        repo=repo,
        config=config_path,
        v31_root=v31_root,
        output_root=output,
        device=args.device,
        resume=args.resume,
    )
    confirmation_ranked = _rank(confirmation_metrics)
    _write_csv(output / "top4_confirmation.csv", confirmation_ranked)
    _write_csv(output / "top4_confirmation_by_condition.csv", confirmation_metrics)
    selected = _select(confirmation_ranked, config)
    selected["elapsed_seconds"] = time.monotonic() - started
    _atomic_json(output / "selected_hocbf.json", selected)
    print(json.dumps(selected, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

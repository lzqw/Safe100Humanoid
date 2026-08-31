"""Development-only sweep for a smoother, earlier sloped-clearance response."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HEIGHT_M = 0.180
CLEARANCE_SLOPE = 0.80
RECOVERY_DISTANCE_M = 0.15
FILTER_ALPHAS = (4.0, 6.0, 8.0, 10.0)
EPISODES = 128
SEED = 154_280_018


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bool(value: str) -> bool:
    return value.lower() == "true"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EPISODES:
        raise RuntimeError(f"expected {EPISODES} rows in {path}, got {len(rows)}")
    return rows


def _run(
    *,
    repo: Path,
    checkpoint: Path,
    output_dir: Path,
    runtime_filter: str,
    alpha: float,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, str]], float]:
    name = "off" if runtime_filter == "off" else f"alpha_{alpha:.1f}"
    arm = output_dir / name
    command = [
        sys.executable,
        str(repo / "experiments/scripts/evaluate_cbf_teacher_v26.py"),
        "--repo",
        str(repo),
        "--checkpoint",
        str(checkpoint),
        "--riser-height",
        str(HEIGHT_M),
        "--clearance-slope",
        str(CLEARANCE_SLOPE),
        "--recovery-distance",
        str(RECOVERY_DISTANCE_M),
        "--filter-alpha",
        str(alpha),
        "--runtime-filter",
        runtime_filter,
        "--num-envs",
        str(EPISODES),
        "--num-episodes",
        str(EPISODES),
        "--seed",
        str(SEED),
        "--device",
        device,
        "--output-json",
        str(arm / "summary.json"),
        "--output-csv",
        str(arm / "episodes.csv"),
    ]
    arm.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (arm / "run.log").open("w") as log:
        completed = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    if completed.returncode:
        tail = (arm / "run.log").read_text().splitlines()[-80:]
        raise RuntimeError("v28 exploratory arm failed:\n" + "\n".join(tail))
    print(
        json.dumps(
            {
                "event": "arm_complete",
                "runtime_filter": runtime_filter,
                "filter_alpha": alpha,
                "elapsed_seconds": round(elapsed, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return (
        json.loads((arm / "summary.json").read_text()),
        _rows(arm / "episodes.csv"),
        elapsed,
    )


def main() -> None:
    args = _args()
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True
    ).stdout:
        raise RuntimeError("v28 exploratory run requires a clean committed worktree")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    off_summary, off_rows, off_elapsed = _run(
        repo=repo,
        checkpoint=checkpoint,
        output_dir=output_dir,
        runtime_filter="off",
        alpha=FILTER_ALPHAS[-1],
        device=args.device,
    )
    off_success = [_bool(row["success"]) for row in off_rows]
    off_kick = [_bool(row["toe_riser_kick"]) for row in off_rows]
    failures = sum(not value for value in off_success)
    aligned = sum(
        (not success) and kick
        for success, kick in zip(off_success, off_kick, strict=True)
    )
    candidates = []
    elapsed = {"off": off_elapsed}
    for alpha in FILTER_ALPHAS:
        summary, rows, duration = _run(
            repo=repo,
            checkpoint=checkpoint,
            output_dir=output_dir,
            runtime_filter="on",
            alpha=alpha,
            device=args.device,
        )
        if summary["initial_state_signature"] != off_summary["initial_state_signature"]:
            raise RuntimeError(f"v28 paired initial-state mismatch for alpha {alpha}")
        on_success = [_bool(row["success"]) for row in rows]
        rescued = sum(
            (not off) and on
            for off, on in zip(off_success, on_success, strict=True)
        )
        regressed = sum(
            off and (not on)
            for off, on in zip(off_success, on_success, strict=True)
        )
        off_rate = sum(off_success) / EPISODES
        on_rate = sum(on_success) / EPISODES
        alignment = aligned / max(1, failures)
        rescue = rescued / max(1, failures)
        candidates.append(
            {
                "filter_alpha": alpha,
                "off_success_rate": off_rate,
                "on_success_rate": on_rate,
                "success_delta": on_rate - off_rate,
                "alignment_coverage": alignment,
                "shield_rescue_rate": rescue,
                "rescued_count": rescued,
                "regressed_count": regressed,
                "mean_correction_norm": summary["mean_correction_norm"],
                "intervention_per_riser": summary["intervention_per_riser"],
                "formal_gate_on_development_sample": (
                    alignment >= 0.80
                    and rescue >= 0.60
                    and 0.40 <= off_rate <= 0.65
                    and 0.80 <= on_rate <= 0.95
                ),
            }
        )
        elapsed[f"alpha_{alpha:.1f}"] = duration

    payload = {
        "schema_version": 1,
        "development_only": True,
        "purpose": "select one smoother filter alpha prospectively for v28",
        "riser_height_m": HEIGHT_M,
        "clearance_barrier_slope": CLEARANCE_SLOPE,
        "recovery_distance_m": RECOVERY_DISTANCE_M,
        "paired_episodes_per_candidate": EPISODES,
        "seed": SEED,
        "filter_alphas": list(FILTER_ALPHAS),
        "candidates": candidates,
        "elapsed_seconds": elapsed,
        "total_elapsed_seconds": sum(elapsed.values()),
    }
    _atomic_json(output_dir / "exploratory_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

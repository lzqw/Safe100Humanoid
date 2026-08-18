"""Short development-only sweep for a less persistent toe-clearance filter."""

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
RECOVERY_DISTANCES_M = (0.02, 0.05, 0.08, 0.12, 0.15)
EPISODES = 64
SEED = 150_270_018


def _parse_args() -> argparse.Namespace:
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


def _arm_dir(output_dir: Path, runtime_filter: str, recovery: float) -> Path:
    if runtime_filter == "off":
        return output_dir / "off"
    return output_dir / f"recovery_{recovery:.3f}"


def _run_arm(
    *,
    repo: Path,
    checkpoint: Path,
    output_dir: Path,
    runtime_filter: str,
    recovery: float,
    device: str,
) -> float:
    arm = _arm_dir(output_dir, runtime_filter, recovery)
    arm.mkdir(parents=True, exist_ok=True)
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
        str(recovery),
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
    started = time.monotonic()
    with (arm / "run.log").open("w") as log:
        completed = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    if completed.returncode:
        tail = (arm / "run.log").read_text().splitlines()[-80:]
        raise RuntimeError("v27 exploratory arm failed:\n" + "\n".join(tail))
    print(
        json.dumps(
            {
                "event": "arm_complete",
                "runtime_filter": runtime_filter,
                "recovery_distance_m": recovery,
                "elapsed_seconds": round(elapsed, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return elapsed


def _bool(value: str) -> bool:
    return value.lower() == "true"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EPISODES:
        raise RuntimeError(f"expected {EPISODES} rows in {path}, got {len(rows)}")
    return rows


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True
    ).stdout:
        raise RuntimeError("v27 exploratory run requires a clean committed worktree")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    elapsed: dict[str, float] = {}
    elapsed["off"] = _run_arm(
        repo=repo,
        checkpoint=checkpoint,
        output_dir=output_dir,
        runtime_filter="off",
        recovery=RECOVERY_DISTANCES_M[-1],
        device=args.device,
    )
    off_summary = json.loads((output_dir / "off" / "summary.json").read_text())
    off_rows = _rows(output_dir / "off" / "episodes.csv")
    off_success = [_bool(row["success"]) for row in off_rows]
    off_kick = [_bool(row["toe_riser_kick"]) for row in off_rows]
    failures = sum(not value for value in off_success)
    aligned = sum(
        (not success) and kick
        for success, kick in zip(off_success, off_kick, strict=True)
    )

    candidates = []
    for recovery in RECOVERY_DISTANCES_M:
        key = f"recovery_{recovery:.3f}"
        elapsed[key] = _run_arm(
            repo=repo,
            checkpoint=checkpoint,
            output_dir=output_dir,
            runtime_filter="on",
            recovery=recovery,
            device=args.device,
        )
        arm = output_dir / key
        on_summary = json.loads((arm / "summary.json").read_text())
        if off_summary["initial_state_signature"] != on_summary["initial_state_signature"]:
            raise RuntimeError(f"paired initial-state mismatch for recovery {recovery}")
        on_success = [_bool(row["success"]) for row in _rows(arm / "episodes.csv")]
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
                "recovery_distance_m": recovery,
                "off_success_rate": off_rate,
                "on_success_rate": on_rate,
                "success_delta": on_rate - off_rate,
                "alignment_coverage": alignment,
                "shield_rescue_rate": rescue,
                "rescued_count": rescued,
                "regressed_count": regressed,
                "formal_gate_on_development_sample": (
                    alignment >= 0.80
                    and rescue >= 0.60
                    and 0.40 <= off_rate <= 0.65
                    and 0.80 <= on_rate <= 0.95
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "development_only": True,
        "excluded_from_v26_formal_evidence": True,
        "purpose": "select one recovery distance prospectively for v27",
        "riser_height_m": HEIGHT_M,
        "clearance_barrier_slope": CLEARANCE_SLOPE,
        "paired_episodes_per_candidate": EPISODES,
        "seed": SEED,
        "recovery_distances_m": list(RECOVERY_DISTANCES_M),
        "candidates": candidates,
        "elapsed_seconds": elapsed,
        "total_elapsed_seconds": sum(elapsed.values()),
    }
    _atomic_json(output_dir / "exploratory_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Disclosed development-only sweep for the v26 higher-riser follow-up.

These episodes locate a plausible rescue corridor.  Their seeds and outcomes
must not be reused as the formal v26 calibration or final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HEIGHTS_M = (0.140, 0.150, 0.160, 0.170, 0.180)
EPISODES = 64
SEED_BASE = 146_260_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-vram-mib", type=int, default=3500)
    parser.add_argument("--parallel-free-vram-mib", type=int, default=6500)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _free_vram_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return min(int(line.strip()) for line in output.splitlines() if line.strip())


def _wait_for_vram(minimum_mib: int, poll_seconds: float) -> int:
    while True:
        free = _free_vram_mib()
        if free >= minimum_mib:
            return free
        print(
            json.dumps(
                {
                    "event": "waiting_for_gpu",
                    "free_vram_mib": free,
                    "required_vram_mib": minimum_mib,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def _arm_command(
    *,
    repo: Path,
    checkpoint: Path,
    output_dir: Path,
    height: float,
    runtime_filter: str,
    seed: int,
    device: str,
) -> list[str]:
    arm = output_dir / f"height_{height:.3f}" / runtime_filter
    return [
        sys.executable,
        str(repo / "experiments/scripts/evaluate_cbf_teacher_v26.py"),
        "--repo",
        str(repo),
        "--checkpoint",
        str(checkpoint),
        "--riser-height",
        str(height),
        "--runtime-filter",
        runtime_filter,
        "--num-envs",
        str(EPISODES),
        "--num-episodes",
        str(EPISODES),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-json",
        str(arm / "summary.json"),
        "--output-csv",
        str(arm / "episodes.csv"),
    ]


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode:
        diagnostic = "\n".join(
            (completed.stdout + "\n" + completed.stderr).splitlines()[-120:]
        )
        raise RuntimeError(f"v26 exploratory arm failed:\n{diagnostic}")


def _run_pair(
    *,
    repo: Path,
    checkpoint: Path,
    output_dir: Path,
    height: float,
    seed: int,
    device: str,
    parallel: bool,
) -> None:
    commands = {
        arm: _arm_command(
            repo=repo,
            checkpoint=checkpoint,
            output_dir=output_dir,
            height=height,
            runtime_filter=arm,
            seed=seed,
            device=device,
        )
        for arm in ("off", "on")
    }
    if not parallel:
        for command in commands.values():
            _run(command, cwd=repo)
        return

    processes = {
        arm: subprocess.Popen(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for arm, command in commands.items()
    }
    failures = []
    for arm, process in processes.items():
        output, _ = process.communicate()
        if process.returncode:
            failures.append(f"{arm}:\n" + "\n".join(output.splitlines()[-120:]))
    if failures:
        raise RuntimeError("v26 parallel exploratory pair failed:\n" + "\n".join(failures))


def _bool(value: str) -> bool:
    return value.lower() == "true"


def _summarize(output_dir: Path, height: float, seed: int) -> dict[str, Any]:
    root = output_dir / f"height_{height:.3f}"
    summaries = {
        arm: json.loads((root / arm / "summary.json").read_text())
        for arm in ("off", "on")
    }
    if summaries["off"]["initial_state_signature"] != summaries["on"][
        "initial_state_signature"
    ]:
        raise RuntimeError("v26 exploratory pair did not share initial conditions")
    rows = {}
    for arm in ("off", "on"):
        with (root / arm / "episodes.csv").open(newline="") as handle:
            rows[arm] = list(csv.DictReader(handle))
    if len(rows["off"]) != EPISODES or len(rows["on"]) != EPISODES:
        raise RuntimeError("v26 exploratory pair is incomplete")
    off_success = [_bool(row["success"]) for row in rows["off"]]
    on_success = [_bool(row["success"]) for row in rows["on"]]
    off_kick = [_bool(row["toe_riser_kick"]) for row in rows["off"]]
    failures = sum(not value for value in off_success)
    aligned = sum(
        (not success) and kick
        for success, kick in zip(off_success, off_kick, strict=True)
    )
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
    formal_gate = (
        alignment >= 0.80
        and rescue >= 0.60
        and 0.40 <= off_rate <= 0.65
        and 0.80 <= on_rate <= 0.95
    )
    return {
        "development_only": True,
        "excluded_from_formal_v26": True,
        "riser_height_m": height,
        "evaluation_seed": seed,
        "paired_count": EPISODES,
        "off_success_rate": off_rate,
        "on_success_rate": on_rate,
        "alignment_coverage": alignment,
        "shield_rescue_rate": rescue,
        "rescued_count": rescued,
        "regressed_count": regressed,
        "formal_gate_on_exploratory_sample": formal_gate,
    }


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True
    ).stdout:
        raise RuntimeError("v26 exploratory run requires a clean committed worktree")
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = output_dir / "exploratory_attempts.json"
    attempts = (
        json.loads(attempts_path.read_text())
        if args.resume and attempts_path.is_file()
        else []
    )
    completed_heights = {float(row["riser_height_m"]) for row in attempts}

    for index, height in enumerate(HEIGHTS_M):
        if height in completed_heights:
            continue
        free = _wait_for_vram(args.minimum_free_vram_mib, args.poll_seconds)
        parallel = free >= args.parallel_free_vram_mib
        seed = SEED_BASE + index
        print(
            json.dumps(
                {
                    "event": "starting_pair",
                    "riser_height_m": height,
                    "seed": seed,
                    "free_vram_mib": free,
                    "parallel_arms": parallel,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _run_pair(
            repo=repo,
            checkpoint=checkpoint,
            output_dir=output_dir,
            height=height,
            seed=seed,
            device=args.device,
            parallel=parallel,
        )
        attempt = _summarize(output_dir, height, seed)
        attempts.append(attempt)
        _atomic_json(attempts_path, attempts)
        print(json.dumps(attempt, sort_keys=True), flush=True)

    _atomic_json(
        output_dir / "exploratory_summary.json",
        {
            "schema_version": 1,
            "status": "complete",
            "development_only": True,
            "excluded_from_formal_v26": True,
            "heights_m": list(HEIGHTS_M),
            "paired_episodes_per_height": EPISODES,
            "attempts": attempts,
        },
    )


if __name__ == "__main__":
    main()

"""Freeze passing v141 development configurations before formal execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from filter_free_v141_protocol import (
    BASE_CHECKPOINT_SHA256,
    FORMAL_EVALUATION_EPISODES,
    FORMAL_EVALUATION_SEED,
    FORMAL_TRAINING_SEEDS,
    METHOD_ID,
    PROTOCOL_ID,
    SPECIALISTS,
)


SOURCE_FILES = (
    "src/tasks/stairs_cbf/filter_free_v141.py",
    "src/tasks/stairs_cbf/teacher_v30.py",
    "src/tasks/stairs_cbf/teacher_v30_math.py",
    "experiments/scripts/filter_free_v141_protocol.py",
    "experiments/scripts/mixed_vec_env_v141.py",
    "experiments/scripts/train_filter_free_v141.py",
    "experiments/scripts/run_filter_free_v141.py",
    "experiments/scripts/freeze_filter_free_v141.py",
    "experiments/scripts/run_formal_filter_free_v141.py",
    "experiments/scripts/publish_filter_free_v141.py",
    "experiments/scripts/supervise_filter_free_v141.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--commit-and-push", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    development_root = args.development_root.resolve()
    output = args.output_json.resolve()
    source_summary = development_root / "development_summary.json"
    if not source_summary.is_file():
        raise FileNotFoundError(source_summary)
    development = json.loads(source_summary.read_text())
    if (
        development.get("protocol_id") != PROTOCOL_ID
        or development.get("method_id") != METHOD_ID
        or development.get("both_specialists_pass") is not True
        or development.get("next_phase") != "freeze_and_formal"
    ):
        raise RuntimeError("v141 development has not satisfied both specialist gates")
    selected = development.get("selected", {})
    if set(selected) != set(SPECIALISTS) or not all(
        selected[name].get("development_success") is True
        for name in SPECIALISTS
    ):
        raise RuntimeError("v141 selected development candidates are incomplete")
    published_summary = output.parent / "development_summary.json"
    if output.exists():
        existing = json.loads(output.read_text())
        if existing.get("frozen_before_formal") is not True:
            raise RuntimeError("existing v141 frozen configuration is invalid")
        _atomic_json(published_summary, development)
        if args.commit_and_push:
            relative = output.relative_to(repo)
            summary_relative = published_summary.relative_to(repo)
            committed = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=repo,
                check=False,
                capture_output=True,
            )
            committed_summary = subprocess.run(
                ["git", "show", f"HEAD:{summary_relative}"],
                cwd=repo,
                check=False,
                capture_output=True,
            )
            if (
                committed.returncode != 0
                or committed.stdout != output.read_bytes()
                or committed_summary.returncode != 0
                or committed_summary.stdout != published_summary.read_bytes()
            ):
                subprocess.run(
                    ["git", "add", str(relative), str(summary_relative)],
                    cwd=repo,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "Freeze v141 formal configuration"],
                    cwd=repo,
                    check=True,
                )
            subprocess.run(
                ["git", "push", "origin", "feature/online-safe-refinement"],
                cwd=repo,
                check=True,
            )
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    worktree_status = _git(repo, "status", "--porcelain")
    if worktree_status:
        raise RuntimeError("v141 freezing requires a clean worktree")
    source_commit = _git(repo, "rev-parse", "HEAD")
    source_hashes = {
        relative: _sha256(repo / relative) for relative in SOURCE_FILES
    }
    frozen = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "status": "frozen_before_formal_execution",
        "frozen_before_formal": True,
        "frozen_unix_time": time.time(),
        "source_commit": source_commit,
        "source_files_sha256": source_hashes,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "formal_training_seeds": list(FORMAL_TRAINING_SEEDS),
        "formal_evaluation_seed": FORMAL_EVALUATION_SEED,
        "formal_evaluation_episodes": FORMAL_EVALUATION_EPISODES,
        "fixed_final_round_checkpoint": True,
        "best_so_far_selection": False,
        "formal_results_seen_before_freeze": False,
        "specialists": {
            name: {
                "configuration": selected[name]["configuration"],
                "development_evidence": {
                    key: selected[name][key]
                    for key in (
                        "candidate",
                        "target_off_success",
                        "target_on_success",
                        "f1_retention_off_success",
                        "shield_gap",
                        "would_intervene_fraction",
                        "counterfactual_correction_norm",
                        "nominal_violation_steps_per_riser",
                        "actor_moving_forward_kl",
                        "development_score",
                        "development_success_checks",
                    )
                },
            }
            for name in SPECIALISTS
        },
    }
    _atomic_json(published_summary, development)
    _atomic_json(output, frozen)
    if args.commit_and_push:
        relative = output.relative_to(repo)
        summary_relative = published_summary.relative_to(repo)
        subprocess.run(
            ["git", "add", str(relative), str(summary_relative)],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Freeze v141 formal configuration"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "push", "origin", "feature/online-safe-refinement"],
            cwd=repo,
            check=True,
        )
    print(json.dumps(frozen, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Freeze the single v27 configuration before confirmatory simulation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cbf_teacher_v27_protocol import (
    ADAPTATION_SEED,
    ALIGNMENT_COVERAGE_MINIMUM,
    BASE_CHECKPOINT_SHA256,
    CALIBRATION_EPISODES,
    CALIBRATION_OFF_SUCCESS_BOUNDS,
    CALIBRATION_ON_SUCCESS_BOUNDS,
    CALIBRATION_REPEATS,
    DEVELOPMENT_SEED,
    EVAL_BATCH_SIZE,
    EXPERIMENT_NAME,
    FINAL_EPISODES,
    FINAL_SEED_BASE,
    PROTOCOL_ID,
    SHIELD_RESCUE_MINIMUM,
    SOURCE_FILES,
    calibration_seed,
    file_sha256,
    fixed_environment_parameters,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    development = args.development_summary.resolve()
    output = args.output.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("worktree must be clean before freezing v27")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v27 base checkpoint differs from frozen pi0")
    if not development.is_file():
        raise FileNotFoundError(development)
    development_payload = json.loads(development.read_text())
    selected = next(
        item
        for item in development_payload["candidates"]
        if item["recovery_distance_m"]
        == fixed_environment_parameters()["recovery_distance_m"]
    )
    if selected.get("formal_gate_on_development_sample") is not True:
        raise RuntimeError("selected v27 development candidate did not pass")

    commit = _git(repo, "rev-parse", "HEAD")
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        content = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if content != path.read_bytes():
            raise RuntimeError(f"uncommitted v27 source: {relative}")
        source_hashes[relative] = file_sha256(path)

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": EXPERIMENT_NAME,
        "status": "prospectively_frozen_before_v27_confirmatory_calibration",
        "environment": fixed_environment_parameters(),
        "development_selection": {
            "development_only": True,
            "summary_sha256": file_sha256(development),
            "seed": DEVELOPMENT_SEED,
            "selected_candidate": selected,
            "excluded_from_formal_evidence": True,
        },
        "calibration": {
            "single_fixed_candidate": True,
            "paired_episodes": CALIBRATION_EPISODES,
            "batch_size": EVAL_BATCH_SIZE,
            "repeats": CALIBRATION_REPEATS,
            "seeds": [calibration_seed(i) for i in range(CALIBRATION_REPEATS)],
            "paired_filter_arms_share_each_seed": True,
            "base_policy_only": True,
            "thresholds": {
                "alignment_coverage_minimum": ALIGNMENT_COVERAGE_MINIMUM,
                "shield_rescue_minimum": SHIELD_RESCUE_MINIMUM,
                "off_success_bounds": list(CALIBRATION_OFF_SUCCESS_BOUNDS),
                "on_success_bounds": list(CALIBRATION_ON_SUCCESS_BOUNDS),
            },
        },
        "conditional_followup": {
            "training_only_if_calibration_qualifies": True,
            "training_algorithm": "unchanged_v26_eight_round_moving_kl_ppo",
            "adaptation_seed": ADAPTATION_SEED,
            "final_episodes": FINAL_EPISODES,
            "final_seed_base": FINAL_SEED_BASE,
        },
        "freshness": {
            "formal_seeds_disjoint_from_development": True,
            "before_any_formal_v27_simulator_episode": True,
        },
        "implementation_boundary": {
            "git_commit": commit,
            "source_files": source_hashes,
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != rendered:
        raise RuntimeError(f"refusing to overwrite a different v27 freeze: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


if __name__ == "__main__":
    main()

"""Create held-out seeds only after v34 development selection is frozen."""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import subprocess
from pathlib import Path
from typing import Any

from proximal_v23_io import file_sha256
from velocity_cbf_v34_protocol import FORMAL_CONTEXTS, PROTOCOL_ID


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fresh_seed(used: set[int]) -> int:
    while True:
        value = 300_000_000 + secrets.randbelow(1_500_000_000)
        if value not in used and value + 1 not in used:
            used.update((value, value + 1))
            return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--development-selection", type=Path, required=True)
    parser.add_argument("--top8-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v34 final freeze requires a clean committed worktree")
    if output.exists():
        raise RuntimeError("refusing to overwrite v34 final parameter/identity freeze")
    config = json.loads(args.search_config.resolve().read_text())
    selected = json.loads(args.development_selection.resolve().read_text())
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or selected.get("protocol_id") != PROTOCOL_ID
    ):
        raise RuntimeError("v34 final freeze inputs differ")
    if selected.get("status") != "development_selected_parameters_and_round8_policies":
        raise RuntimeError("v34 development selection is incomplete")
    checkpoints: dict[str, Any] = {}
    for context in FORMAL_CONTEXTS:
        record = selected["trained_checkpoints"][context]
        checkpoint = Path(record["external_path"])
        actual = file_sha256(checkpoint)
        if actual != record["sha256"]:
            raise RuntimeError(f"v34 trained checkpoint differs for {context}")
        checkpoints[context] = {
            "external_path": str(checkpoint.resolve()),
            "sha256": actual,
            "actor_state_sha256": record["actor_state_sha256"],
        }
    with args.top8_results.resolve().open(newline="") as handle:
        top8 = list(csv.DictReader(handle))
    direct = next(
        (row for row in top8 if row["candidate"] == selected["candidate"]), None
    )
    if direct is None:
        raise RuntimeError("v34 selected candidate is absent from stage2 results")
    used = set()
    target_seeds = {context: _fresh_seed(used) for context in FORMAL_CONTEXTS}
    d0_seeds = {context: _fresh_seed(used) for context in FORMAL_CONTEXTS}
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "globally_selected_and_final_test_frozen",
        "candidate": selected["candidate"],
        "candidate_index": selected["candidate_index"],
        "mode": selected["mode"],
        "parameters": selected["parameters"],
        "selection_objective": selected["selection_objective"],
        "stage2_direct_replacement": {
            "mean_success": float(direct["mean_success"]),
            **{
                f"{context}_success": float(direct[f"{context}_success_rate"])
                for context in FORMAL_CONTEXTS
            },
        },
        "trained_development_mean_success": selected[
            "trained_development_mean_success"
        ],
        "trained_development_results": selected["trained_development_results"],
        "trained_checkpoints": checkpoints,
        "held_out_identity_seeds": {
            "target": target_seeds,
            "D0": d0_seeds,
        },
        "final_design": {
            "target_episodes_per_condition_context": 512,
            "target_batches": 2,
            "target_batch_size": 256,
            "D0_episodes_per_condition_source_context": 256,
            "D0_batches": 1,
            "D0_batch_size": 256,
            "paired_initial_identities": True,
            "deterministic_policy_mean": True,
            "final_audit_maximum_invocations": 1,
        },
        "freeze_boundary": {
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "branch", "--show-current"),
            "search_config_sha256": file_sha256(args.search_config.resolve()),
            "top8_results_sha256": file_sha256(args.top8_results.resolve()),
            "development_selection_sha256": file_sha256(
                args.development_selection.resolve()
            ),
        },
        "final_seeds_created_after_development_freeze": True,
        "final_test_started": False,
    }
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

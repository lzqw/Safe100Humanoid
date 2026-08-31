"""Freeze v34 sources, checkpoints, candidates, and development identities."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from proximal_v23_io import file_sha256
from velocity_cbf_v34_protocol import (
    BASE_CHECKPOINT_SHA256,
    FORMAL_CONTEXTS,
    PROTOCOL_ID,
    SOURCE_FILES,
    V31_CHECKPOINT_SHA256,
    frozen_search_specification,
    stage1_seed,
    stage2_seed,
    trained_development_seed,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v31-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    base = args.base_checkpoint.resolve()
    v31_root = args.v31_root.resolve()
    output = args.output.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v34 search freeze requires a clean committed worktree")
    if output.exists():
        raise RuntimeError("refusing to overwrite frozen v34 search config")
    if file_sha256(base) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v34 base checkpoint hash differs")
    checkpoints: dict[str, Any] = {}
    for context in FORMAL_CONTEXTS:
        checkpoints[context] = {}
        for policy in ("A1", "A2"):
            checkpoint = v31_root / context / policy / "round_08.pt"
            actual = file_sha256(checkpoint)
            expected = V31_CHECKPOINT_SHA256[context][policy]
            if actual != expected:
                raise RuntimeError(f"v34 checkpoint differs for {context}/{policy}")
            checkpoints[context][policy] = {
                "relative_external_path": f"{context}/{policy}/round_08.pt",
                "sha256": actual,
            }
    source_hashes = {}
    for relative in SOURCE_FILES:
        source = repo / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        _git(repo, "ls-files", "--error-unmatch", relative)
        source_hashes[relative] = file_sha256(source)
    payload = {
        "schema_version": 1,
        **frozen_search_specification(),
        "status": "development_search_frozen_before_smoke",
        "source_boundary": {
            "branch": _git(repo, "branch", "--show-current"),
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "source_files": source_hashes,
            "v31_v32_v33_sources_or_results_modified": False,
        },
        "checkpoints": {
            "base": {"sha256": file_sha256(base)},
            "v31_round_8": checkpoints,
        },
        "development_seeds": {
            "stage1": {context: stage1_seed(context) for context in FORMAL_CONTEXTS},
            "stage2": {context: stage2_seed(context) for context in FORMAL_CONTEXTS},
            "trained_policy_evaluation": {
                context: trained_development_seed(context)
                for context in FORMAL_CONTEXTS
            },
        },
        "final_test": {
            "seeds_created": False,
            "identities_accessible": False,
            "runs_after_parameter_and_policy_freeze_only": True,
            "maximum_invocations": 1,
        },
    }
    if payload["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("v34 protocol id differs")
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

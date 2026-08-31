"""Freeze v33 sources, checkpoints, matrix, and seeds before any GPU result."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from hocbf_v33_protocol import (
    BASE_CHECKPOINT_SHA256,
    SMOKE_ENVS,
    SMOKE_SEED,
    SMOKE_STEPS,
    SOURCE_FILES,
    V31_CHECKPOINT_SHA256,
    confirmation_seed,
    frozen_audit_seed,
    frozen_constants,
    screen_seed,
)
from proximal_v23_io import file_sha256


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v31-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    base = args.base_checkpoint.resolve()
    v31 = args.v31_root.resolve()
    output = args.output.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v33 freeze requires a clean committed worktree")
    if output.exists():
        raise RuntimeError("refusing to overwrite the v33 config")
    if file_sha256(base) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v33 base checkpoint differs")
    checkpoints = {}
    for context, policies in V31_CHECKPOINT_SHA256.items():
        checkpoints[context] = {}
        for policy, expected in policies.items():
            path = v31 / context / policy / "round_08.pt"
            actual = file_sha256(path)
            if actual != expected:
                raise RuntimeError(f"v33 v31 checkpoint differs for {context}/{policy}")
            checkpoints[context][policy] = {
                "relative_external_path": f"{context}/{policy}/round_08.pt",
                "sha256": actual,
            }
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if not _git(repo, "ls-files", "--error-unmatch", relative):
            raise RuntimeError(f"v33 source is not tracked: {relative}")
        source_hashes[relative] = file_sha256(path)
    payload = {
        "schema_version": 1,
        **frozen_constants(),
        "status": "frozen_before_single_smoke_and_development",
        "source_boundary": {
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "branch", "--show-current"),
            "source_files": source_hashes,
            "current_CBF0_source_unchanged": True,
            "v31_v32_sources_or_results_modified": False,
        },
        "checkpoints": {
            "base": {"sha256": file_sha256(base)},
            "v31_round_8": checkpoints,
        },
        "smoke": {
            "runs_exactly_once": True,
            "context": "F1",
            "policy": "v31_A2_round_8",
            "num_envs": SMOKE_ENVS,
            "control_steps": SMOKE_STEPS,
            "seed": SMOKE_SEED,
            "throughput_floor_vs_CBF0": 0.70,
        },
        "seeds": {
            "screening": {
                policy: {
                    context: screen_seed(policy, context) for context in ("F1", "F2")
                }
                for policy in ("A1", "A2")
            },
            "confirmation": {
                policy: {
                    context: confirmation_seed(policy, context)
                    for context in ("F1", "F2", "F3")
                }
                for policy in ("A1", "A2")
            },
            "frozen_policy_audit": {
                policy: {
                    context: frozen_audit_seed(policy, context)
                    for context in ("F1", "F2", "F3")
                }
                for policy in ("A1", "A2")
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

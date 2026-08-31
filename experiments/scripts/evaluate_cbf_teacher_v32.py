"""Validate v32 provenance, then use the unchanged v31 low-level evaluator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cbf_teacher_v32_protocol import (
    FORMAL_CONTEXTS,
    PROTOCOL_ID,
    environment_parameters,
)
from proximal_v23_io import file_sha256


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context", choices=(*FORMAL_CONTEXTS, "D0"), required=True)
    parser.add_argument("--runtime-filter", choices=("on", "off"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    committed = subprocess.run(
        ["git", "show", f"HEAD:{protocol_path.relative_to(repo)}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    checks = {
        "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
        "status": protocol.get("status") == "frozen_before_v32_preflight_and_formal",
        "committed": committed == protocol_path.read_bytes(),
        "context": protocol.get("contexts", {}).get(args.context)
        == environment_parameters(args.context),
        "one_episode_per_environment": args.num_envs == args.num_episodes
        and args.num_envs > 0,
        "checkpoint": args.checkpoint.resolve().is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v32 evaluation validation failed: {checks}")
    v31_protocol = repo / "results/online/proximal_v31/protocol.json"
    command = [
        sys.executable,
        str(repo / "experiments/scripts/evaluate_cbf_teacher_v31.py"),
        "--repo",
        str(repo),
        "--protocol",
        str(v31_protocol),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--context",
        args.context,
        "--runtime-filter",
        args.runtime_filter,
        "--num-envs",
        str(args.num_envs),
        "--num-episodes",
        str(args.num_episodes),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output-json",
        str(args.output_json.resolve()),
        "--output-csv",
        str(args.output_csv.resolve()),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo)
    completed = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        diagnostic = "\n".join(
            (completed.stdout + "\n" + completed.stderr).splitlines()[-180:]
        )
        raise RuntimeError(f"v32 low-level evaluation failed:\n{diagnostic}")
    output_json = args.output_json.resolve()
    payload = json.loads(output_json.read_text())
    payload.update(
        {
            "v32_protocol_id": PROTOCOL_ID,
            "v32_protocol_sha256": file_sha256(protocol_path),
            "low_level_evaluator": "unchanged_v31_environment_evaluator",
            "low_level_protocol_sha256": file_sha256(v31_protocol),
        }
    )
    _atomic_json(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

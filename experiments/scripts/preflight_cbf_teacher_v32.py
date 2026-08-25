"""Run the two one-attempt v32 functional preflight cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cbf_teacher_v32_protocol import PREFLIGHT_CASES, PROTOCOL_ID
from proximal_v23_io import file_sha256


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v31-formal-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("v32 preflight output already exists; cases run once only")
    output.mkdir(parents=True)
    summaries = {}
    try:
        for case in PREFLIGHT_CASES:
            label = f"{case['kind']}_{case['context']}_{case['schedule']}"
            case_dir = output / label
            command = [
                sys.executable,
                str(repo / "experiments/scripts/refine_cbf_teacher_v32.py"),
                "--repo",
                str(repo),
                "--base-checkpoint",
                str(args.base_checkpoint.resolve()),
                "--v31-formal-root",
                str(args.v31_formal_root.resolve()),
                "--protocol",
                str(args.protocol.resolve()),
                "--output-dir",
                str(case_dir),
                "--phase",
                "preflight",
                "--kind",
                str(case["kind"]),
                "--context",
                str(case["context"]),
                "--schedule",
                str(case["schedule"]),
                "--device",
                args.device,
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
                raise RuntimeError(f"v32 preflight case {label} failed:\n{diagnostic}")
            summary_path = case_dir / "preflight_case_summary.json"
            summary = json.loads(summary_path.read_text())
            if not summary.get("passed") or summary.get("rounds_completed") != 1:
                raise RuntimeError(f"v32 preflight case {label} is incomplete")
            summaries[label] = {
                "kind": summary["kind"],
                "context": summary["context"],
                "schedule": summary["schedule"],
                "seed": summary["seed"],
                "final_checkpoint_sha256": summary["final_checkpoint_sha256"],
                "summary_sha256": file_sha256(summary_path),
                "integrity_checks": summary["rounds"][0]["integrity_checks"],
                "structural_audit": summary["structural_audit"],
            }
        checks = {
            "exactly_two_cases": len(summaries) == 2,
            "all_integrity_checks": all(
                all(item["integrity_checks"].values()) for item in summaries.values()
            ),
            "continuation_restores_v31_A2": any(
                item["kind"] == "continuation" for item in summaries.values()
            ),
            "mixed_exposes_64_envs": any(
                item["kind"] == "mixed"
                and item["structural_audit"]["mixed"]["exposed_envs"] == 64
                for item in summaries.values()
            ),
            "mixed_contains_all_contexts": any(
                item["kind"] == "mixed"
                and set(item["structural_audit"]) == {"F1", "F2", "F3", "mixed"}
                for item in summaries.values()
            ),
        }
        payload = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "complete": True,
            "passed": all(checks.values()),
            "formal_ready": all(checks.values()),
            "cases": summaries,
            "checks": checks,
            "simulator_case_attempts": 2,
            "performance_evidence": False,
            "simulator_cases_rerun": False,
            "protocol_sha256": file_sha256(args.protocol.resolve()),
        }
        _atomic_json(output / "preflight_summary.json", payload)
        if not payload["passed"]:
            raise RuntimeError(f"v32 preflight aggregate failed: {checks}")
        print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as error:
        _atomic_json(
            output / "preflight_failure.json",
            {
                "protocol_id": PROTOCOL_ID,
                "error_type": type(error).__name__,
                "error": str(error),
                "performance_result": False,
                "simulator_cases_rerun": False,
            },
        )
        raise


if __name__ == "__main__":
    main()

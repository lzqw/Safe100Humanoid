"""Run v31's single three-context, three-method functional preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from cbf_teacher_v31_protocol import (
    BEHAVIOR_LOG_PROB_ATOL,
    PREFLIGHT_CASES,
    PROTOCOL_ID,
)
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    protocol = args.protocol.resolve()
    output = args.output_dir.resolve()
    marker = output / "preflight_attempt_started.json"
    if output.exists() or marker.exists():
        raise RuntimeError("v31 permits exactly one preflight attempt")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout:
        raise RuntimeError("v31 preflight requires a clean committed worktree")
    payload = json.loads(protocol.read_text())
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "frozen_before_v31_preflight_and_formal"
    ):
        raise RuntimeError("v31 preflight protocol is not frozen")
    output.mkdir(parents=True)
    _atomic_json(
        marker,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": file_sha256(protocol),
            "attempt": 1,
            "cases": [list(item) for item in PREFLIGHT_CASES],
        },
    )
    started = time.monotonic()
    cases: dict[str, Any] = {}
    try:
        for context, arm, seed in PREFLIGHT_CASES:
            case_dir = output / "cases" / f"{context}_{arm}"
            command = [
                sys.executable,
                str(repo / "experiments/scripts/refine_cbf_teacher_v31.py"),
                "--repo",
                str(repo),
                "--base-checkpoint",
                str(checkpoint),
                "--protocol",
                str(protocol),
                "--output-dir",
                str(case_dir),
                "--phase",
                "preflight",
                "--context",
                context,
                "--arm",
                arm,
                "--device",
                args.device,
            ]
            completed = subprocess.run(
                command, cwd=repo, capture_output=True, text=True, check=False
            )
            if completed.returncode:
                diagnostic = "\n".join(
                    (completed.stdout + "\n" + completed.stderr).splitlines()[-160:]
                )
                raise RuntimeError(
                    f"v31 preflight failed for {context}/{arm}:\n{diagnostic}"
                )
            summary = json.loads((case_dir / "preflight_case_summary.json").read_text())
            if summary.get("seed") != seed or not summary.get("passed"):
                raise RuntimeError(f"invalid v31 preflight summary for {context}/{arm}")
            cases[f"{context}_{arm}"] = summary

        from src.tasks.stairs_cbf.online import (
            BEHAVIOR_DISTRIBUTION_PARAM_ATOL,
            validate_behavior_log_prob,
        )
        from src.tasks.stairs_cbf.online import (
            BEHAVIOR_LOG_PROB_ATOL as IMPLEMENTED_ATOL,
        )

        accepted_error = validate_behavior_log_prob(
            torch.zeros(1), torch.full((1,), 7.5e-4)
        )
        checks = {
            "exactly_three_cases": len(cases) == 3,
            "all_contexts_construct_reset_step": {
                value["context"] for value in cases.values()
            }
            == {"F1", "F2", "F3"},
            "all_methods_update": {value["arm"] for value in cases.values()}
            == {"A0", "A1", "A2"},
            "patch_counts_F1_F2_F3": {
                key: value["structural_audit"]["stair_target_patch_slots"]
                for key, value in cases.items()
            }
            == {"F1_A0": 10, "F2_A1": 10, "F3_A2": 12},
            "raw_action_storage": all(
                value["checks"]["raw_policy_action_stored"] for value in cases.values()
            ),
            "safe_action_execution": all(
                value["checks"]["safe_action_executed"] for value in cases.values()
            ),
            "A1_A2_teacher_loss_finite": all(
                cases[key]["checks"]["configured_teacher_loss_finite"]
                for key in ("F2_A1", "F3_A2")
            ),
            "actor_and_critic_update": all(
                value["checks"]["actor_backward_and_steps_complete"]
                and value["checks"]["critic_backward_and_steps_complete"]
                for value in cases.values()
            ),
            "no_nan_or_inf": all(
                value["checks"]["no_nan_or_inf"] for value in cases.values()
            ),
            "implemented_log_prob_atol_is_1e_3": IMPLEMENTED_ATOL
            == BEHAVIOR_LOG_PROB_ATOL
            == 1.0e-3,
            "legal_7_5e_4_reduction_error_continues": abs(accepted_error - 7.5e-4)
            < 1.0e-9,
            "distribution_parameter_tolerance_remains_strict": (
                BEHAVIOR_DISTRIBUTION_PARAM_ATOL == 2.0e-5
            ),
        }
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": file_sha256(protocol),
            "attempts": 1,
            "complete": True,
            "passed": all(checks.values()),
            "checks": checks,
            "case_summaries": {
                key: {
                    "context": value["context"],
                    "arm": value["arm"],
                    "seed": value["seed"],
                    "rounds_completed": value["rounds_completed"],
                    "stair_target_patch_slots": value["structural_audit"][
                        "stair_target_patch_slots"
                    ],
                    "num_risers": value["structural_audit"]["num_risers"],
                    "teacher_loss": value["rounds"][0]["metrics"]["teacher_loss"],
                    "behavior_log_prob_max_abs_error": max(
                        value["rounds"][0]["metrics"][
                            "behavior_reference_log_prob_max_abs_error"
                        ],
                        value["rounds"][0]["metrics"][
                            "behavior_current_log_prob_max_abs_error"
                        ],
                    ),
                }
                for key, value in cases.items()
            },
            "elapsed_seconds": time.monotonic() - started,
            "formal_ready": all(checks.values()),
        }
        _atomic_json(output / "preflight_summary.json", summary)
        if not summary["passed"]:
            raise RuntimeError(f"v31 single preflight failed: {checks}")
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as error:
        _atomic_json(
            output / "preflight_failure.json",
            {
                "protocol_id": PROTOCOL_ID,
                "error_type": type(error).__name__,
                "error": str(error),
                "attempts": 1,
            },
        )
        raise


if __name__ == "__main__":
    main()

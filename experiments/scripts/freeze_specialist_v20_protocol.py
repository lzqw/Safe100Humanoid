"""Build the final v20 protocol from frozen base-only calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from specialist_v20_protocol import (
  FORMAL_ADAPTATION_SEEDS,
  POLICY_METHOD,
  PROTOCOL_ID,
  SPECIALIST_MODES,
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _git_blob(repo: Path, commit: str, relative: str) -> bytes:
  return subprocess.run(
    ["git", "show", f"{commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout


def _relative(repo: Path, path: Path) -> str:
  try:
    return str(path.resolve().relative_to(repo))
  except ValueError as exc:
    raise ValueError(f"v20 frozen input is outside the repository: {path}") from exc


def _validate_calibration(
  *,
  repo: Path,
  mode: str,
  context_path: Path,
  summary_path: Path,
  progress_path: Path,
  preprotocol: dict[str, Any],
  preprotocol_commit: str,
  preprotocol_sha256: str,
) -> dict[str, Any]:
  context = json.loads(context_path.read_text())
  summary = json.loads(summary_path.read_text())
  progress = json.loads(progress_path.read_text())
  calibration = context.get("calibration", {})
  attempts = calibration.get("attempts", [])
  declared_seeds = preprotocol["calibration"][
    f"{mode}_candidate_seeds"
  ]
  minimum_purity = 0.80 if mode == "lateral" else 0.75
  maximum_second = 0.30 if mode == "lateral" else 0.20
  if not attempts:
    raise RuntimeError(f"v20 {mode} calibration has no attempt")
  selected = attempts[-1]
  selected_seed = selected["candidate_seed"]
  selected_index = declared_seeds.index(selected_seed)
  checks = {
    "context_mode": context.get("specialist_mode") == mode,
    "first_declared_qualifier": [
      attempt["candidate_seed"] for attempt in attempts
    ]
    == declared_seeds[: selected_index + 1]
    and not any(attempt.get("qualifies") for attempt in attempts[:-1])
    and selected.get("qualifies") is True,
    "base_policy_only": all(
      attempt.get("base_policy_only") is True for attempt in attempts
    ),
    "episodes": all(
      attempt.get("num_episodes") == 512 for attempt in attempts
    ),
    "success_gate": 0.70 <= selected["success_rate"] <= 0.85,
    "fall_gate": selected["fall_count"] >= 100,
    "purity_gate": selected["target_failure_fraction"] >= minimum_purity,
    "second_failure_gate": selected["second_failure_fraction"]
    <= maximum_second,
    "context_parameter_hash": selected["parameters_sha256"]
    == context.get("parameters_sha256")
    == calibration.get("selected_parameters_sha256")
    == summary.get("parameters_sha256"),
    "selected_seed": selected_seed
    == context.get("calibration_candidate_seed")
    == calibration.get("selected_candidate_seed")
    == summary.get("selected_candidate_seed"),
    "summary_context_file_hash": summary.get("frozen_context_file_sha256")
    == _sha256(context_path),
    "base_checkpoint_hash": calibration.get(
      "base_policy_checkpoint_sha256"
    )
    == preprotocol["sealed_inputs"]["base_policy_checkpoint_sha256"],
    "preprotocol_commit": calibration.get(
      "prospective_protocol_git_commit"
    )
    == preprotocol_commit,
    "preprotocol_hash": calibration.get(
      "prospective_protocol_file_sha256"
    )
    == preprotocol_sha256,
    "progress_matches": progress.get("attempts") == attempts,
    "adapted_policy_absent": calibration.get(
      "adapted_policy_evaluations_used"
    )
    is False,
  }
  failed = [name for name, passed in checks.items() if not passed]
  if failed:
    raise RuntimeError(f"v20 {mode} calibration freeze failed: {failed}")
  return {
    "file": _relative(repo, context_path),
    "file_sha256": _sha256(context_path),
    "parameters_sha256": context["parameters_sha256"],
    "selected_calibration_seed": selected_seed,
    "calibration_summary": {
      "file": _relative(repo, summary_path),
      "sha256": _sha256(summary_path),
    },
    "calibration_progress": {
      "file": _relative(repo, progress_path),
      "sha256": _sha256(progress_path),
    },
    "selected_base_success_rate": selected["success_rate"],
    "selected_fall_count": selected["fall_count"],
    "selected_target_failure_fraction": selected[
      "target_failure_fraction"
    ],
    "selected_second_failure_fraction": selected[
      "second_failure_fraction"
    ],
    "validation_checks": checks,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--preprotocol", type=Path, required=True)
  parser.add_argument("--preprotocol-commit", required=True)
  parser.add_argument("--context-dir", type=Path, required=True)
  parser.add_argument("--calibration-dir", type=Path, required=True)
  parser.add_argument("--base-actor-sha256", required=True)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  preprotocol_path = args.preprotocol.resolve()
  preprotocol = json.loads(preprotocol_path.read_text())
  if preprotocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("unexpected v20 pre-calibration protocol")
  if preprotocol.get("adaptation_seeds") != list(
    FORMAL_ADAPTATION_SEEDS
  ):
    raise RuntimeError("v20 adaptation seed freeze differs")
  preprotocol_relative = _relative(repo, preprotocol_path)
  preprotocol_sha256 = _sha256(preprotocol_path)
  if hashlib.sha256(
    _git_blob(repo, args.preprotocol_commit, preprotocol_relative)
  ).hexdigest() != preprotocol_sha256:
    raise RuntimeError("v20 pre-calibration protocol differs from its commit")

  contexts: dict[str, Any] = {}
  for mode in SPECIALIST_MODES:
    summary_name = (
      "lateral_summary.json"
      if mode == "lateral"
      else "contact_summary.json"
    )
    progress_name = (
      "lateral_progress.json"
      if mode == "lateral"
      else "contact_progress.json"
    )
    contexts[mode] = _validate_calibration(
      repo=repo,
      mode=mode,
      context_path=args.context_dir.resolve() / f"{mode}.json",
      summary_path=args.calibration_dir.resolve() / summary_name,
      progress_path=args.calibration_dir.resolve() / progress_name,
      preprotocol=preprotocol,
      preprotocol_commit=args.preprotocol_commit,
      preprotocol_sha256=preprotocol_sha256,
    )

  source_files = list(preprotocol["sealed_inputs"]["source_files"])
  freeze_script = "experiments/scripts/freeze_specialist_v20_protocol.py"
  if freeze_script not in source_files:
    source_files.append(freeze_script)
  source_hashes = {
    relative: _sha256(repo / relative) for relative in source_files
  }
  postcalibration_packaging_files = {
    freeze_script,
    "experiments/tests/test_specialist_v20.py",
  }
  for relative in source_files:
    if relative in postcalibration_packaging_files:
      continue
    committed_hash = hashlib.sha256(
      _git_blob(repo, args.preprotocol_commit, relative)
    ).hexdigest()
    if committed_hash != source_hashes[relative]:
      raise RuntimeError(
        f"v20 source changed after base-only calibration: {relative}"
      )

  protocol = deepcopy(preprotocol)
  protocol["protocol_revision"] = 1
  protocol["status"] = "prospectively_frozen_before_formal_adaptation"
  protocol["policy_method"] = POLICY_METHOD
  protocol["precalibration_protocol"] = {
    "file": preprotocol_relative,
    "git_commit": args.preprotocol_commit,
    "sha256": preprotocol_sha256,
  }
  protocol["sealed_inputs"].update(
    {
      "implementation_and_calibration_commit": args.preprotocol_commit,
      "formal_protocol_git_binding": (
        "the Git commit containing this protocol.json; every formal launcher "
        "requires that exact HEAD and verifies this file against its Git blob"
      ),
      "base_expanded_actor_sha256": args.base_actor_sha256,
      "contexts": contexts,
      "source_files": source_files,
      "source_file_sha256": source_hashes,
      "postcalibration_packaging_files": sorted(
        postcalibration_packaging_files
      ),
      "calibration_external_logs": {
        "lateral": {
          "sha256": "ef2ef592665eacc0a33bffb7275d0e86562a4b22a9fb2d59960660bee3efcb5d"
        },
        "contact_stability": {
          "sha256": "ba6df453c5ffafd74762709ade3bd0278714a879bea9541984baac236cc4f8d2"
        },
      },
    }
  )
  protocol["fresh_evidence_boundary"] = {
    "base_policy_only_calibration_completed": True,
    "adapted_policy_evaluations_used_for_context_selection": False,
    "formal_adaptation_or_audit_outcomes_seen": False,
    "formal_protocol_must_be_committed_before_adaptation": True,
    "historical_v17_v18_v19_evidence_may_not_be_relabelled": True,
  }
  protocol["execution_isolation"] = {
    "queue_argument_is_one_specialist_only": True,
    "lateral_audit_runs_immediately_after_five_lateral_jobs": True,
    "contact_audit_runs_immediately_after_five_contact_jobs": True,
    "one_specialist_failure_cannot_invalidate_or_suppress_the_other": True,
    "joint_two_specialist_claim_defined": False,
  }
  protocol["infrastructure_history_before_formal_adaptation"] = [
    {
      "mode": "lateral",
      "stage": "calibration_runner_initialization",
      "outcome_observed": False,
      "failure": (
        "missing already-frozen Revision-4 full-rate adapter flags in the "
        "calibration wrapper"
      ),
      "corrected_commit": args.preprotocol_commit,
      "retry_same_context_candidates_and_randomness": True,
      "external_log_sha256": (
        "ef2ef592665eacc0a33bffb7275d0e86562a4b22a9fb2d59960660bee3efcb5d"
      ),
    }
  ]
  rendered = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
  output = args.output.resolve()
  if output.exists() and output.read_text() != rendered:
    raise RuntimeError(f"refusing to overwrite a different v20 protocol: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(rendered)
  print(rendered, end="")


if __name__ == "__main__":
  main()

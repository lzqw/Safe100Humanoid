"""Build a compact, machine-readable v13 retention-anchor result ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DOMAINS = ("D0", "DQH", "DQNH")


def _domain_metrics(result: dict[str, Any]) -> dict[str, Any]:
  return {
    "episodes": int(result["num_episodes"]),
    "seeds": result["seeds"],
    "success_rate": float(result["success_rate"]),
    "fall_rate": float(result["fall_rate"]),
    "mean_return": float(result["mean_return"]),
    "intervention_per_riser": float(result["intervention_per_riser"]),
  }


def _evaluation(matrix: dict[str, Any]) -> dict[str, Any]:
  return {
    domain: _domain_metrics(matrix[domain])
    for domain in DOMAINS
    if domain in matrix
  }


def _round(result: dict[str, Any]) -> dict[str, Any]:
  metrics = result["update_metrics"]
  variants = []
  for variant in result["candidate_variants"]:
    screen = variant.get("screen_eval") or {}
    variants.append(
      {
        "fraction": float(variant["fraction"]),
        "precheck_reasons": variant["precheck_reasons"],
        "screen_score_delta": variant.get("screen_score_delta"),
        "screen": _domain_metrics(screen) if screen else None,
      }
    )
  return {
    "round": int(result["round"]),
    "accepted": bool(result["accepted"]),
    "selected_fraction": result["selected_candidate_fraction"],
    "rejection_reasons": result["rejection_reasons"],
    "total_kl_from_base": float(result["total_kl_from_base"]),
    "old_total_kl_from_base": float(result["old_total_kl_from_base"]),
    "fixed_bank_kl": {
      name: metrics.get(f"{name}_retention_anchor_kl")
      for name in ("d0", "neighbor")
    },
    "anchor_weight_after_adaptation": {
      name: metrics.get(
        f"{name}_retention_anchor_weight_after_adaptation"
      )
      for name in ("d0", "neighbor")
    },
    "old_eval": _evaluation(result.get("old_eval", {})),
    "candidate_eval": _evaluation(result.get("candidate_eval", {})),
    "gate_intervals": result.get("gate_intervals", {}),
    "candidate_variants": variants,
  }


def _arm(path: Path) -> dict[str, Any]:
  result = json.loads(path.read_text())
  evaluated_round = next(
    (
      round_result
      for round_result in result["rounds"]
      if round_result.get("old_eval")
    ),
    None,
  )
  return {
    "summary_file": str(path.resolve()),
    "seed": int(result["seed"]),
    "base_anchor_weight": float(result["base_anchor_weight"]),
    "d0_retention_anchor_weight_initial": float(
      result["d0_retention_anchor_weight_initial"]
    ),
    "neighbor_retention_anchor_weight_initial": float(
      result["neighbor_retention_anchor_weight_initial"]
    ),
    "correction_distillation_weight": float(
      result["correction_distillation_weight"]
    ),
    "candidate_fractions": result["candidate_fractions"],
    "candidate_screen_episodes_per_policy": int(
      result["candidate_screen_num_envs"]
      * result["candidate_screen_repeats"]
    ),
    "formal_gate_episodes_per_domain_per_policy": int(
      evaluated_round["old_eval"]["D0"]["num_episodes"]
    )
    if evaluated_round is not None
    else None,
    "accepted_rounds": [
      int(round_result["round"])
      for round_result in result["rounds"]
      if round_result["accepted"]
    ],
    "baseline_eval": _evaluation(result["baseline_eval"]),
    "rounds": [_round(round_result) for round_result in result["rounds"]],
    "final_eval": _evaluation(result["final_eval"]),
    "final_checkpoint": result["final_checkpoint"],
  }


def _audit(path: Path | None) -> dict[str, Any] | None:
  if path is None:
    return None
  result = json.loads(path.read_text())
  return {
    "file": str(path.resolve()),
    "accepted": bool(result["accepted"]),
    "decision": result["decision"],
    "round": int(result["round"]),
    "seed": int(result["seed"]),
    "episodes_per_domain_per_policy": int(
      result["episodes_per_domain_per_policy"]
    ),
    "rejection_reasons": result["rejection_reasons"],
    "old_eval": _evaluation(result["old_eval"]),
    "candidate_eval": _evaluation(result["candidate_eval"]),
    "gate_intervals": result["gate_intervals"],
    "old_checkpoint_sha256": result["old_checkpoint_sha256"],
    "candidate_checkpoint_sha256": result["candidate_checkpoint_sha256"],
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm-a-summary", type=Path, required=True)
  parser.add_argument("--arm-b-summary", type=Path, required=True)
  parser.add_argument("--d0-bank-manifest", type=Path, required=True)
  parser.add_argument("--neighbor-bank-manifest", type=Path, required=True)
  parser.add_argument("--arm-a-audit", type=Path)
  parser.add_argument("--arm-b-audit", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  for path in (
    args.arm_a_summary,
    args.arm_b_summary,
    args.d0_bank_manifest,
    args.neighbor_bank_manifest,
  ):
    if not path.is_file():
      raise FileNotFoundError(path)
  ledger = {
    "schema_version": 1,
    "scope": (
      "simulation evidence for safe adaptation to an unknown fixed combined "
      "deployment shift; not exhaustive sim-to-real validation"
    ),
    "runtime_cbf_removed": False,
    "policy_gradient_scope": "target-domain on-policy rollouts only",
    "banks": {
      "D0": json.loads(args.d0_bank_manifest.read_text()),
      "DQNH": json.loads(args.neighbor_bank_manifest.read_text()),
    },
    "arms": {
      "A_v12_global_anchor": _arm(args.arm_a_summary),
      "B_state_conditioned_retention": _arm(args.arm_b_summary),
    },
    "final_audits": {
      "A_v12_global_anchor": _audit(args.arm_a_audit),
      "B_state_conditioned_retention": _audit(args.arm_b_audit),
    },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
  print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()

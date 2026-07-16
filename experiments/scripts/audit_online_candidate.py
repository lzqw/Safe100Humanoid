"""Apply the candidate gate to externally evaluated large GPU batches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--online-summary", type=Path, required=True)
  parser.add_argument("--base-template", required=True)
  parser.add_argument("--candidate-template", required=True)
  parser.add_argument("--target-domain", default="DQ")
  parser.add_argument("--neighbor-domain", default="DQN")
  parser.add_argument("--retention-domain", default="D0")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo.resolve()))

  from src.tasks.stairs_cbf.online import candidate_gate

  domains = (args.retention_domain, args.target_domain, args.neighbor_domain)
  old_eval = {}
  candidate_eval = {}
  for domain in domains:
    old_eval[domain] = json.loads(
      Path(args.base_template.format(domain=domain)).read_text()
    )
    candidate_eval[domain] = json.loads(
      Path(args.candidate_template.format(domain=domain)).read_text()
    )
    for result in (old_eval[domain], candidate_eval[domain]):
      episodes = int(result["num_episodes"])
      p = float(result["success_rate"])
      result["success_rate_standard_error"] = math.sqrt(p * (1.0 - p) / episodes)

  online = json.loads(args.online_summary.read_text())
  round_result = online["rounds"][0]
  accepted, reasons = candidate_gate(
    update_metrics=round_result["update_metrics"],
    old_eval=old_eval,
    candidate_eval=candidate_eval,
    base_d0_success=old_eval[args.retention_domain]["success_rate"],
    total_kl_from_base=round_result["total_kl_from_base"],
    parameters_finite=True,
    target_domain=args.target_domain,
    retention_domain=args.retention_domain,
    neighbor_domain=args.neighbor_domain,
  )
  result = {
    "accepted": accepted,
    "rejection_reasons": reasons,
    "target_domain": args.target_domain,
    "retention_domain": args.retention_domain,
    "neighbor_domain": args.neighbor_domain,
    "old_eval": old_eval,
    "candidate_eval": candidate_eval,
    "update_metrics": round_result["update_metrics"],
    "total_kl_from_base": round_result["total_kl_from_base"],
    "decision": "accept" if accepted else "rollback",
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()

"""Persist a candidate only after a machine-readable safety gate accepts it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--gate", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  gate = json.loads(args.gate.read_text())
  if not gate.get("accepted") or gate.get("decision") != "accept":
    raise RuntimeError("refusing to persist a candidate not accepted by the gate")
  if gate.get("rejection_reasons"):
    raise RuntimeError("accepted gate contains rejection reasons")
  evaluated_candidates = {
    Path(str(result.get("checkpoint", ""))).resolve()
    for result in gate.get("candidate_eval", {}).values()
  }
  if evaluated_candidates != {args.candidate.resolve()}:
    raise RuntimeError(
      "gate candidate does not match checkpoint being persisted: "
      f"{sorted(map(str, evaluated_candidates))} != {args.candidate.resolve()}"
    )

  payload = torch.load(args.candidate, map_location="cpu", weights_only=False)
  infos = payload.setdefault("infos", {})
  if not isinstance(infos, dict):
    infos = {"source_infos": infos}
    payload["infos"] = infos
  line_search = infos.get("line_search")
  if isinstance(line_search, dict):
    for key in ("base_checkpoint", "candidate_checkpoint"):
      if key in line_search:
        line_search[key] = Path(str(line_search[key])).name
  infos["online_acceptance"] = {
    "gate_artifact": args.gate.name,
    "target_domain": gate["target_domain"],
    "retention_domain": gate["retention_domain"],
    "neighbor_domain": gate["neighbor_domain"],
    "old_eval": {
      domain: {key: value for key, value in result.items() if key != "checkpoint"}
      for domain, result in gate["old_eval"].items()
    },
    "candidate_eval": {
      domain: {key: value for key, value in result.items() if key != "checkpoint"}
      for domain, result in gate["candidate_eval"].items()
    },
    "update_metrics": gate["update_metrics"],
    "total_kl_from_base": gate["total_kl_from_base"],
    "decision": "accept",
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, args.output)
  print(
    json.dumps(
      {
        "accepted": True,
        "candidate": str(args.candidate.resolve()),
        "gate": str(args.gate.resolve()),
        "output": str(args.output.resolve()),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()

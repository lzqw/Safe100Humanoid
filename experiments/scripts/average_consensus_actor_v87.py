"""Average independently estimated actor deltas from one common checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from proximal_v23_io import actor_state_sha256, file_sha256

METHOD_ID = "paper-cbf-consensus-rollout-delta-v87"
TRAINABLE_PREFIX = "mlp."
CONSENSUS_SIGNATURE_FIELDS = (
  "candidate",
  "context",
  "teacher_arm",
  "num_envs",
  "rollout_steps",
  "training_runtime_filter",
  "training_filter_fraction",
  "training_filter_schedule",
  "filter_group_balanced_advantages",
  "training_action_std",
  "actor_learning_rate",
  "moving_kl_beta",
  "full_batch_sgd_actor",
  "actor_gradient_accumulation_microbatches",
)


def average_actor_deltas(
  base: dict[str, torch.Tensor],
  members: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Return ``base + mean(member - base)`` with strict common-state checks."""
  if len(members) < 2:
    raise ValueError("v87 consensus requires at least two independent members")
  keys = tuple(sorted(base))
  trainable = tuple(name for name in keys if name.startswith(TRAINABLE_PREFIX))
  if not trainable:
    raise ValueError("v87 actor state contains no MLP parameters")
  deltas: list[dict[str, torch.Tensor]] = []
  for member_index, member in enumerate(members, start=1):
    if tuple(sorted(member)) != keys:
      raise ValueError(f"v87 member {member_index} actor keys differ from base")
    member_delta: dict[str, torch.Tensor] = {}
    for name in keys:
      base_value = base[name]
      member_value = member[name]
      if (
        member_value.shape != base_value.shape
        or member_value.dtype != base_value.dtype
      ):
        raise ValueError(f"v87 member {member_index} tensor mismatch: {name}")
      if name not in trainable:
        if not torch.equal(member_value, base_value):
          raise ValueError(
            f"v87 member {member_index} changed frozen actor state: {name}"
          )
        continue
      if not base_value.is_floating_point():
        raise TypeError(f"v87 trainable tensor must be floating point: {name}")
      delta = member_value.to(torch.float64) - base_value.to(torch.float64)
      if not bool(torch.isfinite(delta).all()):
        raise ValueError(f"v87 member {member_index} has non-finite delta")
      member_delta[name] = delta
    deltas.append(member_delta)

  squared_norms = [
    sum(float(value.square().sum()) for value in member.values())
    for member in deltas
  ]
  if any(value <= 0.0 or not math.isfinite(value) for value in squared_norms):
    raise ValueError("v87 member actor delta must be finite and non-zero")
  norms = [math.sqrt(value) for value in squared_norms]
  pairwise_cosines: list[float] = []
  for left in range(len(deltas)):
    for right in range(left + 1, len(deltas)):
      dot = sum(
        float((deltas[left][name] * deltas[right][name]).sum())
        for name in trainable
      )
      pairwise_cosines.append(dot / (norms[left] * norms[right]))

  output = {name: value.detach().clone() for name, value in base.items()}
  consensus_deltas: dict[str, torch.Tensor] = {}
  for name in trainable:
    mean_delta = sum(
      (member[name] for member in deltas),
      torch.zeros_like(deltas[0][name]),
    ) / len(deltas)
    consensus_deltas[name] = mean_delta
    output[name] = (base[name].to(torch.float64) + mean_delta).to(base[name].dtype)
  consensus_norm = math.sqrt(
    sum(float(value.square().sum()) for value in consensus_deltas.values())
  )
  mean_member_norm = sum(norms) / len(norms)
  diagnostics = {
    "member_count": len(members),
    "trainable_tensor_count": len(trainable),
    "member_delta_l2_norms": norms,
    "consensus_delta_l2_norm": consensus_norm,
    "consensus_to_mean_member_norm_ratio": consensus_norm / mean_member_norm,
    "pairwise_delta_cosines": pairwise_cosines,
    "mean_pairwise_delta_cosine": (
      sum(pairwise_cosines) / len(pairwise_cosines)
    ),
    "minimum_pairwise_delta_cosine": min(pairwise_cosines),
    "maximum_pairwise_delta_cosine": max(pairwise_cosines),
    "maximum_absolute_consensus_delta": max(
      float(value.abs().max()) for value in consensus_deltas.values()
    ),
  }
  return output, diagnostics


def _load_checkpoint(path: Path) -> dict[str, Any]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(payload, dict) or not isinstance(
    payload.get("actor_state_dict"), dict
  ):
    raise ValueError(f"v87 checkpoint lacks actor_state_dict: {path}")
  return payload


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
    temporary = Path(handle.name)
  try:
    torch.save(payload, temporary)
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
    mode="w", dir=path.parent, delete=False, encoding="utf-8"
  ) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  try:
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument(
    "--proposal-checkpoint", type=Path, action="append", required=True
  )
  parser.add_argument("--expected-member-count", type=int, default=4)
  parser.add_argument("--output-checkpoint", type=Path, required=True)
  parser.add_argument("--output-manifest", type=Path, required=True)
  args = parser.parse_args()

  if args.output_checkpoint.exists() or args.output_manifest.exists():
    raise FileExistsError("v87 refuses to overwrite an existing output")
  if file_sha256(args.base_checkpoint) != args.expected_base_sha256:
    raise ValueError("v87 base checkpoint SHA-256 mismatch")
  if len(args.proposal_checkpoint) != args.expected_member_count:
    raise ValueError("v87 proposal count differs from the prospective count")

  base_payload = _load_checkpoint(args.base_checkpoint)
  base_actor = base_payload["actor_state_dict"]
  base_actor_sha256 = actor_state_sha256(base_actor)
  proposal_payloads: list[dict[str, Any]] = []
  member_records: list[dict[str, Any]] = []
  reference_signature = None
  for proposal in args.proposal_checkpoint:
    member_base = proposal.with_name("round_00.pt")
    summary_path = proposal.with_name("training_summary.json")
    metrics_path = proposal.with_name("round_metrics.json")
    if not member_base.is_file() or not summary_path.is_file() or not metrics_path.is_file():
      raise FileNotFoundError(f"v87 member provenance is incomplete: {proposal}")
    member_base_actor = _load_checkpoint(member_base)["actor_state_dict"]
    if actor_state_sha256(member_base_actor) != base_actor_sha256:
      raise ValueError(f"v87 member did not start from common actor: {proposal}")
    summary = json.loads(summary_path.read_text())
    if summary.get("base_checkpoint_sha256") != args.expected_base_sha256:
      raise ValueError(f"v87 member base file hash differs: {proposal}")
    signature = {name: summary.get(name) for name in CONSENSUS_SIGNATURE_FIELDS}
    if reference_signature is None:
      reference_signature = signature
    elif signature != reference_signature:
      raise ValueError(f"v87 member training configuration differs: {proposal}")
    round_metrics = json.loads(metrics_path.read_text())
    if not round_metrics or round_metrics[0].get("round") != 1:
      raise ValueError(f"v87 member lacks aligned first update: {proposal}")
    payload = _load_checkpoint(proposal)
    proposal_actor = payload["actor_state_dict"]
    proposal_actor_sha256 = actor_state_sha256(proposal_actor)
    if (
      round_metrics[0].get("post_update_actor_sha256")
      != proposal_actor_sha256
    ):
      raise ValueError(f"v87 proposal actor hash differs from telemetry: {proposal}")
    proposal_payloads.append(payload)
    member_records.append(
      {
        "seed": summary.get("seed"),
        "checkpoint": str(proposal),
        "checkpoint_sha256": file_sha256(proposal),
        "actor_sha256": proposal_actor_sha256,
        "rollout_filter_off_success_count": round_metrics[0]["metrics"].get(
          "rollout_filter_off_success_count"
        ),
        "rollout_filter_off_episode_count": round_metrics[0]["metrics"].get(
          "rollout_filter_off_episode_count"
        ),
      }
    )

  consensus_actor, diagnostics = average_actor_deltas(
    base_actor,
    [payload["actor_state_dict"] for payload in proposal_payloads],
  )
  consensus_actor_sha256 = actor_state_sha256(consensus_actor)
  output_payload = copy.deepcopy(base_payload)
  output_payload["actor_state_dict"] = consensus_actor
  output_payload["proximal_method_id"] = METHOD_ID
  output_payload.setdefault("infos", {})["v87_consensus"] = {
    "method_id": METHOD_ID,
    "base_actor_sha256": base_actor_sha256,
    "consensus_actor_sha256": consensus_actor_sha256,
    "training_configuration": reference_signature,
    "members": member_records,
    "diagnostics": diagnostics,
  }
  _atomic_torch(args.output_checkpoint, output_payload)
  manifest = {
    "schema_version": "paper_dual_v35.v87.consensus.v1",
    "method_id": METHOD_ID,
    "base_checkpoint": str(args.base_checkpoint),
    "base_checkpoint_sha256": args.expected_base_sha256,
    "base_actor_sha256": base_actor_sha256,
    "training_configuration": reference_signature,
    "members": member_records,
    "effective_transition_count": (
      len(member_records)
      * int(reference_signature["num_envs"])
      * int(reference_signature["rollout_steps"])
    ),
    "diagnostics": diagnostics,
    "consensus_checkpoint": str(args.output_checkpoint),
    "consensus_checkpoint_sha256": file_sha256(args.output_checkpoint),
    "consensus_actor_sha256": consensus_actor_sha256,
    "critic_and_optimizer_source": "unchanged_base_checkpoint",
  }
  _atomic_json(args.output_manifest, manifest)
  print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()


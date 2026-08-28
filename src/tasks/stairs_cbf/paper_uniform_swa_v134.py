"""Uniform actor-SWA consolidation for the scaled paper PPO trajectory."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

METHOD_ID = "paper-cbf-dual-uniform-actor-swa-v134"
ACTOR_MLP_PREFIX = "mlp."
V132_SNAPSHOT_SHA256 = (
  "79a11beca8d5a85445d7e7a57f30fc81f45601491c1833306d97924653151cbe",
  "01ec7ea71492555db3ed50608c6b3b9b6216cd129f1435cad8579c9562157a46",
  "f0c18b0965668fb8eab5e3fab6e8f2edc6555c35f9cb65df4b419c4c3df34b91",
  "b69fb8d7c88d072a1e328adf7eea0b317cdd4ff9726c60c4047535ffa1428c04",
  "5db53e0c7d7c6cf92f1e72c9418f81a1933a451a00432de99d15f260ab9529d0",
  "3c1c9fae4abaeda786fc4377bfeebd6b45892ac53dbb0cf985f4707c219953fc",
  "a7fdd4d07dc79f1f001b09ff3638bcc5de000c3804f369dcf13ed61ebd18bde3",
)


def uniform_actor_mlp_average(
  actor_states: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Average only MLP tensors and require every representation tensor equal."""
  if len(actor_states) < 2:
    raise ValueError("v134 SWA requires at least two actor snapshots")
  template = actor_states[-1]
  keys = tuple(template)
  if not keys or not any(key.startswith(ACTOR_MLP_PREFIX) for key in keys):
    raise ValueError("v134 actor state has no MLP tensors")
  if any(tuple(state) != keys for state in actor_states):
    raise ValueError("v134 actor snapshot keys or ordering differ")

  averaged = copy.deepcopy(dict(template))
  mlp_parameter_count = 0
  for key in keys:
    tensors = [state[key].detach().cpu() for state in actor_states]
    reference = tensors[-1]
    if any(
      tensor.shape != reference.shape or tensor.dtype != reference.dtype
      for tensor in tensors
    ):
      raise ValueError(f"v134 actor tensor contract differs for {key}")
    if key.startswith(ACTOR_MLP_PREFIX):
      if not reference.is_floating_point():
        raise ValueError(f"v134 MLP tensor is not floating point: {key}")
      if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError(f"v134 MLP tensor is not finite: {key}")
      accumulator = torch.zeros_like(reference, dtype=torch.float64)
      for tensor in tensors:
        accumulator.add_(tensor.to(dtype=torch.float64))
      averaged[key] = (accumulator / len(tensors)).to(dtype=reference.dtype)
      mlp_parameter_count += reference.numel()
    else:
      if not all(torch.equal(tensor, reference) for tensor in tensors[:-1]):
        raise ValueError(f"v134 non-MLP actor tensor changed across snapshots: {key}")
      averaged[key] = reference.clone()

  snapshot_distances: list[float] = []
  for state in actor_states:
    squared_distance = 0.0
    for key in keys:
      if key.startswith(ACTOR_MLP_PREFIX):
        delta = state[key].detach().cpu().double() - averaged[key].double()
        squared_distance += float(delta.square().sum())
    snapshot_distances.append(math.sqrt(squared_distance))
  selected_squared_distance = 0.0
  for key in keys:
    if key.startswith(ACTOR_MLP_PREFIX):
      delta = averaged[key].double() - template[key].detach().cpu().double()
      selected_squared_distance += float(delta.square().sum())

  return averaged, {
    "v134_method_id": METHOD_ID,
    "v134_snapshot_count": len(actor_states),
    "v134_uniform_snapshot_weight": 1.0 / len(actor_states),
    "v134_mlp_parameter_count": mlp_parameter_count,
    "v134_non_mlp_actor_state_exactly_preserved": True,
    "v134_selected_snapshot_mlp_distance": math.sqrt(
      selected_squared_distance
    ),
    "v134_mean_snapshot_mlp_distance": sum(snapshot_distances)
    / len(snapshot_distances),
    "v134_maximum_snapshot_mlp_distance": max(snapshot_distances),
  }

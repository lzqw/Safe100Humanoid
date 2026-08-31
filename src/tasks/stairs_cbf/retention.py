"""Fixed actor-observation retention banks for conservative online PPO."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import torch


RETENTION_BANK_SCHEMA_VERSION = 1
RETENTION_BANK_KIND = "actor_observation_retention_bank"
MINIMUM_RETENTION_BANK_SIZE = 20_000
MAXIMUM_RETENTION_BANK_SIZE = 50_000


def balanced_stage_quotas(total_size: int, num_stages: int) -> tuple[int, ...]:
  """Split ``total_size`` across stages with counts differing by at most one."""
  if total_size < 1 or num_stages < 1:
    raise ValueError("retention bank size and stage count must be positive")
  quotient, remainder = divmod(total_size, num_stages)
  if quotient == 0:
    raise ValueError("retention bank must contain at least one sample per stage")
  return tuple(
    quotient + int(stage < remainder) for stage in range(num_stages)
  )


def interleave_stage_observations(
  observations_by_stage: Sequence[torch.Tensor],
  *,
  generator: torch.Generator,
) -> torch.Tensor:
  """Shuffle within each stage and round-robin interleave the fixed bank.

  A contiguous training slice is therefore stage-balanced without storing or
  exposing stage labels to the actor loss.
  """
  if not observations_by_stage:
    raise ValueError("at least one retention stage is required")
  first = observations_by_stage[0]
  if first.ndim != 2 or first.shape[0] < 1:
    raise ValueError("retention stage observations must have shape [N, actor_dim]")
  actor_dim = first.shape[1]
  shuffled: list[torch.Tensor] = []
  counts: list[int] = []
  for observations in observations_by_stage:
    if observations.ndim != 2 or observations.shape[1] != actor_dim:
      raise ValueError("all retention stages must share one actor dimension")
    if observations.shape[0] < 1:
      raise ValueError("every retention stage must contain observations")
    if observations.device.type != "cpu":
      raise ValueError("retention bank construction requires CPU observations")
    if not observations.dtype.is_floating_point:
      raise ValueError("retention actor observations must be floating point")
    if not bool(torch.isfinite(observations).all()):
      raise ValueError("retention actor observations contain non-finite values")
    permutation = torch.randperm(observations.shape[0], generator=generator)
    shuffled.append(observations[permutation].contiguous())
    counts.append(observations.shape[0])
  if max(counts) - min(counts) > 1:
    raise ValueError("retention stage counts must differ by at most one")

  stacked = torch.cat(shuffled, dim=0)
  offsets = []
  offset = 0
  for count in counts:
    offsets.append(offset)
    offset += count
  order = [
    offsets[stage] + row
    for row in range(max(counts))
    for stage, count in enumerate(counts)
    if row < count
  ]
  return stacked[torch.tensor(order, dtype=torch.long)].contiguous()


def cyclic_retention_batch(
  observations: torch.Tensor,
  *,
  cursor: int,
  batch_size: int,
) -> tuple[torch.Tensor, int]:
  """Return a deterministic cyclic bank slice and the next cursor."""
  if observations.ndim != 2 or observations.shape[0] < 1:
    raise ValueError("retention observations must have shape [N, actor_dim]")
  if batch_size < 1:
    raise ValueError("retention anchor batch size must be positive")
  size = observations.shape[0]
  if not 0 <= cursor < size:
    raise ValueError("retention anchor cursor is outside the bank")
  indices = (torch.arange(batch_size, device=observations.device) + cursor) % size
  return observations.index_select(0, indices), (cursor + batch_size) % size


def actor_observation_sha256(observations: torch.Tensor) -> str:
  """Hash a canonical CPU float32 representation of actor observations."""
  canonical = observations.detach().to(device="cpu", dtype=torch.float32)
  canonical = canonical.contiguous()
  digest = hashlib.sha256()
  digest.update(str(tuple(canonical.shape)).encode())
  digest.update(canonical.numpy().tobytes())
  return digest.hexdigest()


def validate_retention_observation_bank(
  payload: dict[str, Any],
  *,
  expected_actor_dim: int | None = None,
  expected_domain: str | None = None,
  minimum_size: int = MINIMUM_RETENTION_BANK_SIZE,
  maximum_size: int = MAXIMUM_RETENTION_BANK_SIZE,
) -> tuple[torch.Tensor, dict[str, Any]]:
  """Validate and return actor-only observations plus JSON-safe metadata."""
  if not isinstance(payload, dict):
    raise TypeError("retention bank payload must be a dictionary")
  if payload.get("schema_version") != RETENTION_BANK_SCHEMA_VERSION:
    raise ValueError("unsupported retention bank schema version")
  if payload.get("kind") != RETENTION_BANK_KIND:
    raise ValueError("payload is not an actor-observation retention bank")
  if payload.get("actor_observation_key") != "actor":
    raise ValueError("retention bank must contain only the actor observation group")
  if payload.get("contains_privileged_observations") is not False:
    raise ValueError("retention bank must explicitly exclude privileged observations")
  if minimum_size < 1 or maximum_size < minimum_size:
    raise ValueError("retention bank size limits are inconsistent")

  observations = payload.get("observations")
  if not isinstance(observations, torch.Tensor) or observations.ndim != 2:
    raise ValueError("retention bank observations must be a [N, actor_dim] tensor")
  if observations.device.type != "cpu":
    observations = observations.detach().cpu()
  observations = observations.to(dtype=torch.float32).contiguous()
  size, actor_dim = observations.shape
  if not minimum_size <= size <= maximum_size:
    raise ValueError(
      f"retention bank size {size} is outside [{minimum_size}, {maximum_size}]"
    )
  if actor_dim < 1:
    raise ValueError("retention actor observation dimension must be positive")
  if expected_actor_dim is not None and actor_dim != expected_actor_dim:
    raise ValueError(
      f"retention actor dimension {actor_dim} != expected {expected_actor_dim}"
    )
  if int(payload.get("actor_observation_dim", -1)) != actor_dim:
    raise ValueError("retention bank actor dimension metadata is inconsistent")
  if not bool(torch.isfinite(observations).all()):
    raise ValueError("retention bank contains non-finite actor observations")

  domain = payload.get("domain")
  if not isinstance(domain, str) or not domain:
    raise ValueError("retention bank domain metadata is missing")
  if expected_domain is not None and domain != expected_domain:
    raise ValueError(f"retention bank domain {domain!r} != {expected_domain!r}")
  stage_counts_raw = payload.get("stage_counts")
  if not isinstance(stage_counts_raw, (tuple, list)) or not stage_counts_raw:
    raise ValueError("retention bank stage counts are missing")
  stage_counts = tuple(int(value) for value in stage_counts_raw)
  if any(value < 1 for value in stage_counts) or sum(stage_counts) != size:
    raise ValueError("retention bank stage counts do not cover the bank")
  if max(stage_counts) - min(stage_counts) > 1:
    raise ValueError("retention bank is not stage-balanced")
  if int(payload.get("num_stages", -1)) != len(stage_counts):
    raise ValueError("retention bank stage-count metadata is inconsistent")
  if payload.get("ordering") != "stage_round_robin_v1":
    raise ValueError("retention bank does not guarantee balanced training slices")

  observation_sha256 = actor_observation_sha256(observations)
  if payload.get("observation_sha256") != observation_sha256:
    raise ValueError("retention bank actor-observation checksum differs")
  seed = int(payload.get("seed", -1))
  if seed < 0:
    raise ValueError("retention bank seed must be non-negative")
  if not isinstance(payload.get("runtime_filter"), bool):
    raise ValueError("retention bank runtime-filter metadata is missing")
  if payload.get("policy_mode") != "deterministic_mean":
    raise ValueError("retention bank must be collected with the deterministic mean policy")

  metadata = {
    "schema_version": RETENTION_BANK_SCHEMA_VERSION,
    "kind": RETENTION_BANK_KIND,
    "domain": domain,
    "task": str(payload.get("task", "")),
    "seed": seed,
    "runtime_filter": bool(payload["runtime_filter"]),
    "policy_mode": payload["policy_mode"],
    "actor_observation_key": "actor",
    "actor_observation_dim": actor_dim,
    "contains_privileged_observations": False,
    "size": size,
    "num_stages": len(stage_counts),
    "stage_counts": list(stage_counts),
    "ordering": payload["ordering"],
    "checkpoint_sha256": str(payload.get("checkpoint_sha256", "")),
    "observation_sha256": observation_sha256,
  }
  for key in ("checkpoint", "collection_steps", "created_utc"):
    if key in payload:
      value = payload[key]
      if isinstance(value, (str, int, float, bool)) and (
        not isinstance(value, float) or math.isfinite(value)
      ):
        metadata[key] = value
  return observations, metadata


def increase_anchor_weight_on_budget_violation(
  weight: float,
  observed_kl: float,
  budget: float,
  *,
  learning_rate: float,
  maximum: float,
) -> float:
  """Increase an anchor weight only when its fixed-bank KL exceeds budget."""
  values = torch.tensor(
    [weight, observed_kl, budget, learning_rate, maximum], dtype=torch.float64
  )
  if not bool(torch.isfinite(values).all()):
    raise ValueError("retention anchor adaptation values must be finite")
  if weight < 0.0 or observed_kl < 0.0 or budget < 0.0:
    raise ValueError("retention anchor weights and KL values must be non-negative")
  if learning_rate < 0.0 or maximum <= 0.0 or weight > maximum:
    raise ValueError("retention anchor adaptation bounds are inconsistent")
  violation = max(0.0, observed_kl - budget)
  return min(maximum, weight + learning_rate * violation)

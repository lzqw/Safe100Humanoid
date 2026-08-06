"""Pure statistics and gates for the prospective diagonal specialist audit."""

from __future__ import annotations

from typing import Any

import torch


def hierarchical_paired_scene_interval(
  seed_episode_deltas: list[torch.Tensor],
  *,
  bootstrap_samples: int = 10000,
  bootstrap_seed: int = 0,
) -> tuple[float, float, float]:
  """Return a two-level paired-bootstrap interval for one specialist scene.

  The formal protocol has three independently adapted actors. Each bootstrap
  draw resamples those adaptation seeds and then resamples paired episodes
  within each selected seed. It never resamples baseline and final outcomes
  independently.
  """
  if len(seed_episode_deltas) != 3:
    raise ValueError("scene bootstrap requires exactly three adaptation seeds")
  if bootstrap_samples < 1000:
    raise ValueError("formal scene bootstrap requires at least 1000 samples")
  if any(group.ndim != 1 for group in seed_episode_deltas):
    raise ValueError("scene bootstrap groups must be one-dimensional")
  lengths = {int(group.numel()) for group in seed_episode_deltas}
  if len(lengths) != 1 or 0 in lengths:
    raise ValueError("scene bootstrap groups must have one non-zero size")

  values = torch.stack(seed_episode_deltas).to(dtype=torch.float64, device="cpu")
  if not bool(torch.isfinite(values).all()):
    raise ValueError("scene bootstrap deltas contain non-finite values")
  seed_count, episode_count = values.shape
  generator = torch.Generator(device="cpu")
  generator.manual_seed(bootstrap_seed)
  means: list[torch.Tensor] = []
  chunk_size = 100
  for start in range(0, bootstrap_samples, chunk_size):
    count = min(chunk_size, bootstrap_samples - start)
    sampled_seed = torch.randint(
      seed_count, (count, seed_count), generator=generator
    )
    samples = torch.empty(
      (count, seed_count, episode_count), dtype=torch.float64
    )
    for draw in range(count):
      for seed_slot in range(seed_count):
        source_seed = int(sampled_seed[draw, seed_slot])
        episode_ids = torch.randint(
          episode_count, (episode_count,), generator=generator
        )
        samples[draw, seed_slot] = values[source_seed, episode_ids]
    means.append(samples.mean(dim=(1, 2)))
  bootstrap_means = torch.cat(means)
  lower, upper = torch.quantile(
    bootstrap_means,
    torch.tensor([0.025, 0.975], dtype=bootstrap_means.dtype),
  )
  return float(values.mean()), float(lower), float(upper)


def independent_diagonal_scene_gate(
  *,
  diagonal_success_delta: float,
  per_seed_success_deltas: list[float],
  diagonal_fall_delta: float,
  d0_success_delta: float,
) -> dict[str, Any]:
  """Apply the frozen gate to one scene without any cross-scene condition."""
  if len(per_seed_success_deltas) != 3:
    raise ValueError("independent scene gate requires exactly three seed deltas")
  values = torch.tensor(
    [
      diagonal_success_delta,
      *per_seed_success_deltas,
      diagonal_fall_delta,
      d0_success_delta,
    ],
    dtype=torch.float64,
  )
  if not bool(torch.isfinite(values).all()):
    raise ValueError("independent scene gate inputs must be finite")
  positive_seed_count = sum(delta > 0.0 for delta in per_seed_success_deltas)
  criteria = {
    "mean_diagonal_success_gain_positive": diagonal_success_delta > 0.0,
    "at_least_two_of_three_seed_gains_positive": positive_seed_count >= 2,
    "diagonal_fall_increase_at_most_3pp": diagonal_fall_delta <= 0.03,
    "d0_success_drop_at_most_5pp": d0_success_delta >= -0.05,
  }
  return {
    "criteria": criteria,
    "passed": all(criteria.values()),
    "positive_adaptation_seed_count": positive_seed_count,
    "thresholds": {
      "mean_diagonal_success_delta_strictly_above": 0.0,
      "minimum_positive_adaptation_seed_count": 2,
      "maximum_diagonal_fall_increase": 0.03,
      "minimum_d0_success_delta": -0.05,
    },
  }

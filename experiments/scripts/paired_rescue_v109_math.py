"""Pure paired-trajectory targets for matched CBF rescue episodes."""

from __future__ import annotations

import torch


def paired_rescue_action_trace(
  off_nominal: torch.Tensor,
  on_nominal: torch.Tensor,
  on_safe: torch.Tensor,
  on_intervened: torch.Tensor,
  *,
  pre_horizon: int = 20,
  post_horizon: int = 50,
  pre_decay: float = 0.9,
  correction_scale: float = 0.05,
) -> dict[str, torch.Tensor | int | float]:
  """Align a rescued on/off pair around its first causal filter intervention.

  After the first intervention, the target is the filter-on executed action
  minus the same-time filter-off nominal action. Before it, the first actual
  filter correction is traced backward with exponential decay. Each episode
  receives unit total weight so long trajectories cannot dominate the batch.
  """
  tensors = (off_nominal, on_nominal, on_safe)
  if any(value.ndim != 2 for value in tensors):
    raise ValueError("v109 paired actions must have shape [T, A]")
  if any(value.shape[1] != off_nominal.shape[1] for value in tensors[1:]):
    raise ValueError("v109 paired action dimensions must match")
  if on_intervened.ndim != 1 or on_intervened.dtype != torch.bool:
    raise ValueError("v109 intervention trace must be boolean [T]")
  if pre_horizon < 0 or post_horizon < 0:
    raise ValueError("v109 trace horizons must be non-negative")
  if not 0.0 < pre_decay <= 1.0 or correction_scale <= 0.0:
    raise ValueError("v109 trace decay and correction scale are invalid")

  shared_length = min(
    len(off_nominal), len(on_nominal), len(on_safe), len(on_intervened)
  )
  interventions = on_intervened[:shared_length].nonzero(as_tuple=False).flatten()
  if shared_length == 0 or not len(interventions):
    return {
      "indices": torch.empty(0, dtype=torch.long),
      "corrections": off_nominal.new_empty((0, off_nominal.shape[1])),
      "weights": off_nominal.new_empty((0,)),
      "first_intervention_step": -1,
      "pre_transition_count": 0,
      "post_transition_count": 0,
      "shared_length": shared_length,
    }

  first = int(interventions[0])
  start = max(0, first - int(pre_horizon))
  stop = min(shared_length, first + int(post_horizon) + 1)
  indices = torch.arange(start, stop, dtype=torch.long)
  corrections = on_safe[indices] - off_nominal[indices]
  pre = indices < first
  if bool(pre.any()):
    first_correction = on_safe[first] - on_nominal[first]
    lags = (first - indices[pre]).to(off_nominal.dtype)
    decay = torch.pow(
      off_nominal.new_tensor(float(pre_decay)), lags
    ).unsqueeze(-1)
    corrections[pre] = decay * first_correction.unsqueeze(0)

  norms = torch.linalg.vector_norm(corrections, dim=-1)
  effective = norms > 1.0e-7
  indices = indices[effective]
  corrections = corrections[effective]
  norms = norms[effective]
  weights = torch.clamp(norms / float(correction_scale), 0.0, 1.0)
  if len(weights):
    weights = weights / weights.sum().clamp_min(1.0e-12)
  retained_pre = int((indices < first).sum())
  return {
    "indices": indices,
    "corrections": corrections,
    "weights": weights,
    "first_intervention_step": first,
    "pre_transition_count": retained_pre,
    "post_transition_count": int(len(indices) - retained_pre),
    "shared_length": shared_length,
  }

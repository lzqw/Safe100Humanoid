"""Pure tensor checks for the v87 common-base actor-delta consensus."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/scripts"))
SPEC = importlib.util.spec_from_file_location(
  "average_consensus_actor_v87",
  REPO / "experiments/scripts/average_consensus_actor_v87.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _actor(weight: torch.Tensor) -> dict[str, torch.Tensor]:
  return {
    "obs_normalizer._mean": torch.zeros(1, 2),
    "obs_normalizer.count": torch.tensor(7),
    "distribution.std_param": torch.ones(1),
    "mlp.0.weight": weight.clone(),
    "mlp.0.bias": torch.tensor((1.0, -1.0)),
  }


def test_v87_averages_deltas_and_preserves_frozen_actor_state() -> None:
  base = _actor(torch.tensor(((1.0, 2.0), (3.0, 4.0))))
  first = {name: value.clone() for name, value in base.items()}
  second = {name: value.clone() for name, value in base.items()}
  first["mlp.0.weight"] += torch.tensor(((2.0, 0.0), (0.0, -2.0)))
  second["mlp.0.weight"] += torch.tensor(((0.0, 4.0), (-4.0, 0.0)))
  first["mlp.0.bias"] += torch.tensor((2.0, -2.0))
  second["mlp.0.bias"] += torch.tensor((0.0, 4.0))

  consensus, diagnostics = MODULE.average_actor_deltas(base, [first, second])
  assert torch.equal(
    consensus["mlp.0.weight"],
    base["mlp.0.weight"] + torch.tensor(((1.0, 2.0), (-2.0, -1.0))),
  )
  assert torch.equal(
    consensus["mlp.0.bias"],
    base["mlp.0.bias"] + torch.tensor((1.0, 1.0)),
  )
  assert torch.equal(
    consensus["obs_normalizer._mean"], base["obs_normalizer._mean"]
  )
  assert diagnostics["member_count"] == 2
  assert diagnostics["consensus_delta_l2_norm"] > 0.0


def test_v87_rejects_member_that_changes_frozen_normalizer() -> None:
  base = _actor(torch.ones(2, 2))
  changed = {name: value.clone() for name, value in base.items()}
  changed["mlp.0.weight"] += 1.0
  changed["obs_normalizer._mean"] += 0.1
  with pytest.raises(ValueError, match="changed frozen actor state"):
    MODULE.average_actor_deltas(base, [changed, changed])

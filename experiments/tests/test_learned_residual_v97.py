from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_learned_residual_v97 import (  # noqa: E402
  ACTION_DIM,
  RESIDUAL_INPUT_DIM,
  LearnedCbfResidual,
  residual_teacher_target,
)


def test_residual_head_is_exact_zero_and_bounded() -> None:
  model = LearnedCbfResidual(0.25)
  features = torch.randn(7, RESIDUAL_INPUT_DIM)
  initial = model(features)
  assert torch.equal(initial, torch.zeros(7, ACTION_DIM))
  with torch.no_grad():
    model.network[-1].bias.fill_(100.0)
  bounded = model(features)
  assert bool((bounded <= 0.25).all())
  assert bool((bounded >= -0.25).all())


def test_teacher_target_moves_toward_filter_and_clamps() -> None:
  current = torch.tensor([[0.10, -0.10]])
  correction = torch.tensor([[0.40, -0.40]])
  target = residual_teacher_target(
    current, correction, eta=0.5, max_residual=0.25
  )
  assert torch.equal(target, torch.tensor([[0.25, -0.25]]))

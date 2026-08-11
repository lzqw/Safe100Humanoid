"""Small artifact helpers shared only by v23 entrypoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def actor_state(actor) -> dict[str, torch.Tensor]:
  return {
    key: value.detach().clone() for key, value in actor.state_dict().items()
  }


def actor_state_sha256(state: dict[str, torch.Tensor]) -> str:
  digest = hashlib.sha256()
  for name in sorted(state):
    value = state[name].detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
  return digest.hexdigest()


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()

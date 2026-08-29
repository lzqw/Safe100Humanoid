"""Export a frozen 405-D stair actor without starting a simulator or robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from proximal_v23_io import actor_state_sha256, file_sha256

V139_CHECKPOINT_SHA256 = (
  "323f1e00b58d379b8746c0191a44272f2e1df134139050417c56e733cc484728"
)
V139_ACTOR_SHA256 = (
  "4a4926d9227c31fb239ceead6c39bed61304d1f2c7e3a47aea510e060cee2acd"
)
ACTOR_OBSERVATION_DIM = 405
ACTION_DIM = 12
HIDDEN_DIMS = (512, 256, 128)
ONNX_OPSET = 18


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Export the deterministic mean of a frozen stair-policy checkpoint. "
      "This command performs no simulation or hardware control."
    )
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--expected-checkpoint-sha256", default=V139_CHECKPOINT_SHA256
  )
  parser.add_argument("--expected-actor-sha256", default=V139_ACTOR_SHA256)
  parser.add_argument(
    "--checkpoint-label",
    default=(
      "results/online/paper_dual_v35/clearance_mixed_v139/checkpoints/"
      "selected_round_01.pt"
    ),
  )
  parser.add_argument(
    "--artifact-label",
    default=(
      "results/online/paper_dual_v35/deployment_pipeline_v139/"
      "deployment_artifacts/v139_actor_405x12.onnx"
    ),
  )
  parser.add_argument(
    "--manifest",
    type=Path,
    help="Defaults to the ONNX path with '.manifest.json' as its suffix.",
  )
  return parser.parse_args()


def _load_actor(checkpoint: Path):
  from rsl_rl.models import MLPModel
  from tensordict import TensorDict

  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  actor_state = payload.get("actor_state_dict")
  if not isinstance(actor_state, dict):
    raise TypeError("checkpoint actor_state_dict is missing or is not a dict")

  sample_obs = TensorDict(
    {"actor": torch.zeros(1, ACTOR_OBSERVATION_DIM)}, batch_size=[1]
  )
  actor = MLPModel(
    obs=sample_obs,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=HIDDEN_DIMS,
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.05,
      "std_type": "scalar",
    },
  )
  actor.load_state_dict(actor_state, strict=True)
  actor.eval()
  return actor, actor_state


def _export(actor, output: Path) -> None:
  model = actor.as_onnx(verbose=False).cpu().eval()
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),
    str(output),
    export_params=True,
    opset_version=ONNX_OPSET,
    verbose=False,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={},
    dynamo=False,
  )


def _validate_export(actor, output: Path) -> dict[str, Any]:
  import onnx
  from onnx.reference import ReferenceEvaluator

  model = onnx.load(str(output))
  onnx.checker.check_model(model)
  evaluator = ReferenceEvaluator(model)

  generator = torch.Generator(device="cpu")
  generator.manual_seed(139)
  mean = actor.obs_normalizer._mean.detach().cpu()
  std = actor.obs_normalizer._std.detach().cpu()
  probes = [torch.zeros_like(mean), mean]
  probes.extend(
    mean + std * torch.randn(mean.shape, generator=generator) for _ in range(6)
  )

  max_abs_error = 0.0
  with torch.inference_mode():
    for probe in probes:
      expected = actor({"actor": probe}).detach().cpu().numpy()
      observed = evaluator.run(None, {"obs": probe.numpy()})[0]
      max_abs_error = max(
        max_abs_error,
        float(np.max(np.abs(expected - observed))),
      )
  if not np.isfinite(max_abs_error) or max_abs_error > 1.0e-5:
    raise RuntimeError(
      f"ONNX deterministic equivalence failed: max_abs_error={max_abs_error}"
    )
  return {
    "onnx_checker_passed": True,
    "fixed_probe_count": len(probes),
    "fixed_probe_seed": 139,
    "maximum_absolute_output_error": max_abs_error,
    "acceptance_tolerance": 1.0e-5,
  }


def main() -> None:
  args = _parse_args()
  checkpoint = args.checkpoint.resolve()
  output = args.output.resolve()
  manifest = (
    args.manifest.resolve()
    if args.manifest is not None
    else output.with_suffix(".manifest.json")
  )
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

  checkpoint_hash = file_sha256(checkpoint)
  if checkpoint_hash != args.expected_checkpoint_sha256:
    raise RuntimeError(
      "checkpoint SHA-256 differs: "
      f"{checkpoint_hash} != {args.expected_checkpoint_sha256}"
    )

  actor, actor_state = _load_actor(checkpoint)
  actor_hash = actor_state_sha256(actor_state)
  if actor_hash != args.expected_actor_sha256:
    raise RuntimeError(
      f"actor SHA-256 differs: {actor_hash} != {args.expected_actor_sha256}"
    )

  _export(actor, output)
  validation = _validate_export(actor, output)
  record = {
    "schema_version": 1,
    "artifact_role": "frozen_v139_real_robot_initialization_candidate",
    "checkpoint_label": args.checkpoint_label,
    "checkpoint_sha256": checkpoint_hash,
    "actor_state_sha256": actor_hash,
    "artifact_label": args.artifact_label,
    "onnx_sha256": file_sha256(output),
    "onnx_size_bytes": output.stat().st_size,
    "inference": {
      "input_name": "obs",
      "input_shape": [1, ACTOR_OBSERVATION_DIM],
      "output_name": "actions",
      "output_shape": [1, ACTION_DIM],
      "deterministic_policy_mean": True,
      "hidden_dims": list(HIDDEN_DIMS),
      "activation": "elu",
      "observation_normalization_embedded": True,
      "onnx_opset": ONNX_OPSET,
    },
    "validation": validation,
    "hardware_control_ready": False,
    "warning": (
      "This ONNX file is only the nominal actor. Real execution still requires "
      "the exact five-frame observation contract, 12-to-29 joint mapping, "
      "hardware state estimation, fixed-stair localization, runtime CBF, "
      "command/joint limits, watchdog, operator takeover, and emergency stop."
    ),
  }
  manifest.parent.mkdir(parents=True, exist_ok=True)
  manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(record, indent=2))


if __name__ == "__main__":
  main()

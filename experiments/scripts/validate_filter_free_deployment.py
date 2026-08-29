"""Export and validate one fixed filter-free actor for the offline G1 bridge.

This command never starts a simulator and never opens a robot transport.  It
checks the exact five-frame bridge contract, PyTorch/ONNX action parity, the
12-to-29 joint target mapping, and single-thread CPU latency against the 20 ms
policy period.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.export_stairs_actor_onnx import (  # noqa: E402
  ACTION_DIM,
  ACTOR_OBSERVATION_DIM,
  ONNX_OPSET,
  _export,
  _load_actor,
  _validate_export,
)
from experiments.scripts.proximal_v23_io import (  # noqa: E402
  actor_state_sha256,
  file_sha256,
)
from src.tasks.stairs_cbf.real_robot_reference import (  # noqa: E402
  ACTOR_TERM_WIDTHS,
  GAIT_PERIOD_S,
  G1_DEFAULT_JOINT_POSITION,
  HISTORY_LENGTH,
  LOWER_BODY_JOINT_INDICES,
  POLICY_PERIOD_S,
  ActorObservationHistory,
  embed_lower_body_target,
  nominal_lower_body_target,
)

DEFAULT_WARMUP_STEPS = 20
DEFAULT_MEASURED_STEPS = 100
PARITY_TOLERANCE = 1.0e-5


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Offline ONNX/bridge parity and latency validation for a deterministic "
      "405-D stair actor."
    )
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--report", type=Path)
  parser.add_argument("--expected-checkpoint-sha256", required=True)
  parser.add_argument("--expected-actor-sha256", required=True)
  parser.add_argument("--checkpoint-label", required=True)
  parser.add_argument("--artifact-label", required=True)
  parser.add_argument("--context", choices=("F1", "F2", "F3"), required=True)
  parser.add_argument("--training-seed", type=int, required=True)
  parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
  parser.add_argument(
    "--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS
  )
  parser.add_argument(
    "--policy-deadline-ms", type=float, default=POLICY_PERIOD_S * 1000.0
  )
  return parser.parse_args()


def _bridge_state(step: int) -> dict[str, Any]:
  """Create a deterministic finite bridge input without simulator state."""
  phase = float(step) * 0.17
  joint_index = torch.arange(29, dtype=torch.float32)
  action_index = torch.arange(ACTION_DIM, dtype=torch.float32)
  default = torch.tensor(G1_DEFAULT_JOINT_POSITION, dtype=torch.float32)
  return {
    "base_ang_vel": torch.tensor(
      [0.08 * np.sin(phase), -0.06 * np.cos(phase), 0.04 * np.sin(phase / 2)],
      dtype=torch.float32,
    ),
    "projected_gravity": torch.tensor(
      [0.025 * np.sin(phase), -0.035 * np.cos(phase), -0.998],
      dtype=torch.float32,
    ),
    "command": torch.tensor(
      [0.24 + 0.03 * np.sin(phase), 0.02 * np.cos(phase), 0.04 * np.sin(phase)],
      dtype=torch.float32,
    ),
    "episode_step": step,
    "joint_position": default + 0.04 * torch.sin(joint_index * 0.23 + phase),
    "joint_velocity": 0.12 * torch.cos(joint_index * 0.19 + phase),
    "previous_raw_action": 0.18 * torch.sin(action_index * 0.31 + phase),
  }


def _bridge_observations(count: int) -> list[torch.Tensor]:
  history = ActorObservationHistory()
  observations = [history.push(**_bridge_state(step)) for step in range(count)]
  if any(obs.shape != (1, ACTOR_OBSERVATION_DIM) for obs in observations):
    raise RuntimeError("bridge produced an actor observation with the wrong shape")
  return observations


def _percentile(values: list[float], percentile: float) -> float:
  return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _latency_summary(
  step: Callable[[int], None],
  *,
  warmup_steps: int,
  measured_steps: int,
  deadline_ms: float,
) -> dict[str, Any]:
  for index in range(warmup_steps):
    step(index)
  timings_ms: list[float] = []
  for index in range(measured_steps):
    started = time.perf_counter_ns()
    step(index + warmup_steps)
    timings_ms.append((time.perf_counter_ns() - started) / 1.0e6)
  return {
    "warmup_steps": warmup_steps,
    "measured_steps": measured_steps,
    "mean_ms": float(np.mean(timings_ms)),
    "median_ms": float(np.median(timings_ms)),
    "p95_ms": _percentile(timings_ms, 95.0),
    "p99_ms": _percentile(timings_ms, 99.0),
    "maximum_ms": max(timings_ms),
    "policy_deadline_ms": deadline_ms,
    "p95_within_policy_deadline": _percentile(timings_ms, 95.0) <= deadline_ms,
  }


def _bridge_parity(actor, evaluator) -> dict[str, Any]:
  observations = _bridge_observations(HISTORY_LENGTH + 7)
  output_error = 0.0
  lower_target_error = 0.0
  full_target_error = 0.0
  observation_hash = hashlib.sha256()
  with torch.inference_mode():
    for observation in observations:
      observation_hash.update(observation.contiguous().numpy().tobytes())
      torch_action = actor({"actor": observation}).detach().cpu()
      onnx_action = torch.from_numpy(
        evaluator.run(None, {"obs": observation.numpy()})[0]
      )
      output_error = max(
        output_error,
        float(torch.max(torch.abs(torch_action - onnx_action))),
      )
      torch_lower = nominal_lower_body_target(torch_action[0])
      onnx_lower = nominal_lower_body_target(onnx_action[0])
      lower_target_error = max(
        lower_target_error,
        float(torch.max(torch.abs(torch_lower - onnx_lower))),
      )
      torch_full = embed_lower_body_target(
        torch_lower, G1_DEFAULT_JOINT_POSITION
      )
      onnx_full = embed_lower_body_target(onnx_lower, G1_DEFAULT_JOINT_POSITION)
      full_target_error = max(
        full_target_error,
        float(torch.max(torch.abs(torch_full - onnx_full))),
      )
      if not torch.equal(
        torch_full[ACTION_DIM:], onnx_full[ACTION_DIM:]
      ):
        raise RuntimeError("uncontrolled joint posture differs across runtimes")
  passed = max(output_error, lower_target_error, full_target_error) <= PARITY_TOLERANCE
  return {
    "passed": passed,
    "acceptance_tolerance": PARITY_TOLERANCE,
    "bridge_probe_count": len(observations),
    "bridge_observation_sha256": observation_hash.hexdigest(),
    "actor_observation_shape": [1, ACTOR_OBSERVATION_DIM],
    "history_length": HISTORY_LENGTH,
    "term_widths": [[name, width] for name, width in ACTOR_TERM_WIDTHS],
    "action_shape": [1, ACTION_DIM],
    "controlled_joint_indices": list(LOWER_BODY_JOINT_INDICES),
    "maximum_actor_output_error": output_error,
    "maximum_lower_body_target_error": lower_target_error,
    "maximum_full_body_target_error": full_target_error,
  }


def _latency(actor, evaluator, args: argparse.Namespace) -> dict[str, Any]:
  states = [
    _bridge_state(index)
    for index in range(args.warmup_steps + args.measured_steps)
  ]
  approved_default = G1_DEFAULT_JOINT_POSITION

  torch_history = ActorObservationHistory()

  def torch_step(index: int) -> None:
    observation = torch_history.push(**states[index])
    with torch.inference_mode():
      action = actor({"actor": observation})[0]
    embed_lower_body_target(
      nominal_lower_body_target(action), approved_default
    )

  onnx_history = ActorObservationHistory()

  def onnx_step(index: int) -> None:
    observation = onnx_history.push(**states[index])
    action = evaluator.run(None, {"obs": observation.numpy()})[0][0]
    embed_lower_body_target(
      nominal_lower_body_target(action), approved_default
    )

  previous_thread_count = torch.get_num_threads()
  try:
    torch.set_num_threads(1)
    torch_result = _latency_summary(
      torch_step,
      warmup_steps=args.warmup_steps,
      measured_steps=args.measured_steps,
      deadline_ms=args.policy_deadline_ms,
    )
    onnx_result = _latency_summary(
      onnx_step,
      warmup_steps=args.warmup_steps,
      measured_steps=args.measured_steps,
      deadline_ms=args.policy_deadline_ms,
    )
  finally:
    torch.set_num_threads(previous_thread_count)
  return {
    "measurement_scope": (
      "five-frame observation assembly + deterministic actor + 12-to-29 "
      "position-target mapping; no robot transport and runtime CBF disabled"
    ),
    "device": "cpu",
    "torch_threads": 1,
    "policy_period_ms": POLICY_PERIOD_S * 1000.0,
    "gait_period_ms": GAIT_PERIOD_S * 1000.0,
    "pytorch_backend": torch_result,
    "onnx_backend": {
      "backend": "onnx.reference.ReferenceEvaluator",
      "note": (
        "Portable correctness backend, not an optimized production runtime; "
        "latency is reported without being used as a deployment claim."
      ),
      **onnx_result,
    },
  }


def main() -> None:
  args = _parse_args()
  if args.warmup_steps < 1 or args.measured_steps < 2:
    raise ValueError("latency measurement requires warmup >= 1 and samples >= 2")
  if args.policy_deadline_ms <= 0.0:
    raise ValueError("policy deadline must be positive")
  checkpoint = args.checkpoint.resolve()
  output = args.output.resolve()
  report_path = (
    args.report.resolve()
    if args.report is not None
    else output.with_suffix(".validation.json")
  )
  checkpoint_hash = file_sha256(checkpoint)
  if checkpoint_hash != args.expected_checkpoint_sha256:
    raise RuntimeError(
      f"checkpoint SHA-256 differs: {checkpoint_hash} != "
      f"{args.expected_checkpoint_sha256}"
    )
  actor, actor_state = _load_actor(checkpoint)
  actor_hash = actor_state_sha256(actor_state)
  if actor_hash != args.expected_actor_sha256:
    raise RuntimeError(
      f"actor SHA-256 differs: {actor_hash} != {args.expected_actor_sha256}"
    )

  _export(actor, output)
  base_validation = _validate_export(actor, output)
  import onnx
  from onnx.reference import ReferenceEvaluator

  model = onnx.load(str(output))
  evaluator = ReferenceEvaluator(model)
  bridge_parity = _bridge_parity(actor, evaluator)
  latency = _latency(actor, evaluator, args)
  report = {
    "schema_version": 1,
    "artifact_role": "fixed_round_4_dual_filter_free_deployment_candidate",
    "context": args.context,
    "training_seed": args.training_seed,
    "round": 4,
    "selection_used": False,
    "checkpoint_label": args.checkpoint_label,
    "checkpoint_sha256": checkpoint_hash,
    "actor_state_sha256": actor_hash,
    "artifact_label": args.artifact_label,
    "onnx_sha256": file_sha256(output),
    "onnx_size_bytes": output.stat().st_size,
    "onnx_opset": ONNX_OPSET,
    "runtime_filter_enabled": False,
    "base_export_validation": base_validation,
    "bridge_parity": bridge_parity,
    "latency": latency,
    "hardware_control_ready": False,
    "warning": (
      "Offline bridge evidence only. Robot execution still requires reviewed "
      "state estimation, stair localization, command/joint limits, watchdog, "
      "operator takeover, emergency stop, and tethered validation."
    ),
  }
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  if not bridge_parity["passed"]:
    raise RuntimeError(f"offline bridge parity failed; see {report_path}")
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()

"""Fine-tune only the actor output layer from a frozen v38 rescue dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from cbf_teacher_v31_protocol import (
  CLEARANCE_BARRIER_SLOPE,
  CONTEXTS,
  FILTER_ALPHA,
  RECOVERY_DISTANCE_M,
  TASK_ID,
  environment_parameters,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _distill_actor,
  _git,
  _seed_everything,
)

METHOD_ID = "last-layer-multi-seed-rescue-distillation-v40"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--expected-dataset-sha256", required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--teacher-eta", type=float, default=0.25)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--moving-kl-beta", type=float, default=0.5)
  parser.add_argument("--max-reference-kl", type=float, default=2.5e-4)
  parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
  parser.add_argument("--minibatches", type=int, default=4)
  parser.add_argument("--max-grad-norm", type=float, default=0.5)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _validate_sha256(value: str, name: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError(f"{name} must contain 64 hexadecimal digits")
  return normalized


def _load_dataset(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  required = {
    "observations",
    "nominal_actions",
    "safe_actions",
    "would_intervene",
    "environment_ids",
    "trust_observations",
    "rescued_environment_ids",
    "actor_sha256",
    "training_seeds",
  }
  missing = sorted(required - set(payload))
  if missing:
    raise RuntimeError(f"v40 rescue dataset is missing {missing}")
  dataset = {
    key: payload[key]
    for key in (
      "observations",
      "nominal_actions",
      "safe_actions",
      "would_intervene",
      "environment_ids",
    )
  }
  transitions = len(dataset["observations"])
  if transitions < 1 or any(len(value) != transitions for value in dataset.values()):
    raise RuntimeError("v40 rescue transition tensors have inconsistent lengths")
  if payload.get("teacher_state_source") != "on":
    raise RuntimeError("v40 requires shielded rescue teacher states")
  if payload.get("trust_state_source") != "off-success":
    raise RuntimeError("v40 requires off-success trust observations")
  return payload, dataset


def main() -> None:
  args = _parse_args()
  if not 0.0 < args.teacher_eta <= 1.0:
    raise ValueError("v40 teacher eta must lie in (0, 1]")
  if not 1.0e-6 <= args.actor_learning_rate <= 1.0e-4:
    raise ValueError("v40 actor learning rate is outside the safe range")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v40 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.01:
    raise ValueError("v40 requires a positive reference-KL cap <= 0.01")
  if args.minibatches < 1 or args.max_grad_norm <= 0.0:
    raise ValueError("v40 minibatches and gradient norm must be positive")
  repo = args.repo.resolve()
  dataset_path = args.dataset.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v40 requires a clean committed worktree")
  if output.exists():
    raise FileExistsError(output)
  if not dataset_path.is_file() or not checkpoint.is_file():
    raise FileNotFoundError("v40 dataset or checkpoint is missing")
  expected_dataset_sha = _validate_sha256(
    args.expected_dataset_sha256, "v40 expected dataset SHA-256"
  )
  expected_checkpoint_sha = _validate_sha256(
    args.expected_base_sha256, "v40 expected checkpoint SHA-256"
  )
  dataset_sha = file_sha256(dataset_path)
  checkpoint_sha = file_sha256(checkpoint)
  if dataset_sha != expected_dataset_sha:
    raise RuntimeError("v40 rescue dataset SHA-256 differs")
  if checkpoint_sha != expected_checkpoint_sha:
    raise RuntimeError("v40 base checkpoint SHA-256 differs")
  payload, dataset = _load_dataset(dataset_path)
  output.mkdir(parents=True)
  started = time.monotonic()
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": METHOD_ID,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "optimization_seed": args.optimization_seed,
      "dataset_sha256": dataset_sha,
      "base_checkpoint_sha256": checkpoint_sha,
    },
  )

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.environment_v31 import configure_v31_context

  _seed_everything(args.optimization_seed)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  configure_v31_context(
    env_cfg,
    context=args.context,
    runtime_filter=False,
    context_spec=environment_parameters(args.context),
    clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
    recovery_distance_m=RECOVERY_DISTANCE_M,
    filter_alpha=FILTER_ALPHA,
  )
  env_cfg.scene.num_envs = 2
  env_cfg.seed = args.optimization_seed
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  agent_cfg = load_rl_cfg(TASK_ID)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
      raise RuntimeError("v40 task has no runner")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    initial_actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
    if initial_actor_hash != payload["actor_sha256"]:
      raise RuntimeError("v40 dataset actor differs from the loaded base actor")
    _seed_everything(args.optimization_seed)
    training, optimizer = _distill_actor(
      runner.alg.actor,
      dataset,
      payload["rescued_environment_ids"],
      payload["trust_observations"],
      eta=args.teacher_eta,
      learning_rate=args.actor_learning_rate,
      moving_kl_beta=args.moving_kl_beta,
      max_reference_kl=args.max_reference_kl,
      actor_update_scope="last-layer",
      epochs=args.epochs,
      minibatches=args.minibatches,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )
    final_actor_state = actor_state(runner.alg.actor)
    final_actor_hash = actor_state_sha256(final_actor_state)
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_payload["actor_state_dict"] = {
      key: value.cpu() for key, value in final_actor_state.items()
    }
    source_payload["rescue_distill_optimizer_state_dict"] = optimizer.state_dict()
    source_payload["iter"] = int(source_payload.get("iter", 0)) + args.epochs
    infos = dict(source_payload.get("infos") or {})
    infos["last_layer_rescue_distill_v40"] = {
      "method_id": METHOD_ID,
      "source_git_commit": _git(repo, "rev-parse", "HEAD"),
      "source_dataset_sha256": dataset_sha,
      "source_training_seeds": payload["training_seeds"],
      "optimization_seed": args.optimization_seed,
      "actor_update_scope": "last-layer",
      "max_reference_kl": args.max_reference_kl,
    }
    source_payload["infos"] = infos
    candidate_path = output / "candidate.pt"
    _atomic_torch(candidate_path, source_payload)
    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "context": args.context,
      "optimization_seed": args.optimization_seed,
      "source_dataset": str(dataset_path),
      "source_dataset_sha256": dataset_sha,
      "source_training_seeds": payload["training_seeds"],
      "base_checkpoint_sha256": checkpoint_sha,
      "initial_actor_sha256": initial_actor_hash,
      "candidate_path": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "candidate_actor_sha256": final_actor_hash,
      "actor_update_scope": "last-layer",
      "teacher_eta": args.teacher_eta,
      "actor_learning_rate": args.actor_learning_rate,
      "moving_kl_beta": args.moving_kl_beta,
      "max_reference_kl": args.max_reference_kl,
      "epochs": args.epochs,
      "minibatches": args.minibatches,
      "training": training,
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()

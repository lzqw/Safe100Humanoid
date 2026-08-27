"""Take one full-batch output-layer step on a frozen v41 elite dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from cbf_teacher_v31_protocol import (
  CLEARANCE_BARRIER_SLOPE,
  CONTEXTS,
  FILTER_ALPHA,
  RECOVERY_DISTANCE_M,
  TASK_ID,
  environment_parameters,
)
from elite_self_imitation_v41 import METHOD_ID as SOURCE_METHOD_ID
from elite_self_imitation_v41 import _distill_elite
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _seed_everything,
)

METHOD_ID = "full-batch-filter-free-elite-self-imitation-v42"
MULTI_DATASET_METHOD_ID = "six-seed-full-batch-filter-free-elite-self-imitation-v43"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--expected-dataset-sha256", required=True)
  parser.add_argument("--additional-dataset", type=Path, action="append", default=[])
  parser.add_argument(
    "--expected-additional-dataset-sha256", action="append", default=[]
  )
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--moving-kl-beta", type=float, default=0.5)
  parser.add_argument("--max-reference-kl", type=float, default=1.0e-4)
  parser.add_argument("--max-grad-norm", type=float, default=0.5)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _normalized_sha(value: str, label: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError(f"{label} must contain 64 hexadecimal digits")
  return normalized


def main() -> None:
  args = _parse_args()
  if not 1.0e-6 <= args.actor_learning_rate <= 1.0e-4:
    raise ValueError("v42 actor learning rate is outside the safe range")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v42 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.01 or args.max_grad_norm <= 0.0:
    raise ValueError("v42 KL cap and gradient norm must be positive")
  repo = args.repo.resolve()
  dataset_paths = [
    args.dataset.resolve(),
    *(path.resolve() for path in args.additional_dataset),
  ]
  expected_dataset_shas = [
    args.expected_dataset_sha256,
    *args.expected_additional_dataset_sha256,
  ]
  if len(dataset_paths) != len(expected_dataset_shas):
    raise ValueError("each additional dataset requires an expected SHA-256")
  method_id = MULTI_DATASET_METHOD_ID if len(dataset_paths) > 1 else METHOD_ID
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v42 requires a clean committed worktree")
  if not all(path.is_file() for path in dataset_paths) or not checkpoint.is_file():
    raise FileNotFoundError("v42 dataset or checkpoint is missing")
  if output.exists():
    raise FileExistsError(output)
  dataset_shas = [file_sha256(path) for path in dataset_paths]
  checkpoint_sha = file_sha256(checkpoint)
  for index, (dataset_sha, expected_sha) in enumerate(
    zip(dataset_shas, expected_dataset_shas, strict=True)
  ):
    if dataset_sha != _normalized_sha(
      expected_sha, f"v42 expected dataset {index} SHA-256"
    ):
      raise RuntimeError(f"v42 elite dataset {index} SHA-256 differs")
  if checkpoint_sha != _normalized_sha(
    args.expected_base_sha256, "v42 expected checkpoint SHA-256"
  ):
    raise RuntimeError("v42 base checkpoint SHA-256 differs")
  required = {
    "method_id",
    "training_seeds",
    "actor_sha256",
    "observations",
    "sampled_actions",
    "elite",
  }
  payloads = []
  source_training_seeds: list[int] = []
  seen_training_seeds: set[int] = set()
  actor_hash: str | None = None
  for index, dataset_path in enumerate(dataset_paths):
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    missing = sorted(required - set(payload))
    if missing:
      raise RuntimeError(f"v42 elite dataset {index} is missing {missing}")
    if payload["method_id"] != SOURCE_METHOD_ID:
      raise RuntimeError(f"v42 source dataset {index} method differs")
    observations_part = payload["observations"]
    sampled_actions_part = payload["sampled_actions"]
    elite_part = payload["elite"]
    if (
      observations_part.ndim != 2
      or sampled_actions_part.ndim != 2
      or elite_part.ndim != 1
      or len(observations_part) != len(sampled_actions_part)
      or len(observations_part) != len(elite_part)
    ):
      raise RuntimeError(f"v42 elite dataset {index} tensor shapes differ")
    if payloads and (
      observations_part.shape[1] != payloads[0]["observations"].shape[1]
      or sampled_actions_part.shape[1] != payloads[0]["sampled_actions"].shape[1]
    ):
      raise RuntimeError("v42 elite dataset feature shapes differ")
    payload_actor_hash = payload["actor_sha256"]
    if actor_hash is None:
      actor_hash = payload_actor_hash
    elif actor_hash != payload_actor_hash:
      raise RuntimeError("v42 elite datasets use different actors")
    training_seeds = [int(seed) for seed in payload["training_seeds"]]
    duplicate_seeds = sorted(seen_training_seeds.intersection(training_seeds))
    if duplicate_seeds:
      raise RuntimeError(f"v42 elite datasets repeat training seeds {duplicate_seeds}")
    seen_training_seeds.update(training_seeds)
    source_training_seeds.extend(training_seeds)
    payloads.append(payload)
  observations = torch.cat([payload["observations"] for payload in payloads])
  sampled_actions = torch.cat([payload["sampled_actions"] for payload in payloads])
  elite = torch.cat([payload["elite"].bool() for payload in payloads])
  output.mkdir(parents=True)
  started = time.monotonic()
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": method_id,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "source_dataset_sha256": dataset_shas[0],
      "source_dataset_sha256s": dataset_shas,
      "base_checkpoint_sha256": checkpoint_sha,
      "optimization_seed": args.optimization_seed,
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
      raise RuntimeError("v42 task has no runner")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    initial_actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
    if initial_actor_hash != actor_hash:
      raise RuntimeError("v42 dataset actor differs from the base actor")
    _seed_everything(args.optimization_seed)
    training, optimizer = _distill_elite(
      runner.alg.actor,
      observations,
      sampled_actions,
      elite,
      learning_rate=args.actor_learning_rate,
      moving_kl_beta=args.moving_kl_beta,
      max_reference_kl=args.max_reference_kl,
      epochs=1,
      minibatches=1,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )
    offline_loss_delta = (
      training["after"]["elite_smooth_l1"]
      - training["before"]["elite_smooth_l1"]
    )
    offline_gate_passed = offline_loss_delta < 0.0
    final_actor_state = actor_state(runner.alg.actor)
    final_actor_hash = actor_state_sha256(final_actor_state)
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_payload["actor_state_dict"] = {
      key: value.cpu() for key, value in final_actor_state.items()
    }
    source_payload["elite_optimizer_state_dict"] = optimizer.state_dict()
    source_payload["iter"] = int(source_payload.get("iter", 0)) + 1
    infos = dict(source_payload.get("infos") or {})
    info_key = (
      "multi_dataset_elite_self_imitation_v43"
      if len(dataset_paths) > 1
      else "full_batch_elite_self_imitation_v42"
    )
    infos[info_key] = {
      "method_id": method_id,
      "source_git_commit": _git(repo, "rev-parse", "HEAD"),
      "source_dataset_sha256": dataset_shas[0],
      "source_dataset_sha256s": dataset_shas,
      "source_training_seeds": source_training_seeds,
      "optimization_seed": args.optimization_seed,
      "offline_gate_passed": offline_gate_passed,
    }
    source_payload["infos"] = infos
    candidate_path = output / "candidate.pt"
    _atomic_torch(candidate_path, source_payload)
    summary = {
      "schema_version": 1,
      "method_id": method_id,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "context": args.context,
      "source_dataset": str(dataset_paths[0]),
      "source_datasets": [str(path) for path in dataset_paths],
      "source_dataset_sha256": dataset_shas[0],
      "source_dataset_sha256s": dataset_shas,
      "source_training_seeds": source_training_seeds,
      "optimization_seed": args.optimization_seed,
      "base_checkpoint_sha256": checkpoint_sha,
      "initial_actor_sha256": initial_actor_hash,
      "candidate_path": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "candidate_actor_sha256": final_actor_hash,
      "actor_learning_rate": args.actor_learning_rate,
      "moving_kl_beta": args.moving_kl_beta,
      "max_reference_kl": args.max_reference_kl,
      "epochs": 1,
      "minibatches": 1,
      "offline_elite_loss_delta": offline_loss_delta,
      "offline_gate_passed": offline_gate_passed,
      "training": training,
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()

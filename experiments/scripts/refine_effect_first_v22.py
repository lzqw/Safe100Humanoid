"""One fixed-eight-round v22 adaptation with best-so-far validation selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from online_refine_stairs import (
  _actor_state,
  _actor_state_sha256,
  _collect_and_update_specialist,
  _evaluate_state,
  _file_sha256,
  _policy_step_metrics,
  _save_checkpoint,
)
from refine_deployment_v21 import (
  V21_ACTOR_LAYER_MULTIPLIERS,
  _bank_invariant_reasons,
  _finite_actor_state,
  _seed_rollout,
  _validate_algorithm,
)
from specialist_v22_protocol import (
  CANDIDATE_CONFIRM_EPISODES,
  CANDIDATE_D0_EPISODES,
  CANDIDATE_FRACTIONS,
  CANDIDATE_SCREEN_EPISODES,
  CONTEXT_ADAPTATION_SEEDS,
  CONTEXT_RESTART_BALANCE_PROFILES,
  CONTEXT_VALIDATION_SEEDS,
  CONTEXTS,
  DUAL_ROLLOUT_BATCHES,
  EVAL_BATCH_SIZE,
  FAILURE_DISCOVERY_MAX_ROLLOUTS,
  FAILURE_START_FRACTION,
  MODES,
  NORMAL_FAILURE_SUCCESS_SLOTS,
  NUM_ENVS,
  POLICY_METHOD,
  PROTOCOL_ID,
  ROLLOUT_STEPS,
  ROUNDS,
  SUCCESS_START_FRACTION,
  VALIDATION_EPISODES,
  V22_CONTEXT_SCHEMA_VERSION,
  candidate_confirmation_seed,
  candidate_d0_seed,
  candidate_confirmation_gate,
  candidate_screen_seed,
  dual_rollout_seed,
  failure_discovery_seed,
  select_best_so_far,
)

TRAINING_SOURCE_FILES = (
  "src/tasks/velocity/mdp/observations.py",
  "src/tasks/stairs_cbf/actions.py",
  "src/tasks/stairs_cbf/command.py",
  "src/tasks/stairs_cbf/config.py",
  "src/tasks/stairs_cbf/deployment_context.py",
  "src/tasks/stairs_cbf/hard_cases.py",
  "src/tasks/stairs_cbf/mdp.py",
  "src/tasks/stairs_cbf/online.py",
  "experiments/scripts/evaluate_online_stairs.py",
  "experiments/scripts/online_refine_stairs.py",
  "experiments/scripts/refine_deployment_v21.py",
  "experiments/scripts/specialist_v21_protocol.py",
  "experiments/scripts/specialist_v22_protocol.py",
  "experiments/scripts/refine_effect_first_v22.py",
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--deployment-context", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--context-id", choices=CONTEXTS, required=True)
  parser.add_argument("--mode", choices=tuple(MODES.values()), required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
  parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
  parser.add_argument("--maximum-rounds", type=int, default=ROUNDS)
  parser.add_argument(
    "--candidate-screen-episodes", type=int, default=CANDIDATE_SCREEN_EPISODES
  )
  parser.add_argument(
    "--candidate-confirm-episodes", type=int, default=CANDIDATE_CONFIRM_EPISODES
  )
  parser.add_argument(
    "--candidate-fractions", nargs="+", type=float, default=CANDIDATE_FRACTIONS
  )
  parser.add_argument(
    "--d0-check-num-episodes", type=int, default=CANDIDATE_D0_EPISODES
  )
  parser.add_argument(
    "--validation-num-episodes", type=int, default=VALIDATION_EPISODES
  )
  parser.add_argument(
    "--failure-start-fraction", type=float, default=FAILURE_START_FRACTION
  )
  parser.add_argument(
    "--success-start-fraction", type=float, default=SUCCESS_START_FRACTION
  )
  parser.add_argument("--failure-policy-weight", type=float, default=1.0)
  parser.add_argument("--success-policy-weight", type=float, default=1.25)
  parser.add_argument("--bank-capacity", type=int, default=256)
  parser.add_argument("--success-pool-capacity", type=int, default=1024)
  parser.add_argument(
    "--failure-discovery-max-rollouts",
    type=int,
    default=FAILURE_DISCOVERY_MAX_ROLLOUTS,
  )
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
  parser.add_argument("--fall-penalty-weight", type=float, default=-100.0)
  parser.add_argument("--fall-redistribution-horizon", type=int, default=100)
  parser.add_argument("--fall-redistribution-decay", type=float, default=0.97)
  parser.add_argument("--fall-redistribution-amount", type=float, default=2.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--gate-device", default="cuda:0")
  return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
  exact = {
    "mode": (args.mode, MODES[args.context_id]),
    "seed": (args.seed, CONTEXT_ADAPTATION_SEEDS[args.context_id]),
    "num_envs": (args.num_envs, NUM_ENVS),
    "rollout_steps": (args.rollout_steps, ROLLOUT_STEPS),
    "maximum_rounds": (args.maximum_rounds, ROUNDS),
    "candidate_fractions": (tuple(args.candidate_fractions), CANDIDATE_FRACTIONS),
    "candidate_screen_episodes": (
      args.candidate_screen_episodes,
      CANDIDATE_SCREEN_EPISODES,
    ),
    "candidate_confirm_episodes": (
      args.candidate_confirm_episodes,
      CANDIDATE_CONFIRM_EPISODES,
    ),
    "d0_check_num_episodes": (
      args.d0_check_num_episodes,
      CANDIDATE_D0_EPISODES,
    ),
    "validation_num_episodes": (
      args.validation_num_episodes,
      VALIDATION_EPISODES,
    ),
    "failure_start_fraction": (
      args.failure_start_fraction,
      FAILURE_START_FRACTION,
    ),
    "success_start_fraction": (
      args.success_start_fraction,
      SUCCESS_START_FRACTION,
    ),
    "failure_discovery_max_rollouts": (
      args.failure_discovery_max_rollouts,
      FAILURE_DISCOVERY_MAX_ROLLOUTS,
    ),
    "failure_policy_weight": (args.failure_policy_weight, 1.0),
    "success_policy_weight": (args.success_policy_weight, 1.25),
    "actor_learning_rate": (args.actor_learning_rate, 5.0e-6),
    "critic_learning_rate": (args.critic_learning_rate, 1.0e-4),
    "fall_penalty_weight": (args.fall_penalty_weight, -100.0),
    "fall_redistribution_horizon": (args.fall_redistribution_horizon, 100),
    "fall_redistribution_decay": (args.fall_redistribution_decay, 0.97),
    "fall_redistribution_amount": (args.fall_redistribution_amount, 2.0),
  }
  failed = {
    name: {"actual": actual, "required": required}
    for name, (actual, required) in exact.items()
    if actual != required
  }
  if failed:
    raise ValueError(f"v22 training configuration mismatch: {failed}")


def _v22_bank_invariant_reasons(
  *,
  mode: str,
  failure_bank,
  success_pool,
  success_bank,
  minimum_required: int,
  restart_balance_profile: str,
) -> list[str]:
  """Require diversity only over causal axes applicable to the context.

  The pure L_effect context has one frozen signed bias/pulse direction.  Its
  direction signs remain exact failure/success match fields and diagnostics,
  but forcing counter-direction coverage would manufacture a second context.
  Stage, support-foot, and growth coverage remain mandatory in both the raw
  failure bank and its exactly matched subset.  Contact keeps the unchanged
  v19/v21 full-diversity invariant.
  """
  expected_profile = CONTEXT_RESTART_BALANCE_PROFILES[
    "L_effect" if mode == "lateral" else "C_effect"
  ]
  if restart_balance_profile != expected_profile:
    raise ValueError(
      "v22 restart balance profile differs from the context mechanism"
    )
  if mode != "lateral":
    return _bank_invariant_reasons(
      mode=mode,
      failure_bank=failure_bank,
      success_pool=success_pool,
      success_bank=success_bank,
      minimum_required=minimum_required,
      require_full_diversity=True,
    )

  reasons = _bank_invariant_reasons(
    mode=mode,
    failure_bank=failure_bank,
    success_pool=success_pool,
    success_bank=success_bank,
    minimum_required=minimum_required,
    require_full_diversity=False,
  )
  failure = failure_bank.audit_metadata()
  matched_failure_indices = {
    entry.matched_failure_index
    for entry in success_bank.entries
    if entry.matched_failure_index is not None
    and 0 <= entry.matched_failure_index < len(failure_bank.entries)
  }
  matched_failures = [
    failure_bank.entries[index] for index in sorted(matched_failure_indices)
  ]
  if not {"early", "mid", "late"}.issubset(failure["riser_stage_counts"]):
    reasons.append("lateral bank lacks early/mid/late riser coverage")
  if set(failure["support_foot_counts"]) != {"0", "1"}:
    reasons.append("lateral bank lacks both support feet")
  if not {"low", "high"}.issubset(failure["error_growth_bin_counts"]):
    reasons.append("lateral bank lacks low/high error-growth coverage")
  if not {"early", "mid", "late"}.issubset(
    {entry.riser_stage for entry in matched_failures}
  ):
    reasons.append("matched lateral pairs lack early/mid/late riser coverage")
  if {entry.support_foot for entry in matched_failures} != {0, 1}:
    reasons.append("matched lateral pairs lack both support feet")
  if {
    "high" if entry.error_growth_rate >= 0.25 else "low"
    for entry in matched_failures
  } != {"low", "high"}:
    reasons.append("matched lateral pairs lack low/high error-growth coverage")
  return reasons


def _git_output(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _validate_frozen_protocol(
  args: argparse.Namespace,
  *,
  repo: Path,
  checkpoint: Path,
  context_path: Path,
  context: dict[str, Any],
) -> dict[str, Any]:
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if current_commit != args.protocol_commit:
    raise RuntimeError("v22 training HEAD differs from its frozen protocol")
  if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("v22 training requires a clean tracked worktree")
  relative = protocol_path.relative_to(repo)
  frozen_blob = subprocess.run(
    ["git", "show", f"{current_commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  protocol_sha256 = hashlib.sha256(frozen_blob).hexdigest()
  if protocol_sha256 != _file_sha256(protocol_path):
    raise RuntimeError("v22 training protocol differs from its committed blob")
  sealed = protocol.get("sealed_inputs", {})
  declared_context = sealed.get("contexts", {}).get(args.context_id, {})
  expected_status = f"prospectively_frozen_before_{args.context_id}_adaptation"
  checks = {
    "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
    "context_schema": protocol.get("context_schema_version")
    == V22_CONTEXT_SCHEMA_VERSION,
    "revision": isinstance(protocol.get("protocol_revision"), int)
    and protocol["protocol_revision"] >= 1,
    "status": protocol.get("status") == expected_status,
    "context_id": context.get("context_id") == args.context_id,
    "context_mode": context.get("specialist_mode") == args.mode,
    "runtime_seed": args.seed == CONTEXT_ADAPTATION_SEEDS[args.context_id],
    "declared_seed": protocol.get("adaptation_seeds", {}).get(args.context_id)
    == args.seed,
    "base_checkpoint": sealed.get("base_policy_checkpoint_sha256")
    == _file_sha256(checkpoint),
    "context_file": declared_context.get("file_sha256")
    == _file_sha256(context_path),
    "context_parameters": declared_context.get("parameters_sha256")
    == context["parameters_sha256"],
    "selected_seed": declared_context.get("selected_calibration_seed")
    == context["calibration"]["selected_candidate_seed"],
    "beta_zero": protocol.get("training", {}).get("matched_success_beta") == 0.0,
    "single_branch": protocol.get("training", {}).get(
      "control_or_parallel_comparison_branch"
    )
    is False,
    "restart_balance_profile": protocol.get("training", {}).get(
      "restart_balance_profiles", {}
    ).get(args.context_id)
    == CONTEXT_RESTART_BALANCE_PROFILES[args.context_id],
  }
  if args.context_id == "C_effect":
    checks["lateral_gate_dependency"] = protocol.get(
      "conditional_execution", {}
    ).get("lateral_final_gate_passed") is True
  failed = [name for name, passed in checks.items() if not passed]
  if failed:
    raise RuntimeError(f"v22 frozen training protocol check failed: {failed}")
  return {
    "path": str(protocol_path),
    "relative_path": str(relative),
    "sha256": protocol_sha256,
    "git_commit": current_commit,
    "checks": checks,
  }


def _write_validation_rows(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames = [
    "context_id",
    "round",
    "actor_sha256",
    "accepted_checkpoint",
    "d0_safe",
    "success_rate",
    "fall_rate",
    "success_delta_from_pi0",
    "fall_delta_from_pi0",
    "fall_constraint_passed",
    "best_so_far_after_evaluation",
  ]
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  _validate_args(args)
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import configure_v19_observable_refinement_runner
  from src.tasks.stairs_cbf.deployment_context import (
    V19_SPECIALIST_FAILURE_TYPES,
    apply_frozen_deployment_context,
    load_calibrated_v22_context,
  )
  from src.tasks.stairs_cbf.hard_cases import (
    SPECIALIST_FAILURE_BANK_KIND,
    SPECIALIST_SUCCESS_BANK_KIND,
    SPECIALIST_SUCCESS_POOL_KIND,
    HardCaseStateBank,
    v19_restart_pair_feasibility,
  )
  from src.tasks.stairs_cbf.online import (
    backtrack_actor_state,
    specialist_d0_retention_gate,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  context_path = args.deployment_context.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  context = load_calibrated_v22_context(context_path)
  restart_balance_profile = CONTEXT_RESTART_BALANCE_PROFILES[args.context_id]
  frozen_protocol = _validate_frozen_protocol(
    args,
    repo=repo,
    checkpoint=checkpoint,
    context_path=context_path,
    context=context,
  )
  source_file_sha256 = {
    relative: _file_sha256(repo / relative) for relative in TRAINING_SOURCE_FILES
  }
  output_dir = args.output_dir.resolve()
  if (output_dir / "specialist_summary.json").exists():
    raise RuntimeError("refusing to rerun a completed v22 training directory")
  output_dir.mkdir(parents=True, exist_ok=True)

  task = "Unitree-G1-Stairs-Online-DQHMED"
  env_cfg = load_env_cfg(task)
  context_metadata = apply_frozen_deployment_context(env_cfg, context, role="target")
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env_cfg.actions["joint_pos"].enabled = True
  env_cfg.rewards["cbf_dual"].weight = 0.0
  env_cfg.rewards["fall_termination"].weight = args.fall_penalty_weight

  agent_cfg = load_rl_cfg(task)
  configure_v19_observable_refinement_runner(agent_cfg)
  agent_cfg.seed = args.seed
  agent_cfg.num_steps_per_env = args.rollout_steps
  alg_cfg = agent_cfg.algorithm
  alg_cfg.actor_learning_rate = args.actor_learning_rate
  alg_cfg.critic_learning_rate = args.critic_learning_rate
  alg_cfg.actor_layer_multipliers = V21_ACTOR_LAYER_MULTIPLIERS
  alg_cfg.log_std_learning_rate = 0.0
  alg_cfg.std_scale_from_base = 0.35
  alg_cfg.pre_intervention_weight = 0.0
  alg_cfg.intervention_advantage_weight = 0.0
  alg_cfg.base_anchor_weight = 0.0
  alg_cfg.d0_retention_anchor_weight = 0.0
  alg_cfg.neighbor_retention_anchor_weight = 0.0
  alg_cfg.safe_bc_weight = 0.0
  alg_cfg.correction_distillation_weight = 0.0
  alg_cfg.task_first_constrained = False
  alg_cfg.brief_ppo_refinement = True
  alg_cfg.failure_focused_refinement = True
  alg_cfg.observable_failure_conditioned_refinement = True
  alg_cfg.actor_new_feature_count = 5
  alg_cfg.actor_new_feature_learning_rate_multiplier = 1.0
  alg_cfg.freeze_legacy_actor_input_columns = True
  alg_cfg.kl_early_stopping = True
  alg_cfg.fall_redistribution_horizon = args.fall_redistribution_horizon
  alg_cfg.fall_redistribution_decay = args.fall_redistribution_decay
  alg_cfg.fall_redistribution_amount = args.fall_redistribution_amount
  alg_cfg.hard_case_policy_weight = args.failure_policy_weight
  alg_cfg.success_counterexample_policy_weight = args.success_policy_weight
  alg_cfg.matched_success_preservation_beta = 0.0
  alg_cfg.clip_param = 0.05
  alg_cfg.desired_kl = 0.003
  alg_cfg.num_learning_epochs = 1
  alg_cfg.num_mini_batches = 4
  alg_cfg.schedule = "fixed"
  alg_cfg.entropy_coef = 0.0
  alg_cfg.normalize_advantage_per_mini_batch = True

  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  terminal_fall_penalty = args.fall_penalty_weight * env.unwrapped.step_dt
  if not math.isclose(terminal_fall_penalty, -2.0, abs_tol=1.0e-12):
    raise RuntimeError("v22 terminal scalar fall penalty must equal -2")
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v22 task has no custom online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  initial_actor_state = _actor_state(runner.alg.actor)
  initial_actor_sha256 = _actor_state_sha256(initial_actor_state)
  first_layer_key = "mlp.0.weight"
  legacy_width = warm_start["source_actor_width"]
  zero_column_max_abs = float(
    runner.alg.actor.state_dict()[first_layer_key][:, -5:].abs().max()
  )
  if (
    warm_start["zero_initialized_actor_columns"] != 5
    or warm_start["pi0_exact_preservation_proof"] is not True
    or zero_column_max_abs != 0.0
  ):
    raise RuntimeError("v22 actor expansion did not exactly preserve pi0")
  structural_checks = _validate_algorithm(runner, beta=0.0)

  failure_bank = HardCaseStateBank(
    capacity=args.bank_capacity,
    bank_kind=SPECIALIST_FAILURE_BANK_KIND,
    source_domain="DQHMED",
    context_sha256=context["parameters_sha256"],
    specialist_mode=args.mode,
  )
  success_pool = HardCaseStateBank(
    capacity=args.success_pool_capacity,
    bank_kind=SPECIALIST_SUCCESS_POOL_KIND,
    source_domain="DQHMED",
    context_sha256=context["parameters_sha256"],
    specialist_mode=args.mode,
  )
  success_bank = HardCaseStateBank(
    capacity=args.bank_capacity,
    bank_kind=SPECIALIST_SUCCESS_BANK_KIND,
    source_domain="DQHMED",
    context_sha256=context["parameters_sha256"],
    specialist_mode=args.mode,
  )
  specialist_generator = torch.Generator(device="cpu")
  required_failure_starts = round(args.num_envs * args.failure_start_fraction)
  required_success_starts = round(args.num_envs * args.success_start_fraction)
  if (
    args.num_envs - required_failure_starts - required_success_starts,
    required_failure_starts,
    required_success_starts,
  ) != NORMAL_FAILURE_SUCCESS_SLOTS:
    raise RuntimeError("v22 must realize exactly 40/12/12 environment slots")
  obs, _ = env.reset()

  d0_seed = candidate_d0_seed(args.context_id)
  baseline_eval = _evaluate_state(
    runner,
    initial_actor_state,
    domains=("D0", "DQHMED"),
    num_envs=args.d0_check_num_episodes,
    num_episodes=args.d0_check_num_episodes,
    seed=d0_seed,
    device=args.gate_device,
    runtime_filter=True,
    deployment_context=context_path,
    v19_context=context_path,
  )
  (output_dir / "baseline_candidate_gate_eval.json").write_text(
    json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n"
  )

  validation_seed = CONTEXT_VALIDATION_SEEDS[args.context_id]
  validation_root = output_dir / "validation_monitor" / "raw" / "round_000"
  baseline_validation = _evaluate_state(
    runner,
    initial_actor_state,
    domains=("DQHMED",),
    num_envs=EVAL_BATCH_SIZE,
    num_episodes=EVAL_BATCH_SIZE,
    seed=validation_seed,
    repeats=VALIDATION_EPISODES // EVAL_BATCH_SIZE,
    device=args.gate_device,
    runtime_filter=True,
    artifact_dir=validation_root,
    resume=True,
    deployment_context=context_path,
    v19_context=context_path,
  )["DQHMED"]
  validation_rows: list[dict[str, Any]] = [
    {
      "context_id": args.context_id,
      "round": 0,
      "actor_sha256": initial_actor_sha256,
      "accepted_checkpoint": True,
      "d0_safe": True,
      "success_rate": float(baseline_validation["success_rate"]),
      "fall_rate": float(baseline_validation["fall_rate"]),
      "success_delta_from_pi0": 0.0,
      "fall_delta_from_pi0": 0.0,
      "fall_constraint_passed": True,
      "best_so_far_after_evaluation": True,
    }
  ]

  discovery: list[dict[str, Any]] = []
  for discovery_index in range(args.failure_discovery_max_rollouts):
    before_discovery = runner.snapshot_candidate_state()
    rollout_seed = failure_discovery_seed(args.context_id, discovery_index)
    _seed_rollout(rollout_seed, specialist_generator)
    obs, metrics, _ = _collect_and_update_specialist(
      runner,
      obs,
      critic_only=False,
      specialist_mode=args.mode,
      failure_bank=failure_bank,
      success_pool=success_pool,
      success_bank=success_bank,
      failure_fraction=0.0,
      success_fraction=0.0,
      specialist_generator=specialist_generator,
      minimum_riser=1,
      protocol_version=19,
      defer_update=True,
      restart_balance_profile=restart_balance_profile,
    )
    runner.restore_candidate_state(before_discovery)
    metrics.update(
      discovery_rollout=discovery_index + 1,
      rollout_seed=rollout_seed,
      parameters_restored_after_discovery=True,
    )
    discovery.append(metrics)
    reasons = _v22_bank_invariant_reasons(
      mode=args.mode,
      failure_bank=failure_bank,
      success_pool=success_pool,
      success_bank=success_bank,
      minimum_required=12,
      restart_balance_profile=restart_balance_profile,
    )
    joint = v19_restart_pair_feasibility(
      failure_bank,
      success_bank,
      12,
      balance_profile=restart_balance_profile,
    )
    metrics["joint_balance_preflight"] = joint
    if not joint["passed"]:
      reasons.append("matched restart marginals are not jointly feasible")
    if not reasons:
      break
  bank_reasons = _v22_bank_invariant_reasons(
    mode=args.mode,
    failure_bank=failure_bank,
    success_pool=success_pool,
    success_bank=success_bank,
    minimum_required=12,
    restart_balance_profile=restart_balance_profile,
  )
  joint_balance_preflight = v19_restart_pair_feasibility(
    failure_bank,
    success_bank,
    12,
    balance_profile=restart_balance_profile,
  )
  if not joint_balance_preflight["passed"]:
    bank_reasons.append("matched restart marginals are not jointly feasible")
  (output_dir / "bank_discovery.json").write_text(
    json.dumps(discovery, indent=2, sort_keys=True) + "\n"
  )
  if bank_reasons:
    raise RuntimeError(f"v22 specialist bank invariant failed: {bank_reasons}")

  round_zero_path = output_dir / "post_round_000.pt"
  _save_checkpoint(
    runner,
    round_zero_path,
    iteration=0,
    metadata={
      "protocol_id": PROTOCOL_ID,
      "context_id": args.context_id,
      "round": 0,
      "actor_sha256": initial_actor_sha256,
      "validation_monitor_is_candidate_gate": False,
    },
    hard_case_bank=failure_bank,
    hard_case_generator=specialist_generator,
    specialist_success_pool=success_pool,
    specialist_success_bank=success_bank,
  )
  best_path = output_dir / "best_so_far.pt"
  shutil.copy2(round_zero_path, best_path)
  best_row = dict(validation_rows[0])

  rounds: list[dict[str, Any]] = []
  last_d0_safe_state = runner.snapshot_candidate_state()
  last_d0_safe_eval = baseline_eval["D0"]
  last_d0_safe_round = 0
  accepted_update_count = 0
  for round_index in range(1, args.maximum_rounds + 1):
    before = runner.snapshot_candidate_state()
    old_actor_state = _actor_state(runner.alg.actor)
    rollout_batches = []
    rollout_metrics = []
    rollout_seeds = []
    for batch_index in range(DUAL_ROLLOUT_BATCHES):
      runner.restore_candidate_state(before)
      rollout_seed = dual_rollout_seed(
        args.context_id, round_index, batch_index
      )
      rollout_seeds.append(rollout_seed)
      _seed_rollout(rollout_seed, specialist_generator)
      obs, metrics, batch = _collect_and_update_specialist(
        runner,
        obs,
        critic_only=False,
        specialist_mode=args.mode,
        failure_bank=failure_bank,
        success_pool=success_pool,
        success_bank=success_bank,
        failure_fraction=args.failure_start_fraction,
        success_fraction=args.success_start_fraction,
        specialist_generator=specialist_generator,
        minimum_riser=1,
        protocol_version=19,
        defer_update=True,
        restart_balance_profile=restart_balance_profile,
      )
      metrics["rollout_seed"] = rollout_seed
      metrics["dual_batch_index"] = batch_index + 1
      if (
        metrics["failure_start_count"] != required_failure_starts
        or metrics["success_start_count"] != required_success_starts
      ):
        raise RuntimeError("v22 rollout did not realize the 40/12/12 allocation")
      rollout_batches.append(batch)
      rollout_metrics.append(metrics)
    runner.restore_candidate_state(before)
    update_metrics = runner.alg.update_dual_rollouts(tuple(rollout_batches))
    update_metrics["rollout_seeds"] = rollout_seeds
    update_metrics["collector_metrics"] = rollout_metrics
    full_candidate_state = _actor_state(runner.alg.actor)

    screening_seed = candidate_screen_seed(args.context_id, round_index)
    old_screen = _evaluate_state(
      runner,
      old_actor_state,
      domains=("DQHMED",),
      num_envs=args.candidate_screen_episodes,
      num_episodes=args.candidate_screen_episodes,
      seed=screening_seed,
      device=args.gate_device,
      runtime_filter=True,
      deployment_context=context_path,
      v19_context=context_path,
    )["DQHMED"]
    variants: list[dict[str, Any]] = []
    for fraction in args.candidate_fractions:
      state = backtrack_actor_state(old_actor_state, full_candidate_state, fraction)
      candidate_metrics = _policy_step_metrics(runner, state, update_metrics)
      finite = _finite_actor_state(state) and math.isfinite(
        float(candidate_metrics["mean_kl"])
      )
      candidate_screen = _evaluate_state(
        runner,
        state,
        domains=("DQHMED",),
        num_envs=args.candidate_screen_episodes,
        num_episodes=args.candidate_screen_episodes,
        seed=screening_seed,
        device=args.gate_device,
        runtime_filter=True,
        deployment_context=context_path,
        v19_context=context_path,
      )["DQHMED"]
      variants.append(
        {
          "fraction": float(fraction),
          "state": state,
          "update_metrics": candidate_metrics,
          "screen_eval": candidate_screen,
          "screen_success_delta": float(
            candidate_screen["success_rate"] - old_screen["success_rate"]
          ),
          "screen_fall_delta": float(
            candidate_screen["fall_rate"] - old_screen["fall_rate"]
          ),
          "screen_eligible": finite,
        }
      )
    eligible = [variant for variant in variants if variant["screen_eligible"]]
    screened_best = (
      max(
        eligible,
        key=lambda variant: (
          variant["screen_success_delta"],
          -variant["screen_fall_delta"],
          -variant["fraction"],
        ),
      )
      if eligible
      else None
    )

    confirmation_seed = candidate_confirmation_seed(
      args.context_id, round_index
    )
    target_gate_accepted = False
    selected_fraction = None
    if screened_best is not None:
      old_confirm = _evaluate_state(
        runner,
        old_actor_state,
        domains=("DQHMED",),
        num_envs=args.candidate_confirm_episodes,
        num_episodes=args.candidate_confirm_episodes,
        seed=confirmation_seed,
        device=args.gate_device,
        runtime_filter=True,
        deployment_context=context_path,
        v19_context=context_path,
      )["DQHMED"]
      candidate_confirm = _evaluate_state(
        runner,
        screened_best["state"],
        domains=("DQHMED",),
        num_envs=args.candidate_confirm_episodes,
        num_episodes=args.candidate_confirm_episodes,
        seed=confirmation_seed,
        device=args.gate_device,
        runtime_filter=True,
        deployment_context=context_path,
        v19_context=context_path,
      )["DQHMED"]
      if old_confirm["initial_state_signatures"] != candidate_confirm[
        "initial_state_signatures"
      ]:
        raise RuntimeError("v22 confirmation conditions are not strictly paired")
      confirm_success_delta = float(
        candidate_confirm["success_rate"] - old_confirm["success_rate"]
      )
      confirm_fall_delta = float(
        candidate_confirm["fall_rate"] - old_confirm["fall_rate"]
      )
      target_gate_accepted, confirm_reasons = candidate_confirmation_gate(
        success_delta=confirm_success_delta,
        fall_delta=confirm_fall_delta,
        finite=_finite_actor_state(screened_best["state"]),
      )
      confirmation = {
        "seed": confirmation_seed,
        "paired_episodes": args.candidate_confirm_episodes,
        "old": old_confirm,
        "candidate": candidate_confirm,
        "success_delta": confirm_success_delta,
        "fall_delta": confirm_fall_delta,
        "accepted": target_gate_accepted,
        "reasons": confirm_reasons,
        "confidence_interval_or_block_gate_used": False,
      }
    else:
      confirmation = {
        "seed": confirmation_seed,
        "paired_episodes": args.candidate_confirm_episodes,
        "accepted": False,
        "reasons": ["no finite screening candidate"],
        "confidence_interval_or_block_gate_used": False,
      }

    if target_gate_accepted:
      assert screened_best is not None
      runner.alg.actor.load_state_dict(screened_best["state"], strict=True)
      runner.alg.reset_online_optimizer()
      selected_fraction = float(screened_best["fraction"])
    else:
      runner.restore_candidate_state(before)
      runner.reduce_after_rejection()

    d0_eval = _evaluate_state(
      runner,
      _actor_state(runner.alg.actor),
      domains=("D0",),
      num_envs=args.d0_check_num_episodes,
      num_episodes=args.d0_check_num_episodes,
      seed=d0_seed,
      device=args.gate_device,
      runtime_filter=True,
      v19_context=context_path,
    )["D0"]
    d0_passed, d0_reasons = specialist_d0_retention_gate(
      baseline_eval=baseline_eval["D0"], candidate_eval=d0_eval
    )
    d0_rollback = False
    if d0_passed:
      last_d0_safe_eval = d0_eval
      if target_gate_accepted:
        accepted_update_count += 1
        last_d0_safe_round = round_index
      last_d0_safe_state = runner.snapshot_candidate_state()
    else:
      runner.restore_candidate_state(last_d0_safe_state)
      runner.reduce_after_rejection()
      d0_rollback = True
    policy_changed = target_gate_accepted and not d0_rollback
    round_actor_state = _actor_state(runner.alg.actor)
    round_actor_sha256 = _actor_state_sha256(round_actor_state)

    validation_record = None
    previous_best_round = int(best_row["round"])
    if policy_changed:
      root = output_dir / "validation_monitor" / "raw" / f"round_{round_index:03d}"
      validation = _evaluate_state(
        runner,
        round_actor_state,
        domains=("DQHMED",),
        num_envs=EVAL_BATCH_SIZE,
        num_episodes=EVAL_BATCH_SIZE,
        seed=validation_seed,
        repeats=VALIDATION_EPISODES // EVAL_BATCH_SIZE,
        device=args.gate_device,
        runtime_filter=True,
        artifact_dir=root,
        resume=True,
        deployment_context=context_path,
        v19_context=context_path,
      )["DQHMED"]
      if validation["initial_state_signatures"] != baseline_validation[
        "initial_state_signatures"
      ]:
        raise RuntimeError("v22 validation monitor conditions changed")
      validation_record = {
        "context_id": args.context_id,
        "round": round_index,
        "actor_sha256": round_actor_sha256,
        "accepted_checkpoint": True,
        "d0_safe": True,
        "success_rate": float(validation["success_rate"]),
        "fall_rate": float(validation["fall_rate"]),
        "success_delta_from_pi0": float(
          validation["success_rate"] - baseline_validation["success_rate"]
        ),
        "fall_delta_from_pi0": float(
          validation["fall_rate"] - baseline_validation["fall_rate"]
        ),
        "fall_constraint_passed": float(validation["fall_rate"])
        <= float(baseline_validation["fall_rate"]) + 0.02,
        "best_so_far_after_evaluation": False,
      }
      validation_rows.append(validation_record)
      selected = select_best_so_far(validation_rows)
      best_row = selected
      for row in validation_rows:
        row["best_so_far_after_evaluation"] = int(row["round"]) == int(
          best_row["round"]
        )

    serializable_variants = [
      {key: value for key, value in variant.items() if key != "state"}
      for variant in variants
    ]
    record = {
      "round": round_index,
      "specialist_mode": args.mode,
      "dual_rollout_seeds": rollout_seeds,
      "full_update_metrics": update_metrics,
      "candidate_screening": {
        "seed": screening_seed,
        "episodes_per_candidate": args.candidate_screen_episodes,
        "old": old_screen,
        "variants": serializable_variants,
        "best_fraction": None if screened_best is None else screened_best["fraction"],
        "selection_uses_point_estimate_only": True,
      },
      "candidate_confirmation": confirmation,
      "target_gate_accepted": target_gate_accepted,
      "candidate_accepted_after_target_and_d0_gates": policy_changed,
      "selected_candidate_fraction": selected_fraction,
      "d0_check": {
        "passed": d0_passed,
        "reasons": d0_reasons,
        "baseline": baseline_eval["D0"],
        "candidate": d0_eval,
        "accepted_round_end_actor": last_d0_safe_eval,
      },
      "d0_rollback": d0_rollback,
      "rolled_back_to_d0_safe_round": last_d0_safe_round if d0_rollback else None,
      "policy_changed_at_round_end": policy_changed,
      "accepted_update_count": accepted_update_count,
      "round_end_actor_sha256": round_actor_sha256,
      "validation_monitor_evaluated": validation_record is not None,
      "validation_monitor_result": validation_record,
      "validation_monitor_used_for_candidate_or_training": False,
      "best_so_far_round_after_round": int(best_row["round"]),
    }
    rounds.append(record)
    checkpoint_path = output_dir / f"post_round_{round_index:03d}.pt"
    _save_checkpoint(
      runner,
      checkpoint_path,
      iteration=round_index,
      metadata=record,
      hard_case_bank=failure_bank,
      hard_case_generator=specialist_generator,
      specialist_success_pool=success_pool,
      specialist_success_bank=success_bank,
    )
    if int(best_row["round"]) == round_index and previous_best_round != round_index:
      shutil.copy2(checkpoint_path, best_path)
    (output_dir / "online_rounds.json").write_text(
      json.dumps(rounds, indent=2, sort_keys=True) + "\n"
    )
    validation_dir = output_dir / "validation_monitor"
    validation_dir.mkdir(parents=True, exist_ok=True)
    _write_validation_rows(validation_dir / "validation_curve.csv", validation_rows)
    (validation_dir / "validation_curve.json").write_text(
      json.dumps(
        {
          "schema_version": 1,
          "protocol_id": PROTOCOL_ID,
          "context_id": args.context_id,
          "seed": validation_seed,
          "paired_conditions": VALIDATION_EPISODES,
          "same_conditions_for_every_evaluated_checkpoint": True,
          "candidate_gate_or_ppo_input": False,
          "evaluated_checkpoint_rounds": [row["round"] for row in validation_rows],
          "best_so_far": best_row,
          "rows": validation_rows,
        },
        indent=2,
        sort_keys=True,
      )
      + "\n"
    )

  final_actor_state = _actor_state(runner.alg.actor)
  initial_first_layer = initial_actor_state[first_layer_key]
  final_first_layer = final_actor_state[first_layer_key]
  legacy_change = float(
    (
      final_first_layer[:, :legacy_width]
      - initial_first_layer[:, :legacy_width]
    )
    .abs()
    .max()
  )
  new_column_max_abs = float(final_first_layer[:, legacy_width:].abs().max())
  final_joint = v19_restart_pair_feasibility(
    failure_bank,
    success_bank,
    12,
    balance_profile=restart_balance_profile,
  )
  if not final_joint["passed"]:
    raise RuntimeError("v22 final replay bank lost joint marginal feasibility")
  if len(rounds) != ROUNDS:
    raise RuntimeError("v22 did not complete its fixed eight-round budget")

  result = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "method": POLICY_METHOD,
    "context_id": args.context_id,
    "specialist_mode": args.mode,
    "restart_balance_profile": restart_balance_profile,
    "target_failure_type": V19_SPECIALIST_FAILURE_TYPES[args.mode],
    "seed": args.seed,
    "task": task,
    "runtime_cbf": True,
    "raw_policy_action_for_ppo": True,
    "executed_action": "runtime_cbf_safe_action",
    "matched_success_preservation_beta": 0.0,
    "control_or_parallel_comparison_branch": False,
    "frozen_protocol": frozen_protocol,
    "source_file_sha256": source_file_sha256,
    "base_policy_checkpoint": str(checkpoint),
    "base_policy_checkpoint_sha256": _file_sha256(checkpoint),
    "initial_actor_sha256": initial_actor_sha256,
    "last_round_actor_sha256": _actor_state_sha256(final_actor_state),
    "deployment_context_path": str(context_path),
    "deployment_context_file_sha256": _file_sha256(context_path),
    "deployment_context": context_metadata,
    "actor_observation_expansion": {
      "legacy_width": warm_start["source_actor_width"],
      "expanded_width": warm_start["expanded_actor_width"],
      "new_feature_count": 5,
      "pre_adaptation_policy_exactly_preserved": True,
      "legacy_input_column_change_max_abs": legacy_change,
      "legacy_input_columns_frozen": legacy_change == 0.0,
      "new_input_column_max_abs": new_column_max_abs,
    },
    "learning_core": "v20/v21 beta-zero matched-success PPO core",
    "single_actor": True,
    "single_privileged_critic": True,
    "auxiliary_risk_or_cost_heads": False,
    "rollout_per_round": {
      "batch_count": DUAL_ROLLOUT_BATCHES,
      "environments_per_batch": args.num_envs,
      "steps_per_environment": args.rollout_steps,
      "normal_failure_success_slots": list(NORMAL_FAILURE_SUCCESS_SLOTS),
    },
    "candidate_selection": {
      "fractions": list(args.candidate_fractions),
      "screening_paired_episodes_per_candidate": args.candidate_screen_episodes,
      "single_fresh_confirmation_paired_episodes": args.candidate_confirm_episodes,
      "confidence_interval_or_block_gate_used": False,
      "confirmation_success_delta_strictly_positive": True,
      "confirmation_maximum_fall_delta": 0.03,
      "d0_minimum_success_delta": -0.05,
    },
    "round_protocol": {
      "fixed_round_budget": ROUNDS,
      "actual_rounds": len(rounds),
      "accepted_updates": accepted_update_count,
      "early_termination_enabled": False,
    },
    "validation_monitor": {
      "seed": validation_seed,
      "paired_conditions": VALIDATION_EPISODES,
      "excluded_from_ppo_replay_and_candidate_gates": True,
      "same_conditions_for_all_accepted_checkpoints": True,
      "maximum_fall_increase_from_pi0": 0.02,
      "selection_primary": "highest target success rate",
      "tie_breaks": ["lower fall rate", "earlier round"],
      "rows": validation_rows,
      "best_so_far": best_row,
      "best_checkpoint": str(best_path),
      "best_checkpoint_sha256": _file_sha256(best_path),
    },
    "structural_checks": structural_checks,
    "warm_start": warm_start,
    "bank_discovery": discovery,
    "failure_bank": failure_bank.audit_metadata(),
    "success_pool": success_pool.audit_metadata(),
    "success_counterexample_bank": success_bank.audit_metadata(),
    "bank_discovery_joint_balance_preflight": joint_balance_preflight,
    "bank_joint_balance_preflight": final_joint,
    "baseline_candidate_gate_eval": baseline_eval,
    "rounds": rounds,
    "last_d0_safe_round": last_d0_safe_round,
    "final_test_accessed": False,
    "final_test_checkpoint": "best_so_far, not the last accepted checkpoint",
  }
  _save_checkpoint(
    runner,
    output_dir / "last_round.pt",
    iteration=len(rounds),
    metadata=result,
    hard_case_bank=failure_bank,
    hard_case_generator=specialist_generator,
    specialist_success_pool=success_pool,
    specialist_success_bank=success_bank,
  )
  (output_dir / "specialist_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  env.close()


if __name__ == "__main__":
  main()

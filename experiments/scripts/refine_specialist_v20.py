"""Independent fixed-eight-round v20 adaptation with the v19-R4 PPO core."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
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
from specialist_v20_protocol import (
  FORMAL_ADAPTATION_SEEDS,
  FORMAL_ROUNDS,
  POLICY_METHOD,
  PROTOCOL_ID,
  SPECIALIST_MODES,
  fixed_budget_status,
)
from specialist_v20_tables import write_training_tables

V20_ACTOR_LAYER_MULTIPLIERS = (0.10, 0.25, 0.50, 1.0)
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
  "experiments/scripts/refine_specialist_v20.py",
  "experiments/scripts/specialist_v20_protocol.py",
  "experiments/scripts/specialist_v20_tables.py",
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--deployment-context", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path)
  parser.add_argument("--protocol-commit")
  parser.add_argument("--mode", choices=SPECIALIST_MODES, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--rollout-steps", type=int, default=1024)
  parser.add_argument("--maximum-rounds", type=int, default=FORMAL_ROUNDS)
  parser.add_argument("--candidate-screen-episodes", type=int, default=64)
  parser.add_argument("--candidate-confirm-episodes", type=int, default=128)
  parser.add_argument("--candidate-fractions", nargs="+", type=float, default=(0.5, 1.0, 1.5))
  parser.add_argument("--d0-check-num-episodes", type=int, default=128)
  parser.add_argument("--final-eval-num-episodes", type=int, default=128)
  parser.add_argument("--failure-start-fraction", type=float, default=0.1875)
  parser.add_argument("--success-start-fraction", type=float, default=0.1875)
  parser.add_argument("--failure-policy-weight", type=float, default=1.0)
  parser.add_argument("--success-policy-weight", type=float, default=1.25)
  parser.add_argument("--bank-capacity", type=int, default=256)
  parser.add_argument("--success-pool-capacity", type=int, default=1024)
  parser.add_argument("--failure-discovery-max-rollouts", type=int, default=12)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
  parser.add_argument("--fall-penalty-weight", type=float, default=-100.0)
  parser.add_argument("--fall-redistribution-horizon", type=int, default=100)
  parser.add_argument("--fall-redistribution-decay", type=float, default=0.97)
  parser.add_argument("--fall-redistribution-amount", type=float, default=2.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--gate-device", default="cuda:0")
  run_kind = parser.add_mutually_exclusive_group()
  run_kind.add_argument("--smoke", action="store_true")
  run_kind.add_argument(
    "--development",
    action="store_true",
    help="enforce formal-sized invariants but prohibit a formal evidence label",
  )
  return parser.parse_args()


def _validate_protocol(args: argparse.Namespace) -> None:
  exact_values = {
    "candidate_fractions": (tuple(args.candidate_fractions), (0.5, 1.0, 1.5)),
    "failure_start_fraction": (args.failure_start_fraction, 0.1875),
    "success_start_fraction": (args.success_start_fraction, 0.1875),
    "failure_policy_weight": (args.failure_policy_weight, 1.0),
    "success_policy_weight": (args.success_policy_weight, 1.25),
  }
  mismatches = {
    key: {"actual": actual, "required": required}
    for key, (actual, required) in exact_values.items()
    if actual != required
  }
  if mismatches:
    raise ValueError(f"v20 replay/candidate protocol mismatch: {mismatches}")
  positive = (
    args.num_envs,
    args.rollout_steps,
    args.maximum_rounds,
    args.candidate_screen_episodes,
    args.candidate_confirm_episodes,
    args.d0_check_num_episodes,
    args.bank_capacity,
    args.success_pool_capacity,
    args.failure_discovery_max_rollouts,
  )
  if min(positive) < 1:
    raise ValueError("v20 counts and capacities must be positive")
  if not args.smoke and (
    args.protocol_file is None or args.protocol_commit is None
  ):
    raise ValueError("formal/development v20 requires its frozen protocol")
  if not args.smoke:
    required = {
      "num_envs": (args.num_envs, 64),
      "rollout_steps": (args.rollout_steps, 1024),
      "maximum_rounds": (args.maximum_rounds, FORMAL_ROUNDS),
      "candidate_screen_episodes": (args.candidate_screen_episodes, 64),
      "candidate_confirm_episodes": (args.candidate_confirm_episodes, 128),
      "d0_check_num_episodes": (args.d0_check_num_episodes, 128),
    }
    size_mismatches = {
      key: {"actual": actual, "required": expected}
      for key, (actual, expected) in required.items()
      if actual != expected
    }
    floats = {
      "actor_learning_rate": (args.actor_learning_rate, 5.0e-6),
      "critic_learning_rate": (args.critic_learning_rate, 1.0e-4),
      "fall_penalty_weight": (args.fall_penalty_weight, -100.0),
      "fall_redistribution_decay": (args.fall_redistribution_decay, 0.97),
      "fall_redistribution_amount": (args.fall_redistribution_amount, 2.0),
    }
    float_mismatches = {
      key: {"actual": actual, "required": expected}
      for key, (actual, expected) in floats.items()
      if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15)
    }
    if size_mismatches or float_mismatches or args.fall_redistribution_horizon != 100:
      raise ValueError(
        "formal v20 configuration mismatch: "
        f"sizes={size_mismatches}, floats={float_mismatches}, "
        f"fall_horizon={args.fall_redistribution_horizon}"
      )


def _finite_actor_state(state: dict[str, torch.Tensor]) -> bool:
  return all(bool(torch.isfinite(value).all()) for value in state.values())


def _validate_algorithm(runner) -> dict[str, Any]:
  alg = runner.alg
  new_feature_groups = [
    group
    for group in alg.optimizer.param_groups
    if group.get("role") == "actor_new_features"
  ]
  first_layer_weight = next(
    module.weight
    for module in alg.actor.mlp
    if isinstance(module, torch.nn.Linear)
  )
  checks = {
    "observable_failure_conditioned_refinement": alg.observable_failure_conditioned_refinement,
    "brief_ppo_refinement": alg.brief_ppo_refinement,
    "failure_focused_refinement": alg.failure_focused_refinement,
    "actor_module_count": int(isinstance(alg.actor, torch.nn.Module)),
    "critic_module_count": int(isinstance(alg.critic, torch.nn.Module)),
    "auxiliary_critics_absent": all(
      module is None for module in (alg.fall_critic, alg.intervention_critic, alg.risk_head)
    ),
    "task_first_constrained": alg.task_first_constrained,
    "actor_learning_rate": alg.actor_learning_rate,
    "critic_learning_rate": alg.critic_learning_rate,
    "actor_layer_multipliers": list(alg.actor_layer_multipliers),
    "actor_new_feature_count": alg.actor_new_feature_count,
    "actor_new_feature_learning_rate_multiplier": (
      alg.actor_new_feature_learning_rate_multiplier
    ),
    "freeze_legacy_actor_input_columns": (
      alg.freeze_legacy_actor_input_columns
    ),
    "actor_new_feature_optimizer_group_count": len(new_feature_groups),
    "actor_new_feature_optimizer_learning_rate": (
      None if not new_feature_groups else new_feature_groups[0]["lr"]
    ),
    "actor_new_feature_optimizer_owns_first_layer_weight": (
      len(new_feature_groups) == 1
      and any(
        parameter is first_layer_weight
        for parameter in new_feature_groups[0]["params"]
      )
    ),
    "ppo_epochs": alg.num_learning_epochs,
    "ppo_minibatches": alg.num_mini_batches,
    "ppo_clip": alg.clip_param,
    "target_kl": alg.desired_kl,
    "target_kl_early_stopping": alg.kl_early_stopping,
    "hard_case_policy_weight": alg.hard_case_policy_weight,
    "success_counterexample_policy_weight": alg.success_counterexample_policy_weight,
    "grouped_advantage_deferred": alg.normalize_advantage_per_mini_batch,
    "actor_obs_dim": alg.actor.obs_dim,
    "critic_obs_dim": alg.critic.obs_dim,
  }
  valid = (
    checks["observable_failure_conditioned_refinement"] is True
    and checks["brief_ppo_refinement"] is True
    and checks["failure_focused_refinement"] is True
    and checks["actor_module_count"] == 1
    and checks["critic_module_count"] == 1
    and checks["auxiliary_critics_absent"]
    and checks["task_first_constrained"] is False
    and math.isclose(checks["actor_learning_rate"], 5.0e-6)
    and math.isclose(checks["critic_learning_rate"], 1.0e-4)
    and checks["actor_layer_multipliers"] == list(V20_ACTOR_LAYER_MULTIPLIERS)
    and checks["actor_new_feature_count"] == 5
    and math.isclose(
      checks["actor_new_feature_learning_rate_multiplier"], 1.0
    )
    and checks["freeze_legacy_actor_input_columns"] is True
    and checks["actor_new_feature_optimizer_group_count"] == 1
    and math.isclose(
      checks["actor_new_feature_optimizer_learning_rate"], 5.0e-6
    )
    and checks["actor_new_feature_optimizer_owns_first_layer_weight"] is True
    and checks["ppo_epochs"] == 1
    and checks["ppo_minibatches"] == 4
    and math.isclose(checks["ppo_clip"], 0.05)
    and math.isclose(checks["target_kl"], 0.003)
    and checks["target_kl_early_stopping"] is True
    and math.isclose(checks["hard_case_policy_weight"], 1.0)
    and math.isclose(checks["success_counterexample_policy_weight"], 1.25)
    and checks["grouped_advantage_deferred"] is True
    and checks["actor_obs_dim"] == 410
    and checks["critic_obs_dim"] == 843
  )
  if not valid:
    raise RuntimeError(f"v19 PPO structural invariant failed: {checks}")
  return checks


def _bank_invariant_reasons(
  *,
  mode: str,
  failure_bank,
  success_pool,
  success_bank,
  minimum_required: int,
  require_full_diversity: bool = True,
) -> list[str]:
  reasons: list[str] = []
  failure = failure_bank.audit_metadata()
  pool = success_pool.audit_metadata()
  success = success_bank.audit_metadata()
  if len(failure_bank) < minimum_required:
    reasons.append("failure precursor bank is too small")
  if len(success_bank) < minimum_required:
    reasons.append("matched success bank is too small")
  if failure["outcome_counts"] != {"failure": len(failure_bank)}:
    reasons.append("failure bank outcome purity failed")
  if pool["outcome_counts"] != {"success": len(success_pool)}:
    reasons.append("success pool outcome purity failed")
  if success["outcome_counts"] != {"success": len(success_bank)}:
    reasons.append("matched success bank outcome purity failed")
  if success["matched_entry_count"] != len(success_bank):
    reasons.append("a replayed success is not matched to a failure")
  matched_failure_indices = {
    entry.matched_failure_index
    for entry in success_bank.entries
    if entry.matched_failure_index is not None
    and 0 <= entry.matched_failure_index < len(failure_bank.entries)
  }
  matched_failures = [
    failure_bank.entries[index] for index in sorted(matched_failure_indices)
  ]
  if len(matched_failures) < minimum_required:
    reasons.append("exactly matched failure subset is too small")
  if require_full_diversity and mode == "lateral":
    if set(failure["centerline_sign_counts"]) != {"-1", "1"}:
      reasons.append("lateral bank lacks both centerline-error signs")
    if set(failure["heading_sign_counts"]) != {"-1", "1"}:
      reasons.append("lateral bank lacks both heading-error signs")
    if not {"early", "mid", "late"}.issubset(failure["riser_stage_counts"]):
      reasons.append("lateral bank lacks early/mid/late riser coverage")
    if set(failure["support_foot_counts"]) != {"0", "1"}:
      reasons.append("lateral bank lacks both support feet")
    if not {"low", "high"}.issubset(failure["error_growth_bin_counts"]):
      reasons.append("lateral bank lacks low/high error-growth coverage")
    if {entry.centerline_sign for entry in matched_failures} != {-1, 1}:
      reasons.append("matched lateral pairs lack both centerline-error signs")
    if {entry.heading_sign for entry in matched_failures} != {-1, 1}:
      reasons.append("matched lateral pairs lack both heading-error signs")
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
  elif require_full_diversity:
    if set(failure["touchdown_foot_counts"]) != {"0", "1"}:
      reasons.append("contact bank lacks left/right touchdown")
    if set(failure["slip_foot_counts"]) != {"0", "1"}:
      reasons.append("contact bank lacks left/right slip")
    if set(failure["contact_timing_counts"]) != {"early", "delayed"}:
      reasons.append("contact bank lacks early/delayed contact")
    if set(failure["support_foot_counts"]) != {"0", "1"}:
      reasons.append("contact bank lacks both support feet")
    if {entry.touchdown_foot for entry in matched_failures} != {0, 1}:
      reasons.append("matched contact pairs lack left/right touchdown")
    if {entry.slip_foot for entry in matched_failures} != {0, 1}:
      reasons.append("matched contact pairs lack left/right slip")
    if {entry.contact_timing for entry in matched_failures} != {
      "early",
      "delayed",
    }:
      reasons.append("matched contact pairs lack early/delayed contact")
    if {entry.support_foot for entry in matched_failures} != {0, 1}:
      reasons.append("matched contact pairs lack both support feet")
  return reasons


def _v19_confirmation_gate(
  *, update_metrics: dict[str, Any], old_eval: dict[str, Any], candidate_eval: dict[str, Any], finite: bool
) -> tuple[bool, list[str], dict[str, float]]:
  reasons: list[str] = []
  if old_eval["initial_state_signatures"] != candidate_eval["initial_state_signatures"]:
    reasons.append("confirmation evaluation is not paired")
  success_delta = float(candidate_eval["success_rate"] - old_eval["success_rate"])
  fall_delta = float(candidate_eval["fall_rate"] - old_eval["fall_rate"])
  kl = float(update_metrics["mean_kl"])
  if not finite:
    reasons.append("candidate parameters are non-finite")
  if not success_delta > 0.0:
    reasons.append("confirmation success delta is not strictly positive")
  if fall_delta > 0.03:
    reasons.append("confirmation fall increase exceeds 3 percentage points")
  if not math.isfinite(kl) or not kl < 0.01:
    reasons.append("candidate KL is not below 0.01")
  return not reasons, reasons, {
    "success_delta": success_delta,
    "fall_delta": fall_delta,
    "mean_kl": kl,
  }


def _seed_rollout(seed: int, generator: torch.Generator) -> None:
  random.seed(seed)
  np.random.seed(seed % (2**32 - 1))
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  generator.manual_seed(seed + 19_003)


def _git_output(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def _validate_frozen_protocol(
  *,
  args: argparse.Namespace,
  repo: Path,
  checkpoint: Path,
  context_path: Path,
  context: dict[str, Any],
) -> dict[str, Any] | None:
  if args.smoke:
    return None
  assert args.protocol_file is not None
  assert args.protocol_commit is not None
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  tracked_clean = not _git_output(
    repo, "status", "--porcelain", "--untracked-files=no"
  )
  if current_commit != args.protocol_commit:
    raise RuntimeError("v20 training HEAD differs from the frozen protocol commit")
  if not tracked_clean:
    raise RuntimeError("formal/development v20 requires a clean tracked worktree")
  try:
    relative = protocol_path.relative_to(repo)
  except ValueError as exc:
    raise RuntimeError("v20 training protocol must be inside the repository") from exc
  frozen_blob = subprocess.run(
    ["git", "show", f"{current_commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  protocol_sha256 = hashlib.sha256(frozen_blob).hexdigest()
  if protocol_sha256 != _file_sha256(protocol_path):
    raise RuntimeError("v20 training protocol differs from its committed blob")
  sealed = protocol.get("sealed_inputs", {})
  declared_context = sealed.get("contexts", {}).get(args.mode, {})
  checks = {
    "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
    "protocol_revision": protocol.get("protocol_revision") == 1,
    "status": protocol.get("status")
    == "prospectively_frozen_before_formal_adaptation",
    "adaptation_seeds": protocol.get("adaptation_seeds")
    == list(FORMAL_ADAPTATION_SEEDS),
    "runtime_seed": args.seed in FORMAL_ADAPTATION_SEEDS,
    "base_checkpoint": sealed.get("base_policy_checkpoint_sha256")
    == _file_sha256(checkpoint),
    "context_file": declared_context.get("file_sha256")
    == _file_sha256(context_path),
    "context_parameters": declared_context.get("parameters_sha256")
    == context["parameters_sha256"],
    "context_calibration_seed": declared_context.get(
      "selected_calibration_seed"
    )
    == context["calibration"]["selected_candidate_seed"],
  }
  failed = [name for name, passed in checks.items() if not passed]
  if failed:
    raise RuntimeError(f"v20 frozen training protocol check failed: {failed}")
  return {
    "path": str(protocol_path),
    "relative_path": str(relative),
    "sha256": protocol_sha256,
    "git_commit": current_commit,
    "tracked_worktree_and_index_clean": tracked_clean,
    "checks": checks,
  }


def main() -> None:
  args = _parse_args()
  _validate_protocol(args)
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
    load_calibrated_v19_context,
    load_frozen_deployment_context,
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
  context = (
    load_frozen_deployment_context(context_path)
    if args.smoke
    else load_calibrated_v19_context(context_path)
  )
  if context["specialist_mode"] != args.mode:
    raise ValueError("v19 frozen context mode differs from requested specialist")
  context_hash = context["parameters_sha256"]
  frozen_protocol = _validate_frozen_protocol(
    args=args,
    repo=repo,
    checkpoint=checkpoint,
    context_path=context_path,
    context=context,
  )
  source_file_sha256 = {
    relative: _file_sha256(repo / relative) for relative in TRAINING_SOURCE_FILES
  }

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
  alg_cfg.actor_layer_multipliers = V20_ACTOR_LAYER_MULTIPLIERS
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
  alg_cfg.clip_param = 0.05
  alg_cfg.desired_kl = 0.003
  alg_cfg.num_learning_epochs = 1
  alg_cfg.num_mini_batches = 4
  alg_cfg.schedule = "fixed"
  alg_cfg.entropy_coef = 0.0
  # RSL-RL then leaves GAE unnormalized; v19 normalizes three replay groups
  # jointly across both batches immediately before its paired PPO update.
  alg_cfg.normalize_advantage_per_mini_batch = True

  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  terminal_fall_penalty = args.fall_penalty_weight * env.unwrapped.step_dt
  if not math.isclose(terminal_fall_penalty, -2.0, abs_tol=1.0e-12):
    raise RuntimeError("v19 terminal scalar fall penalty must equal -2")
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v19 task has no custom online runner")
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
    raise RuntimeError("v19 actor expansion did not preserve π0 with five zero columns")
  structural_checks = _validate_algorithm(runner)

  failure_bank = HardCaseStateBank(
    capacity=args.bank_capacity,
    bank_kind=SPECIALIST_FAILURE_BANK_KIND,
    source_domain="DQHMED",
    context_sha256=context_hash,
    specialist_mode=args.mode,
  )
  success_pool = HardCaseStateBank(
    capacity=args.success_pool_capacity,
    bank_kind=SPECIALIST_SUCCESS_POOL_KIND,
    source_domain="DQHMED",
    context_sha256=context_hash,
    specialist_mode=args.mode,
  )
  success_bank = HardCaseStateBank(
    capacity=args.bank_capacity,
    bank_kind=SPECIALIST_SUCCESS_BANK_KIND,
    source_domain="DQHMED",
    context_sha256=context_hash,
    specialist_mode=args.mode,
  )
  specialist_generator = torch.Generator(device="cpu")
  required_failure_starts = round(args.num_envs * args.failure_start_fraction)
  required_success_starts = round(args.num_envs * args.success_start_fraction)
  if (required_failure_starts, required_success_starts) != (12, 12) and not args.smoke:
    raise RuntimeError("formal v19 must realize exactly 40/12/12 environment slots")
  minimum_riser = 1
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  obs, _ = env.reset()

  d0_seed = args.seed + 300_000
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
  (output_dir / "baseline_eval.json").write_text(
    json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n"
  )

  discovery: list[dict[str, Any]] = []
  for discovery_index in range(args.failure_discovery_max_rollouts):
    before_discovery = runner.snapshot_candidate_state()
    rollout_seed = args.seed + 500_000 + discovery_index
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
      minimum_riser=minimum_riser,
      protocol_version=19,
      defer_update=True,
    )
    runner.restore_candidate_state(before_discovery)
    metrics.update(
      discovery_rollout=discovery_index + 1,
      rollout_seed=rollout_seed,
      parameters_restored_after_discovery=True,
    )
    discovery.append(metrics)
    reasons = _bank_invariant_reasons(
      mode=args.mode,
      failure_bank=failure_bank,
      success_pool=success_pool,
      success_bank=success_bank,
      minimum_required=min(required_failure_starts, required_success_starts),
      require_full_diversity=not args.smoke,
    )
    joint_balance_preflight = v19_restart_pair_feasibility(
      failure_bank,
      success_bank,
      min(required_failure_starts, required_success_starts),
    )
    metrics["joint_balance_preflight"] = joint_balance_preflight
    if not joint_balance_preflight["passed"]:
      reasons.append("matched restart marginals are not jointly feasible")
    if not reasons:
      break
  bank_reasons = _bank_invariant_reasons(
    mode=args.mode,
    failure_bank=failure_bank,
    success_pool=success_pool,
    success_bank=success_bank,
    minimum_required=min(required_failure_starts, required_success_starts),
    require_full_diversity=not args.smoke,
  )
  joint_balance_preflight = v19_restart_pair_feasibility(
    failure_bank,
    success_bank,
    min(required_failure_starts, required_success_starts),
  )
  if not joint_balance_preflight["passed"]:
    bank_reasons.append("matched restart marginals are not jointly feasible")
  (output_dir / "bank_discovery.json").write_text(
    json.dumps(discovery, indent=2, sort_keys=True) + "\n"
  )
  if bank_reasons:
    raise RuntimeError(f"v19 specialist bank invariant failed: {bank_reasons}")

  rounds: list[dict[str, Any]] = []
  last_d0_safe_state = runner.snapshot_candidate_state()
  last_d0_safe_eval = baseline_eval["D0"]
  last_d0_safe_round = 0
  accepted_update_count = 0
  consecutive_rejections = 0
  for round_index in range(1, args.maximum_rounds + 1):
    before = runner.snapshot_candidate_state()
    old_actor_state = _actor_state(runner.alg.actor)
    rollout_batches = []
    rollout_metrics = []
    rollout_seeds = []
    for batch_index in range(2):
      runner.restore_candidate_state(before)
      rollout_seed = args.seed + 1_000_000 + 10 * round_index + batch_index
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
        minimum_riser=minimum_riser,
        protocol_version=19,
        defer_update=True,
      )
      metrics["rollout_seed"] = rollout_seed
      metrics["dual_batch_index"] = batch_index + 1
      if (
        metrics["failure_start_count"] != required_failure_starts
        or metrics["success_start_count"] != required_success_starts
      ):
        raise RuntimeError("v19 rollout did not realize the 40/12/12 replay allocation")
      rollout_batches.append(batch)
      rollout_metrics.append(metrics)
    runner.restore_candidate_state(before)
    update_metrics = runner.alg.update_dual_rollouts(tuple(rollout_batches))
    update_metrics["rollout_seeds"] = rollout_seeds
    update_metrics["collector_metrics"] = rollout_metrics
    if not math.isclose(
      update_metrics["hard_case_transition_fraction"],
      12 / 64,
      rel_tol=0.0,
      abs_tol=1.0e-7,
    ) and not args.smoke:
      raise RuntimeError("v19 failure-transition fraction drifted from 12/64")

    full_candidate_state = _actor_state(runner.alg.actor)
    old_first_layer = old_actor_state[first_layer_key]
    full_first_layer = full_candidate_state[first_layer_key]
    input_adapter_update = {
      "legacy_input_column_change_max_abs": float(
        (
          full_first_layer[:, :legacy_width]
          - old_first_layer[:, :legacy_width]
        )
        .abs()
        .max()
      ),
      "new_input_column_change_max_abs": float(
        (
          full_first_layer[:, legacy_width:]
          - old_first_layer[:, legacy_width:]
        )
        .abs()
        .max()
      ),
      "new_input_column_max_abs": float(
        full_first_layer[:, legacy_width:].abs().max()
      ),
    }
    screening_seed = args.seed + 20_000 * round_index
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
      finite = _finite_actor_state(state)
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
      screen_success_delta = float(
        candidate_screen["success_rate"] - old_screen["success_rate"]
      )
      screen_fall_delta = float(
        candidate_screen["fall_rate"] - old_screen["fall_rate"]
      )
      screen_eligible = (
        finite
        and math.isfinite(float(candidate_metrics["mean_kl"]))
        and float(candidate_metrics["mean_kl"]) < 0.01
      )
      variants.append(
        {
          "fraction": float(fraction),
          "state": state,
          "update_metrics": candidate_metrics,
          "screen_eval": candidate_screen,
          "screen_success_delta": screen_success_delta,
          "screen_fall_delta": screen_fall_delta,
          "screen_eligible": screen_eligible,
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

    confirmation_seed = args.seed + 20_000 * round_index + 10_000
    confirmation: dict[str, Any] = {}
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
      target_gate_accepted, confirm_reasons, confirm_deltas = _v19_confirmation_gate(
        update_metrics=screened_best["update_metrics"],
        old_eval=old_confirm,
        candidate_eval=candidate_confirm,
        finite=_finite_actor_state(screened_best["state"]),
      )
      confirmation = {
        "seed": confirmation_seed,
        "old": old_confirm,
        "candidate": candidate_confirm,
        "accepted": target_gate_accepted,
        "reasons": confirm_reasons,
        "deltas": confirm_deltas,
      }
    else:
      confirmation = {
        "seed": confirmation_seed,
        "accepted": False,
        "reasons": ["no finite screening candidate below the hard KL ceiling"],
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
    consecutive_rejections = 0 if policy_changed else consecutive_rejections + 1
    round_end_actor_state = _actor_state(runner.alg.actor)
    round_end_first_layer = round_end_actor_state[first_layer_key]
    round_end_adapter = {
      "legacy_input_column_change_from_initial_max_abs": float(
        (
          round_end_first_layer[:, :legacy_width]
          - initial_actor_state[first_layer_key][:, :legacy_width]
        )
        .abs()
        .max()
      ),
      "new_input_column_max_abs": float(
        round_end_first_layer[:, legacy_width:].abs().max()
      ),
      "new_input_column_rms": float(
        round_end_first_layer[:, legacy_width:].square().mean().sqrt()
      ),
    }

    serializable_variants = [
      {key: value for key, value in variant.items() if key != "state"}
      for variant in variants
    ]
    record = {
      "round": round_index,
      "specialist_mode": args.mode,
      "dual_rollout_seeds": rollout_seeds,
      "dual_rollout_same_behavior_policy": True,
      "full_update_metrics": update_metrics,
      "input_adapter_update": input_adapter_update,
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
      "selected_candidate_fraction": selected_fraction,
      "retained_candidate_fraction": (
        selected_fraction if policy_changed else None
      ),
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
      "accepted_update_count_is_protocol_gate": False,
      "consecutive_rejections": consecutive_rejections,
      "round_end_actor_sha256": _actor_state_sha256(round_end_actor_state),
      "round_end_adapter": round_end_adapter,
    }
    rounds.append(record)
    _save_checkpoint(
      runner,
      output_dir / f"post_round_{round_index:03d}.pt",
      iteration=round_index,
      metadata=record,
      hard_case_bank=failure_bank,
      hard_case_generator=specialist_generator,
      specialist_success_pool=success_pool,
      specialist_success_bank=success_bank,
    )
    (output_dir / "online_rounds.json").write_text(
      json.dumps(rounds, indent=2, sort_keys=True) + "\n"
    )
  budget_status = fixed_budget_status(
    actual_rounds=len(rounds),
    retained_update_count=accepted_update_count,
    formal=not args.smoke,
  )
  final_actor_state = _actor_state(runner.alg.actor)
  initial_first_layer = initial_actor_state[first_layer_key]
  final_first_layer = final_actor_state[first_layer_key]
  legacy_input_column_change_max_abs = float(
    (final_first_layer[:, :legacy_width] - initial_first_layer[:, :legacy_width])
    .abs()
    .max()
  )
  new_input_column_max_abs = float(final_first_layer[:, legacy_width:].abs().max())
  new_input_column_rms = float(
    final_first_layer[:, legacy_width:].square().mean().sqrt()
  )
  final_eval = _evaluate_state(
    runner,
    final_actor_state,
    domains=("D0", "DQHMED"),
    num_envs=args.final_eval_num_episodes,
    num_episodes=args.final_eval_num_episodes,
    seed=args.seed + 900_000,
    device=args.gate_device,
    runtime_filter=True,
    deployment_context=context_path,
    v19_context=context_path,
  )
  final_joint_balance_preflight = v19_restart_pair_feasibility(
    failure_bank,
    success_bank,
    min(required_failure_starts, required_success_starts),
  )
  if not final_joint_balance_preflight["passed"]:
    raise RuntimeError("v19 final replay bank lost joint marginal feasibility")
  final_path = output_dir / "accepted_final.pt"
  result = {
    "method": POLICY_METHOD,
    "learning_core": "v19 Revision-4 Observable Failure-Conditioned Brief PPO",
    "learning_core_collector_protocol_version": 19,
    "formal_protocol": not args.smoke and not args.development,
    "development_run": args.development,
    "smoke_run": args.smoke,
    "protocol_completed": budget_status.protocol_valid,
    "specialist_mode": args.mode,
    "target_failure_type": V19_SPECIALIST_FAILURE_TYPES[args.mode],
    "seed": args.seed,
    "task": task,
    "runtime_cbf": True,
    "raw_policy_action_for_ppo": True,
    "executed_action": "runtime_cbf_safe_action",
    "independent_training_branch": True,
    "frozen_protocol": frozen_protocol,
    "source_file_sha256": source_file_sha256,
    "base_policy_checkpoint": str(checkpoint),
    "base_policy_checkpoint_sha256": _file_sha256(checkpoint),
    "initial_actor_sha256": initial_actor_sha256,
    "final_actor_sha256": _actor_state_sha256(final_actor_state),
    "deployment_context_path": str(context_path),
    "deployment_context_file_sha256": _file_sha256(context_path),
    "deployment_context": context_metadata,
    "actor_observation_expansion": {
      "legacy_width": warm_start["source_actor_width"],
      "expanded_width": warm_start["expanded_actor_width"],
      "new_feature_count": 5,
      "new_first_layer_column_max_abs_before_adaptation": zero_column_max_abs,
      "legacy_first_layer_input_column_change_max_abs": (
        legacy_input_column_change_max_abs
      ),
      "new_first_layer_column_max_abs_after_adaptation": (
        new_input_column_max_abs
      ),
      "new_first_layer_column_rms_after_adaptation": new_input_column_rms,
      "legacy_input_columns_frozen_during_adaptation": (
        legacy_input_column_change_max_abs == 0.0
      ),
      "new_input_columns_use_full_actor_learning_rate": True,
      "pre_adaptation_policy_exactly_preserved": (
        warm_start["pi0_exact_preservation_proof"] is True
        and zero_column_max_abs == 0.0
      ),
      "features": (
        ["normalized_centerline_error", "sin_heading_error", "cos_heading_error", "centerline_error_rate", "heading_error_rate"]
        if args.mode == "lateral"
        else ["left_contact", "right_contact", "left_slip_speed", "right_slip_speed", "phase_contact_mismatch"]
      ),
    },
    "scalar_reward": (
      "task + terminal fall (-2) + redistributed pre-fall penalty (-2) + "
      "one potential recovery term"
    ),
    "dual_cbf_reward_weight": 0.0,
    "rollout_per_round": {
      "batch_count": 2,
      "environments_per_batch": args.num_envs,
      "steps_per_environment": args.rollout_steps,
      "same_frozen_behavior_policy": True,
      "different_rollout_seeds": True,
      "loss_and_gradient_combination": "arithmetic mean",
      "gradient_cosine_is_diagnostic_only": True,
    },
    "requested_start_mixture": {"normal": 0.625, "failure": 0.1875, "success": 0.1875},
    "integer_start_mixture_for_64_envs": {"normal": 40, "failure": 12, "success": 12},
    "advantage_normalization": "separate normal/failure/success groups across both batches",
    "failure_precursor_policy_weight": args.failure_policy_weight,
    "success_counterexample_policy_weight": args.success_policy_weight,
    "candidate_selection": {
      "fractions": list(args.candidate_fractions),
      "screening_paired_episodes_per_candidate": args.candidate_screen_episodes,
      "screening_candidate_episode_total": 3 * args.candidate_screen_episodes,
      "independent_confirmation_paired_episodes": args.candidate_confirm_episodes,
      "candidate_episode_total": 3 * args.candidate_screen_episodes + args.candidate_confirm_episodes,
      "confirmation_gate": {"success_delta_strictly_above": 0.0, "maximum_fall_increase": 0.03, "maximum_kl_strictly_below": 0.01},
    },
    "round_protocol": {
      "fixed_round_budget": args.maximum_rounds,
      "early_termination_enabled": False,
      "minimum_accepted_updates": None,
      "rejection_patience": None,
      "zero_to_eight_retained_updates_are_valid": True,
      "retained_update_count_is_protocol_gate": False,
      "actual_rounds": len(rounds),
      "accepted_updates": accepted_update_count,
      "stop_reason": budget_status.stop_reason,
    },
    "structural_checks": structural_checks,
    "warm_start": warm_start,
    "bank_discovery": discovery,
    "failure_bank": failure_bank.audit_metadata(),
    "success_pool": success_pool.audit_metadata(),
    "success_counterexample_bank": success_bank.audit_metadata(),
    "bank_discovery_joint_balance_preflight": joint_balance_preflight,
    "bank_joint_balance_preflight": final_joint_balance_preflight,
    "baseline_eval": baseline_eval,
    "rounds": rounds,
    "last_d0_safe_round": last_d0_safe_round,
    "final_eval_training_diagnostic_only": final_eval,
    "final_checkpoint": str(final_path),
  }
  result["training_tables"] = write_training_tables(result, output_dir)
  _save_checkpoint(
    runner,
    final_path,
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

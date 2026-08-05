"""Independent failure-mode-conditioned safe online refinement (v17)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
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


TRAINING_SOURCE_FILES = (
  "experiments/scripts/evaluate_online_stairs.py",
  "experiments/scripts/online_refine_stairs.py",
  "experiments/scripts/refine_specialist_v17.py",
  "src/tasks/stairs_cbf/command.py",
  "src/tasks/stairs_cbf/config.py",
  "src/tasks/stairs_cbf/deployment_context.py",
  "src/tasks/stairs_cbf/hard_cases.py",
  "src/tasks/stairs_cbf/mdp.py",
  "src/tasks/stairs_cbf/online.py",
)


def _finite_actor_state(state: dict[str, torch.Tensor]) -> bool:
  return all(bool(torch.isfinite(value).all()) for value in state.values())


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--deployment-context", type=Path, required=True)
  parser.add_argument("--mode", choices=("lateral", "cbf", "balance"), required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--rollout-steps", type=int, default=1024)
  parser.add_argument("--online-rounds", type=int, default=5)
  parser.add_argument("--candidate-num-episodes", type=int, default=128)
  parser.add_argument("--d0-check-num-episodes", type=int, default=128)
  parser.add_argument("--final-eval-num-episodes", type=int, default=128)
  parser.add_argument(
    "--candidate-fractions", nargs="+", type=float, default=(0.5, 1.0, 1.5)
  )
  parser.add_argument("--failure-start-fraction", type=float, default=0.15)
  parser.add_argument("--success-start-fraction", type=float, default=0.15)
  parser.add_argument("--failure-policy-weight", type=float, default=0.75)
  parser.add_argument("--bank-capacity", type=int, default=256)
  parser.add_argument("--success-pool-capacity", type=int, default=512)
  parser.add_argument("--failure-discovery-max-rollouts", type=int, default=8)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
  parser.add_argument("--fall-penalty-weight", type=float, default=-100.0)
  parser.add_argument("--fall-redistribution-horizon", type=int, default=100)
  parser.add_argument("--fall-redistribution-decay", type=float, default=0.97)
  parser.add_argument("--fall-redistribution-amount", type=float, default=2.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--gate-device", default="cuda:0")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Allow reduced sizes for code-path validation; never formal evidence.",
  )
  return parser.parse_args()


def _validate_protocol(args: argparse.Namespace) -> None:
  if tuple(args.candidate_fractions) != (0.5, 1.0, 1.5):
    raise ValueError("v17 candidate fractions must be exactly 0.5, 1.0, 1.5")
  if not math.isclose(args.failure_start_fraction, 0.15):
    raise ValueError("v17 failure precursor start fraction must be 0.15")
  if not math.isclose(args.success_start_fraction, 0.15):
    raise ValueError("v17 success counterexample start fraction must be 0.15")
  if not math.isclose(args.failure_policy_weight, 0.75):
    raise ValueError("v17 failure-precursor actor weight must be 0.75")
  if args.failure_discovery_max_rollouts < 1:
    raise ValueError("v17 requires a positive bank-discovery limit")
  positive = (
    args.num_envs,
    args.rollout_steps,
    args.online_rounds,
    args.candidate_num_episodes,
    args.d0_check_num_episodes,
    args.final_eval_num_episodes,
    args.bank_capacity,
    args.success_pool_capacity,
  )
  if min(positive) < 1:
    raise ValueError("v17 counts and capacities must be positive")
  if not args.smoke:
    exact = {
      "num_envs": (args.num_envs, 64),
      "rollout_steps": (args.rollout_steps, 1024),
      "online_rounds": (args.online_rounds, 5),
      "candidate_num_episodes": (args.candidate_num_episodes, 128),
      "d0_check_num_episodes": (args.d0_check_num_episodes, 128),
    }
    mismatches = {
      name: {"actual": actual, "required": required}
      for name, (actual, required) in exact.items()
      if actual != required
    }
    if mismatches:
      raise ValueError(f"formal v17 rollout protocol mismatch: {mismatches}")
    floats = {
      "actor_learning_rate": (args.actor_learning_rate, 5.0e-6),
      "critic_learning_rate": (args.critic_learning_rate, 1.0e-4),
      "fall_penalty_weight": (args.fall_penalty_weight, -100.0),
      "fall_redistribution_decay": (args.fall_redistribution_decay, 0.97),
      "fall_redistribution_amount": (args.fall_redistribution_amount, 2.0),
    }
    float_mismatches = {
      name: {"actual": actual, "required": required}
      for name, (actual, required) in floats.items()
      if not math.isclose(actual, required, rel_tol=0.0, abs_tol=1.0e-15)
    }
    if float_mismatches or args.fall_redistribution_horizon != 100:
      raise ValueError(
        "formal v17 scalar-reward protocol mismatch: "
        f"{float_mismatches}, horizon={args.fall_redistribution_horizon}"
      )


def _validate_algorithm(runner) -> dict[str, Any]:
  alg = runner.alg
  checks = {
    "brief_ppo_refinement": alg.brief_ppo_refinement,
    "failure_focused_refinement": alg.failure_focused_refinement,
    "actor_module_count": int(isinstance(alg.actor, torch.nn.Module)),
    "critic_module_count": int(isinstance(alg.critic, torch.nn.Module)),
    "fall_critic_absent": alg.fall_critic is None,
    "intervention_critic_absent": alg.intervention_critic is None,
    "risk_head_absent": alg.risk_head is None,
    "base_actor_reference_absent": alg.base_actor_reference is None,
    "retention_actor_reference_absent": alg.retention_actor_reference is None,
    "retention_bank_count": len(alg.retention_anchor_banks),
    "task_first_constrained": alg.task_first_constrained,
    "actor_learning_rate": alg.actor_learning_rate,
    "critic_learning_rate": alg.critic_learning_rate,
    "actor_layer_multipliers": list(alg.actor_layer_multipliers),
    "ppo_epochs": alg.num_learning_epochs,
    "ppo_clip": alg.clip_param,
    "target_kl": alg.desired_kl,
    "target_kl_early_stopping": alg.kl_early_stopping,
    "hard_case_policy_weight": alg.hard_case_policy_weight,
    "std_scale_from_base": alg.std_scale_from_base,
    "base_anchor_weight": alg.base_anchor_weight,
    "d0_retention_anchor_weight": alg.d0_retention_anchor_weight,
    "neighbor_retention_anchor_weight": alg.neighbor_retention_anchor_weight,
    "pre_intervention_weight": alg.pre_intervention_weight,
    "intervention_advantage_weight": alg.intervention_advantage_weight,
    "safe_bc_weight": alg.safe_bc_weight,
    "correction_distillation_weight": alg.correction_distillation_weight,
    "log_std_learning_rate": alg.log_std_learning_rate,
  }
  zero_fields = (
    "base_anchor_weight",
    "d0_retention_anchor_weight",
    "neighbor_retention_anchor_weight",
    "pre_intervention_weight",
    "intervention_advantage_weight",
    "safe_bc_weight",
    "correction_distillation_weight",
    "log_std_learning_rate",
  )
  valid = (
    checks["brief_ppo_refinement"] is True
    and checks["failure_focused_refinement"] is True
    and checks["actor_module_count"] == 1
    and checks["critic_module_count"] == 1
    and checks["fall_critic_absent"]
    and checks["intervention_critic_absent"]
    and checks["risk_head_absent"]
    and checks["base_actor_reference_absent"]
    and checks["retention_actor_reference_absent"]
    and checks["retention_bank_count"] == 0
    and checks["task_first_constrained"] is False
    and all(checks[name] == 0.0 for name in zero_fields)
    and math.isclose(checks["actor_learning_rate"], 5.0e-6)
    and math.isclose(checks["critic_learning_rate"], 1.0e-4)
    and checks["actor_layer_multipliers"] == [1.0, 1.0, 1.0, 1.0]
    and checks["ppo_epochs"] == 1
    and math.isclose(checks["ppo_clip"], 0.05)
    and math.isclose(checks["target_kl"], 0.003)
    and checks["target_kl_early_stopping"] is True
    and math.isclose(checks["hard_case_policy_weight"], 0.75)
    and math.isclose(checks["std_scale_from_base"], 0.35)
  )
  if not valid:
    raise RuntimeError(f"v17 PPO structural invariant failed: {checks}")
  return checks


def _bank_invariant_reasons(
  *,
  mode: str,
  failure_bank,
  success_pool,
  success_bank,
  minimum_required: int,
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
  if mode == "lateral":
    counts = failure["riser_index_counts"]
    if len(counts) < 2:
      reasons.append("lateral failure bank does not cover multiple late risers")
    if counts and max(counts.values()) == len(failure_bank):
      reasons.append("lateral failure bank collapsed onto one riser")
  if mode == "balance":
    supports = failure["support_foot_counts"]
    if set(supports) != {"0", "1"}:
      reasons.append("balance failure bank does not cover both support feet")
    if len(failure["balance_bucket_counts"]) < 2:
      reasons.append("balance failure bank has no support/phase diversity")
  return reasons


def main() -> None:
  args = _parse_args()
  _validate_protocol(args)
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.deployment_context import (
    SPECIALIST_FAILURE_TYPES,
    apply_frozen_deployment_context,
    load_calibrated_specialist_context,
    load_frozen_deployment_context,
  )
  from src.tasks.stairs_cbf.hard_cases import (
    HardCaseStateBank,
    SPECIALIST_FAILURE_BANK_KIND,
    SPECIALIST_SUCCESS_BANK_KIND,
    SPECIALIST_SUCCESS_POOL_KIND,
  )
  from src.tasks.stairs_cbf.online import (
    SpecialistGateThresholds,
    backtrack_actor_state,
    specialist_candidate_gate,
    specialist_candidate_precheck,
    specialist_d0_retention_gate,
    specialist_target_score,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  context_path = args.deployment_context.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  context = (
    load_frozen_deployment_context(context_path)
    if args.smoke
    else load_calibrated_specialist_context(context_path)
  )
  if context["specialist_mode"] != args.mode:
    raise ValueError("frozen context mode does not match requested specialist")
  context_hash = context["parameters_sha256"]
  source_file_sha256 = {
    relative: _file_sha256(repo / relative) for relative in TRAINING_SOURCE_FILES
  }
  task = "Unitree-G1-Stairs-Online-DQHMED"
  env_cfg = load_env_cfg(task)
  context_metadata = apply_frozen_deployment_context(
    env_cfg, context, role="target"
  )
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env_cfg.actions["joint_pos"].enabled = True
  env_cfg.rewards["cbf_dual"].weight = 0.0
  env_cfg.rewards["fall_termination"].weight = args.fall_penalty_weight
  signal_weight = float(env_cfg.rewards["specialist_failure_signal"].weight)
  if signal_weight >= 0.0:
    raise RuntimeError("specialist scalar failure signal must be a penalty")

  agent_cfg = load_rl_cfg(task)
  agent_cfg.seed = args.seed
  agent_cfg.num_steps_per_env = args.rollout_steps
  alg_cfg = agent_cfg.algorithm
  alg_cfg.actor_learning_rate = args.actor_learning_rate
  alg_cfg.critic_learning_rate = args.critic_learning_rate
  alg_cfg.actor_layer_multipliers = (1.0, 1.0, 1.0, 1.0)
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
  # This switch activates scalar fall-credit redistribution, not an extra head.
  alg_cfg.failure_focused_refinement = True
  alg_cfg.kl_early_stopping = True
  alg_cfg.fall_redistribution_horizon = args.fall_redistribution_horizon
  alg_cfg.fall_redistribution_decay = args.fall_redistribution_decay
  alg_cfg.fall_redistribution_amount = args.fall_redistribution_amount
  alg_cfg.hard_case_policy_weight = args.failure_policy_weight
  alg_cfg.clip_param = 0.05
  alg_cfg.desired_kl = 0.003
  alg_cfg.num_learning_epochs = 1
  alg_cfg.schedule = "fixed"
  alg_cfg.entropy_coef = 0.0
  alg_cfg.normalize_advantage_per_mini_batch = False

  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  terminal_fall_penalty = args.fall_penalty_weight * env.unwrapped.step_dt
  if not math.isclose(terminal_fall_penalty, -2.0, abs_tol=1.0e-12):
    raise RuntimeError("v17 terminal scalar fall penalty must equal -2")
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("specialist task has no custom online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  initial_actor_state = _actor_state(runner.alg.actor)
  initial_actor_sha256 = _actor_state_sha256(initial_actor_state)
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
  specialist_generator.manual_seed(args.seed + 170003)
  required_failure_starts = int(round(args.num_envs * 0.15))
  required_success_starts = int(round(args.num_envs * 0.15))
  minimum_riser = math.ceil(int(context["target"]["num_steps"]) / 2)
  thresholds = SpecialistGateThresholds()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  obs, _ = env.reset()

  d0_seed = args.seed + 200000
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
  )
  (output_dir / "baseline_eval.json").write_text(
    json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n"
  )

  discovery: list[dict[str, Any]] = []
  for discovery_index in range(args.failure_discovery_max_rollouts):
    before_discovery = runner.snapshot_candidate_state()
    obs, metrics = _collect_and_update_specialist(
      runner,
      obs,
      critic_only=True,
      specialist_mode=args.mode,
      failure_bank=failure_bank,
      success_pool=success_pool,
      success_bank=success_bank,
      failure_fraction=0.0,
      success_fraction=0.0,
      specialist_generator=specialist_generator,
      minimum_riser=minimum_riser,
    )
    runner.restore_candidate_state(before_discovery)
    metrics["discovery_rollout"] = discovery_index + 1
    metrics["parameters_restored_after_discovery"] = True
    discovery.append(metrics)
    reasons = _bank_invariant_reasons(
      mode=args.mode,
      failure_bank=failure_bank,
      success_pool=success_pool,
      success_bank=success_bank,
      minimum_required=min(required_failure_starts, required_success_starts),
    )
    if not reasons:
      break
  bank_reasons = _bank_invariant_reasons(
    mode=args.mode,
    failure_bank=failure_bank,
    success_pool=success_pool,
    success_bank=success_bank,
    minimum_required=min(required_failure_starts, required_success_starts),
  )
  if bank_reasons:
    raise RuntimeError(f"v17 specialist bank invariant failed: {bank_reasons}")
  (output_dir / "bank_discovery.json").write_text(
    json.dumps(discovery, indent=2, sort_keys=True) + "\n"
  )

  rounds: list[dict[str, Any]] = []
  last_d0_safe_state = runner.snapshot_candidate_state()
  last_d0_safe_round = 0
  for round_index in range(1, args.online_rounds + 1):
    before = runner.snapshot_candidate_state()
    old_actor_state = _actor_state(runner.alg.actor)
    obs, update_metrics = _collect_and_update_specialist(
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
    )
    if (
      update_metrics["failure_start_count"] != required_failure_starts
      or update_metrics["success_start_count"] != required_success_starts
    ):
      raise RuntimeError("v17 rollout did not realize both replay start quotas")
    realized_failure = required_failure_starts / args.num_envs
    if not math.isclose(
      update_metrics["hard_case_transition_fraction"],
      realized_failure,
      rel_tol=0.0,
      abs_tol=1.0e-7,
    ):
      raise RuntimeError("failure-precursor transition fraction drifted")

    full_candidate_state = _actor_state(runner.alg.actor)
    target_seed = args.seed + 1000 * round_index
    old_eval = _evaluate_state(
      runner,
      old_actor_state,
      domains=("DQHMED",),
      num_envs=args.candidate_num_episodes,
      num_episodes=args.candidate_num_episodes,
      seed=target_seed,
      device=args.gate_device,
      runtime_filter=True,
      deployment_context=context_path,
    )["DQHMED"]
    variants: list[dict[str, Any]] = []
    for fraction in args.candidate_fractions:
      state = backtrack_actor_state(old_actor_state, full_candidate_state, fraction)
      candidate_metrics = _policy_step_metrics(runner, state, update_metrics)
      finite = runner.parameters_are_finite() and _finite_actor_state(state)
      reasons = specialist_candidate_precheck(
        update_metrics=candidate_metrics,
        parameters_finite=finite,
        thresholds=thresholds,
      )
      candidate_eval: dict[str, Any] = {}
      scores = {
        "old": specialist_target_score(old_eval),
        "candidate": {"total": float("nan")},
      }
      accepted = False
      if not reasons:
        candidate_eval = _evaluate_state(
          runner,
          state,
          domains=("DQHMED",),
          num_envs=args.candidate_num_episodes,
          num_episodes=args.candidate_num_episodes,
          seed=target_seed,
          device=args.gate_device,
          runtime_filter=True,
          deployment_context=context_path,
        )["DQHMED"]
        accepted, reasons, scores = specialist_candidate_gate(
          update_metrics=candidate_metrics,
          old_eval=old_eval,
          candidate_eval=candidate_eval,
          parameters_finite=finite,
          thresholds=thresholds,
        )
      variants.append(
        {
          "fraction": fraction,
          "state": state,
          "update_metrics": candidate_metrics,
          "candidate_eval": candidate_eval,
          "gate_passed": accepted,
          "gate_reasons": reasons,
          "scores": scores,
        }
      )
    passing = [variant for variant in variants if variant["gate_passed"]]
    selected = (
      max(
        passing,
        key=lambda variant: (
          variant["scores"]["candidate"]["total"],
          -variant["fraction"],
        ),
      )
      if passing
      else None
    )
    target_gate_accepted = selected is not None
    if selected is None:
      runner.restore_candidate_state(before)
      runner.reduce_after_rejection()
      selected_fraction = None
      selected_score = None
    else:
      runner.alg.actor.load_state_dict(selected["state"], strict=True)
      runner.alg.reset_online_optimizer()
      selected_fraction = float(selected["fraction"])
      selected_score = float(selected["scores"]["candidate"]["total"])

    d0_check: dict[str, Any]
    d0_rollback = False
    d0_eval = _evaluate_state(
      runner,
      _actor_state(runner.alg.actor),
      domains=("D0",),
      num_envs=args.d0_check_num_episodes,
      num_episodes=args.d0_check_num_episodes,
      seed=d0_seed,
      device=args.gate_device,
      runtime_filter=True,
    )["D0"]
    d0_passed, d0_reasons = specialist_d0_retention_gate(
      baseline_eval=baseline_eval["D0"],
      candidate_eval=d0_eval,
      thresholds=thresholds,
    )
    d0_check = {
      "passed": d0_passed,
      "reasons": d0_reasons,
      "baseline": baseline_eval["D0"],
      "candidate": d0_eval,
    }
    if d0_passed:
      last_d0_safe_state = runner.snapshot_candidate_state()
      last_d0_safe_round = round_index
    else:
      runner.restore_candidate_state(last_d0_safe_state)
      runner.reduce_after_rejection()
      d0_rollback = True

    serializable_variants = [
      {key: value for key, value in variant.items() if key != "state"}
      for variant in variants
    ]
    record = {
      "round": round_index,
      "specialist_mode": args.mode,
      "scalar_scenario_reward_weight": signal_weight,
      "dual_cbf_reward_weight": 0.0,
      "fall_reward_weight": args.fall_penalty_weight,
      "target_gate_accepted": target_gate_accepted,
      "selected_candidate_fraction": selected_fraction,
      "selected_target_score": selected_score,
      "old_target_eval": old_eval,
      "full_update_metrics": update_metrics,
      "candidate_variants": serializable_variants,
      "d0_check": d0_check,
      "d0_rollback": d0_rollback,
      "rolled_back_to_d0_safe_round": last_d0_safe_round if d0_rollback else None,
      "policy_changed_at_round_end": target_gate_accepted and not d0_rollback,
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

  final_actor_state = _actor_state(runner.alg.actor)
  final_eval = _evaluate_state(
    runner,
    final_actor_state,
    domains=("D0", "DQHMED"),
    num_envs=args.final_eval_num_episodes,
    num_episodes=args.final_eval_num_episodes,
    seed=args.seed + 900000,
    device=args.gate_device,
    runtime_filter=True,
    deployment_context=context_path,
  )
  final_path = output_dir / "accepted_final.pt"
  result = {
    "method": "Failure-Mode-Conditioned Brief PPO v17",
    "formal_protocol": not args.smoke,
    "specialist_mode": args.mode,
    "target_failure_type": SPECIALIST_FAILURE_TYPES[args.mode],
    "seed": args.seed,
    "task": task,
    "runtime_cbf": True,
    "raw_policy_action_for_ppo": True,
    "executed_action": "runtime_cbf_safe_action",
    "independent_training_branch": True,
    "source_file_sha256": source_file_sha256,
    "base_policy_checkpoint": str(checkpoint),
    "base_policy_checkpoint_sha256": _file_sha256(checkpoint),
    "initial_actor_sha256": initial_actor_sha256,
    "final_actor_sha256": _actor_state_sha256(final_actor_state),
    "deployment_context_path": str(context_path),
    "deployment_context_file_sha256": _file_sha256(context_path),
    "deployment_context": context_metadata,
    "scalar_reward": (
      "task + terminal fall (-2) + redistributed pre-fall penalty (-2) "
      "+ one negative mode-specific failure signal"
    ),
    "scalar_scenario_reward_weight": signal_weight,
    "dual_cbf_reward_weight": 0.0,
    "fall_redistribution": {
      "enabled": True,
      "terminal_penalty": terminal_fall_penalty,
      "redistributed_penalty": -args.fall_redistribution_amount,
      "horizon": args.fall_redistribution_horizon,
      "decay": args.fall_redistribution_decay,
    },
    "requested_start_mixture": {"normal": 0.70, "failure": 0.15, "success": 0.15},
    "integer_start_mixture_for_64_envs": {"normal": 44, "failure": 10, "success": 10},
    "realized_start_fractions_for_64_envs": {
      "normal": 44 / 64,
      "failure": 10 / 64,
      "success": 10 / 64,
    },
    "failure_precursor_policy_weight": args.failure_policy_weight,
    "success_counterexample_policy_weight": 1.0,
    "candidate_fractions": list(args.candidate_fractions),
    "candidate_paired_episodes": args.candidate_num_episodes,
    "target_training_score": "success_rate - fall_rate",
    "target_fall_tolerance": thresholds.maximum_target_fall_increase,
    "training_maximum_kl": thresholds.maximum_kl,
    "d0_check_period_rounds": 1,
    "d0_success_tolerance": thresholds.d0_success_tolerance,
    "other_specialist_training_gates": False,
    "neighbor_training_gate": False,
    "cbf_demand_training_gate": False,
    "structural_checks": structural_checks,
    "warm_start": warm_start,
    "bank_discovery": discovery,
    "failure_bank": failure_bank.audit_metadata(),
    "success_pool": success_pool.audit_metadata(),
    "success_counterexample_bank": success_bank.audit_metadata(),
    "baseline_eval": baseline_eval,
    "rounds": rounds,
    "last_d0_safe_round": last_d0_safe_round,
    "final_eval_training_diagnostic_only": final_eval,
    "final_checkpoint": str(final_path),
  }
  _save_checkpoint(
    runner,
    final_path,
    iteration=args.online_rounds,
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

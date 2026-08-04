"""CBF-Guided Brief PPO refinement on one fixed deployment domain.

This entrypoint intentionally does not expose the v12/v13 constrained heads,
policy anchors, retention banks, correction distillation, neighboring-domain
training gate, or per-round confidence intervals.  Those older experiments
remain reproducible through ``online_refine_stairs.py``.
"""

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
  _collect_and_update,
  _evaluate_state,
  _policy_step_metrics,
  _save_checkpoint,
)


def _finite_actor_state(state: dict[str, torch.Tensor]) -> bool:
  return all(bool(torch.isfinite(value).all()) for value in state.values())


def _set_reward_weight(env, term_name: str, weight: float) -> None:
  if not math.isfinite(weight):
    raise ValueError(f"reward weight for {term_name} must be finite")
  term_cfg = env.unwrapped.reward_manager.get_term_cfg(term_name)
  term_cfg.weight = float(weight)
  if term_cfg.weight != float(weight):
    raise RuntimeError(f"failed to set reward weight for {term_name}")


def _validate_brief_algorithm(
  runner, *, failure_focused_v15: bool = False
) -> dict[str, Any]:
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
    "base_anchor_weight": alg.base_anchor_weight,
    "d0_retention_anchor_weight": alg.d0_retention_anchor_weight,
    "neighbor_retention_anchor_weight": alg.neighbor_retention_anchor_weight,
    "pre_intervention_weight": alg.pre_intervention_weight,
    "intervention_advantage_weight": alg.intervention_advantage_weight,
    "safe_bc_weight": alg.safe_bc_weight,
    "correction_distillation_weight": alg.correction_distillation_weight,
    "actor_learning_rate": alg.actor_learning_rate,
    "critic_learning_rate": alg.critic_learning_rate,
    "actor_layer_multipliers": list(alg.actor_layer_multipliers),
    "ppo_epochs": alg.num_learning_epochs,
    "ppo_clip": alg.clip_param,
    "target_kl": alg.desired_kl,
    "target_kl_early_stopping": alg.kl_early_stopping,
    "std_scale_from_base": alg.std_scale_from_base,
    "log_std_learning_rate": alg.log_std_learning_rate,
    "hard_case_policy_weight": alg.hard_case_policy_weight,
  }
  valid = (
    checks["brief_ppo_refinement"] is True
    and checks["fall_critic_absent"] is True
    and checks["intervention_critic_absent"] is True
    and checks["risk_head_absent"] is True
    and checks["base_actor_reference_absent"] is True
    and checks["retention_actor_reference_absent"] is True
    and checks["retention_bank_count"] == 0
    and checks["task_first_constrained"] is False
    and all(
      checks[name] == 0.0
      for name in (
        "base_anchor_weight",
        "d0_retention_anchor_weight",
        "neighbor_retention_anchor_weight",
        "pre_intervention_weight",
        "intervention_advantage_weight",
        "safe_bc_weight",
        "correction_distillation_weight",
        "log_std_learning_rate",
      )
    )
    and checks["ppo_epochs"] == 1
    and math.isclose(checks["ppo_clip"], 0.05)
    and checks["failure_focused_refinement"] is failure_focused_v15
    and math.isclose(
      float(checks["target_kl"]), 0.003 if failure_focused_v15 else 0.005
    )
    and checks["target_kl_early_stopping"] is True
    and math.isclose(
      checks["actor_learning_rate"], 5.0e-6 if failure_focused_v15 else 2.0e-6
    )
    and (
      not failure_focused_v15
      or math.isclose(checks["critic_learning_rate"], 1.0e-4)
    )
    and checks["actor_layer_multipliers"] == [1.0, 1.0, 1.0, 1.0]
    and math.isclose(
      checks["hard_case_policy_weight"], 0.75 if failure_focused_v15 else 0.5
    )
    and (
      not failure_focused_v15
      or math.isclose(checks["std_scale_from_base"], 0.35)
    )
  )
  if not valid:
    raise RuntimeError(f"brief PPO structural invariant failed: {checks}")
  return checks


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--resume-online-checkpoint", type=Path)
  parser.add_argument(
    "--resume-hard-case-bank",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--failure-focused-v15",
    action="store_true",
    help="Enable the frozen-context, redistributed-fall, late-failure v15 protocol.",
  )
  parser.add_argument("--deployment-context", type=Path)
  parser.add_argument("--train-domain", default="DQH")
  parser.add_argument("--neighbor-domain", default="DQNH")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--rollout-steps", type=int, default=1024)
  parser.add_argument("--online-rounds", type=int, default=5)
  parser.add_argument("--critic-burn-in-rounds", type=int, default=0)
  parser.add_argument("--candidate-num-episodes", type=int, default=128)
  parser.add_argument("--d0-check-num-episodes", type=int, default=128)
  parser.add_argument("--final-eval-num-episodes", type=int, default=128)
  parser.add_argument(
    "--candidate-fractions", nargs="+", type=float, default=(0.5, 1.0)
  )
  parser.add_argument("--hard-case-fraction", type=float, default=0.20)
  parser.add_argument("--hard-case-policy-weight", type=float, default=0.5)
  parser.add_argument("--hard-case-pre-steps", type=int, default=10)
  parser.add_argument("--hard-case-capacity", type=int, default=256)
  parser.add_argument("--late-failure-minimum-steps", type=int, default=50)
  parser.add_argument("--late-failure-maximum-steps", type=int, default=150)
  parser.add_argument("--late-failure-minimum-riser", type=int)
  parser.add_argument("--failure-discovery-max-rollouts", type=int, default=4)
  parser.add_argument("--fall-redistribution-horizon", type=int, default=100)
  parser.add_argument("--fall-redistribution-decay", type=float, default=0.97)
  parser.add_argument("--fall-redistribution-amount", type=float, default=2.0)
  parser.add_argument("--actor-learning-rate", type=float, default=2.0e-6)
  parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
  parser.add_argument("--std-scale-from-base", type=float, default=0.35)
  parser.add_argument("--fall-penalty-weight", type=float, default=-200.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--gate-device", default="cuda:0")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Allow reduced rounds/episode counts for code-path validation only.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  v15 = args.failure_focused_v15
  if not args.smoke and args.online_rounds != 5:
    raise ValueError("the formal brief PPO protocol requires exactly five rounds")
  required_fractions = (0.5, 1.0, 1.5) if v15 else (0.5, 1.0)
  if tuple(args.candidate_fractions) != required_fractions:
    raise ValueError(
      f"brief PPO candidate fractions must be exactly {required_fractions}"
    )
  if not args.smoke and args.candidate_num_episodes not in (96, 128):
    raise ValueError("candidate evaluation must use 96 or 128 paired episodes")
  required_hard_fraction = 10.0 / 64.0 if v15 else 0.20
  if not math.isclose(args.hard_case_fraction, required_hard_fraction):
    raise ValueError(
      f"brief PPO mode requires hard-case fraction {required_hard_fraction}"
    )
  required_hard_weight = 0.75 if v15 else 0.5
  if not math.isclose(args.hard_case_policy_weight, required_hard_weight):
    raise ValueError(
      f"brief PPO mode requires hard-case actor weight {required_hard_weight}"
    )
  if v15:
    if args.deployment_context is None:
      raise ValueError("failure-focused v15 requires --deployment-context")
    if args.train_domain != "DQHMED" or args.neighbor_domain != "DQNHMED":
      raise ValueError("formal v15 domains must be DQHMED and DQNHMED")
    if args.resume_hard_case_bank:
      raise ValueError("v15 must not restore a general historical hard-case bank")
    if not math.isclose(args.fall_penalty_weight, -100.0):
      raise ValueError("v15 retains half of the -4 fall event as terminal -2")
    if args.failure_discovery_max_rollouts < 1:
      raise ValueError("v15 requires at least one failure-discovery rollout")
    if not math.isclose(args.fall_redistribution_horizon, 100):
      raise ValueError("v15 requires a 100-step fall-credit horizon")
    if not math.isclose(args.fall_redistribution_decay, 0.97):
      raise ValueError("v15 requires fall-credit decay 0.97")
    if not math.isclose(args.fall_redistribution_amount, 2.0):
      raise ValueError("v15 must redistribute exactly 2 reward units per fall")
    if not args.smoke:
      formal_values = {
        "num_envs": (args.num_envs, 64),
        "rollout_steps": (args.rollout_steps, 1024),
        "critic_burn_in_rounds": (args.critic_burn_in_rounds, 0),
        "candidate_num_episodes": (args.candidate_num_episodes, 128),
        "d0_check_num_episodes": (args.d0_check_num_episodes, 128),
      }
      mismatches = {
        name: {"actual": actual, "required": required}
        for name, (actual, required) in formal_values.items()
        if actual != required
      }
      if mismatches:
        raise ValueError(f"formal v15 rollout protocol mismatch: {mismatches}")
      exact_float_values = {
        "actor_learning_rate": (args.actor_learning_rate, 5.0e-6),
        "critic_learning_rate": (args.critic_learning_rate, 1.0e-4),
        "std_scale_from_base": (args.std_scale_from_base, 0.35),
      }
      float_mismatches = {
        name: {"actual": actual, "required": required}
        for name, (actual, required) in exact_float_values.items()
        if not math.isclose(actual, required, rel_tol=0.0, abs_tol=1.0e-15)
      }
      if float_mismatches:
        raise ValueError(f"formal v15 PPO protocol mismatch: {float_mismatches}")
  if min(
    args.num_envs,
    args.rollout_steps,
    args.d0_check_num_episodes,
    args.final_eval_num_episodes,
  ) < 1:
    raise ValueError("environment, rollout, and evaluation sizes must be positive")
  if args.critic_burn_in_rounds < 0:
    raise ValueError("critic burn-in count cannot be negative")

  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.deployment_context import (
    apply_frozen_deployment_context,
    load_calibrated_deployment_context,
    load_frozen_deployment_context,
  )
  from src.tasks.stairs_cbf.hard_cases import HardCaseStateBank
  from src.tasks.stairs_cbf.online import (
    BriefPpoGateThresholds,
    FailureFocusedGateThresholds,
    backtrack_actor_state,
    brief_candidate_gate,
    brief_candidate_precheck,
    brief_d0_retention_gate,
    brief_dual_reward_weight,
    brief_target_score,
    failure_focused_candidate_gate,
    failure_focused_candidate_precheck,
    failure_focused_target_score,
  )

  task = f"Unitree-G1-Stairs-Online-{args.train_domain}"
  env_cfg = load_env_cfg(task)
  deployment_context = (
    (
      load_calibrated_deployment_context(args.deployment_context)
      if v15
      else load_frozen_deployment_context(args.deployment_context)
    )
    if args.deployment_context is not None
    else None
  )
  context_metadata = None
  if v15:
    assert deployment_context is not None
    context_metadata = apply_frozen_deployment_context(
      env_cfg, deployment_context, role="target"
    )
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env_cfg.actions["joint_pos"].enabled = True
  env_cfg.rewards["cbf_dual"].weight = 0.0
  env_cfg.rewards["fall_termination"].weight = args.fall_penalty_weight

  agent_cfg = load_rl_cfg(task)
  agent_cfg.seed = args.seed
  agent_cfg.num_steps_per_env = args.rollout_steps
  alg_cfg = agent_cfg.algorithm
  alg_cfg.actor_learning_rate = args.actor_learning_rate
  alg_cfg.critic_learning_rate = args.critic_learning_rate
  alg_cfg.actor_layer_multipliers = (1.0, 1.0, 1.0, 1.0)
  alg_cfg.log_std_learning_rate = 0.0
  alg_cfg.std_scale_from_base = args.std_scale_from_base
  alg_cfg.pre_intervention_weight = 0.0
  alg_cfg.intervention_advantage_weight = 0.0
  alg_cfg.base_anchor_weight = 0.0
  alg_cfg.d0_retention_anchor_weight = 0.0
  alg_cfg.neighbor_retention_anchor_weight = 0.0
  alg_cfg.safe_bc_weight = 0.0
  alg_cfg.correction_distillation_weight = 0.0
  alg_cfg.task_first_constrained = False
  alg_cfg.brief_ppo_refinement = True
  alg_cfg.failure_focused_refinement = v15
  alg_cfg.kl_early_stopping = True
  alg_cfg.fall_redistribution_horizon = args.fall_redistribution_horizon
  alg_cfg.fall_redistribution_decay = args.fall_redistribution_decay
  alg_cfg.fall_redistribution_amount = args.fall_redistribution_amount
  alg_cfg.hard_case_policy_weight = args.hard_case_policy_weight
  alg_cfg.clip_param = 0.05
  alg_cfg.desired_kl = 0.003 if v15 else 0.005
  alg_cfg.num_learning_epochs = 1
  alg_cfg.schedule = "fixed"
  alg_cfg.entropy_coef = 0.0
  alg_cfg.normalize_advantage_per_mini_batch = False

  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  terminal_fall_penalty = args.fall_penalty_weight * env.unwrapped.step_dt
  if v15 and not math.isclose(
    terminal_fall_penalty, -2.0, rel_tol=0.0, abs_tol=1.0e-12
  ):
    raise RuntimeError(
      "v15 terminal fall penalty must equal -2 after RewardManager dt scaling: "
      f"{terminal_fall_penalty}"
    )
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("online refinement task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)

  hard_case_bank = HardCaseStateBank(
    capacity=args.hard_case_capacity,
    bank_kind="target_late_failure" if v15 else "general_intervention",
    source_domain=args.train_domain if v15 else None,
    context_sha256=(
      deployment_context["parameters_sha256"]
      if deployment_context is not None
      else None
    ),
  )
  hard_case_generator = torch.Generator(device="cpu")
  hard_case_generator.manual_seed(args.seed + 100003)
  if args.resume_online_checkpoint is None:
    warm_start = runner.load_base_checkpoint(
      str(args.base_checkpoint.resolve()), map_location=args.device
    )
  else:
    resume_path = args.resume_online_checkpoint.resolve()
    warm_start = runner.load_online_checkpoint(
      str(resume_path), map_location=args.device
    )
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    if not v15 and args.resume_hard_case_bank and "hard_case_bank" in payload:
      hard_case_bank.load_state_dict(payload["hard_case_bank"])
    # The historical states are shared initialization data, but each v14
    # training seed gets its own curriculum sampling stream.  Reusing v13's
    # saved generator state would make the three adaptation seeds artificially
    # share the same hard-case order.

  structural_checks = _validate_brief_algorithm(
    runner, failure_focused_v15=v15
  )
  required_hard_starts = int(round(args.num_envs * args.hard_case_fraction))
  if not v15 and len(hard_case_bank) < required_hard_starts:
    raise RuntimeError(
      "formal 80/20 sampling requires a compatible historical hard-case bank; "
      f"need {required_hard_starts}, found {len(hard_case_bank)}"
    )

  obs, _ = env.reset()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  thresholds = (
    FailureFocusedGateThresholds() if v15 else BriefPpoGateThresholds()
  )
  late_failure_minimum_riser = args.late_failure_minimum_riser
  if v15 and late_failure_minimum_riser is None:
    assert deployment_context is not None
    late_failure_minimum_riser = math.ceil(
      int(deployment_context["target"]["num_steps"]) / 2
    )
  if late_failure_minimum_riser is None:
    late_failure_minimum_riser = 1
  d0_seed = args.seed + 200000
  baseline_eval = _evaluate_state(
    runner,
    _actor_state(runner.alg.actor),
    domains=("D0", args.train_domain),
    num_envs=args.d0_check_num_episodes,
    num_episodes=args.d0_check_num_episodes,
    seed=d0_seed,
    device=args.gate_device,
    runtime_filter=True,
    deployment_context=args.deployment_context,
  )
  (output_dir / "baseline_eval.json").write_text(
    json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n"
  )

  failure_discovery: list[dict[str, Any]] = []
  if v15:
    for discovery_index in range(args.failure_discovery_max_rollouts):
      before_discovery = runner.snapshot_candidate_state()
      obs, discovery_metrics = _collect_and_update(
        runner,
        obs,
        critic_only=True,
        hard_case_bank=hard_case_bank,
        hard_case_fraction=0.0,
        neighbor_command_fraction=0.0,
        neighbor_forward_scale_range=(1.0, 1.0),
        neighbor_delay_step_offset_range=(0, 0),
        hard_case_pre_steps=args.hard_case_pre_steps,
        hard_case_generator=hard_case_generator,
        persistent_hard_case_slots=False,
        late_failure_hard_cases=True,
        late_failure_minimum_steps=args.late_failure_minimum_steps,
        late_failure_maximum_steps=args.late_failure_maximum_steps,
        late_failure_minimum_riser=late_failure_minimum_riser,
      )
      # Discovery may fit the critic internally, but no discovery update is
      # retained. Only target-fall states survive into the formal protocol.
      runner.restore_candidate_state(before_discovery)
      discovery_metrics["discovery_rollout"] = discovery_index + 1
      discovery_metrics["parameters_restored_after_discovery"] = True
      failure_discovery.append(discovery_metrics)
      if len(hard_case_bank) >= required_hard_starts:
        break
    if len(hard_case_bank) < required_hard_starts:
      raise RuntimeError(
        "target-only discovery did not produce enough late-failure starts: "
        f"need {required_hard_starts}, found {len(hard_case_bank)}"
      )
    bank_audit = hard_case_bank.audit_metadata()
    if (
      bank_audit["bank_kind"] != "target_late_failure"
      or bank_audit["source_domain"] != args.train_domain
      or bank_audit["context_sha256"]
      != deployment_context["parameters_sha256"]
      or bank_audit["late_failure_entry_count"] != len(hard_case_bank)
      or bank_audit["steps_before_fall_min"] < args.late_failure_minimum_steps
      or bank_audit["steps_before_fall_max"] > args.late_failure_maximum_steps
      or bank_audit["riser_index_min"] < late_failure_minimum_riser
      or bank_audit["successful_crossing_exclusion_passed"] is not True
    ):
      raise RuntimeError(f"late-failure bank purity invariant failed: {bank_audit}")
    (output_dir / "failure_discovery.json").write_text(
      json.dumps(failure_discovery, indent=2, sort_keys=True) + "\n"
    )

  burn_in: list[dict[str, Any]] = []
  _set_reward_weight(env, "cbf_dual", 0.0)
  for _ in range(args.critic_burn_in_rounds):
    obs, metrics = _collect_and_update(
      runner,
      obs,
      critic_only=True,
      hard_case_bank=hard_case_bank,
      hard_case_fraction=args.hard_case_fraction,
      neighbor_command_fraction=0.0,
      neighbor_forward_scale_range=(1.0, 1.0),
      neighbor_delay_step_offset_range=(0, 0),
      hard_case_pre_steps=args.hard_case_pre_steps,
      hard_case_generator=hard_case_generator,
      persistent_hard_case_slots=True,
      late_failure_hard_cases=v15,
      late_failure_minimum_steps=args.late_failure_minimum_steps,
      late_failure_maximum_steps=args.late_failure_maximum_steps,
      late_failure_minimum_riser=late_failure_minimum_riser,
    )
    burn_in.append(metrics)

  runner.alg.set_critic_only(False)
  last_d0_safe_state = runner.snapshot_candidate_state()
  last_d0_safe_round = 0
  rounds: list[dict[str, Any]] = []
  for round_index in range(1, args.online_rounds + 1):
    dual_weight = 0.0 if v15 else brief_dual_reward_weight(round_index)
    _set_reward_weight(env, "cbf_dual", dual_weight)
    before = runner.snapshot_candidate_state()
    old_actor_state = _actor_state(runner.alg.actor)
    obs, full_update_metrics = _collect_and_update(
      runner,
      obs,
      critic_only=False,
      hard_case_bank=hard_case_bank,
      hard_case_fraction=args.hard_case_fraction,
      neighbor_command_fraction=0.0,
      neighbor_forward_scale_range=(1.0, 1.0),
      neighbor_delay_step_offset_range=(0, 0),
      hard_case_pre_steps=args.hard_case_pre_steps,
      hard_case_generator=hard_case_generator,
      persistent_hard_case_slots=True,
      late_failure_hard_cases=v15,
      late_failure_minimum_steps=args.late_failure_minimum_steps,
      late_failure_maximum_steps=args.late_failure_maximum_steps,
      late_failure_minimum_riser=late_failure_minimum_riser,
    )
    if (
      full_update_metrics["hard_case_start_count"]
      != full_update_metrics["hard_case_start_requested_count"]
    ):
      raise RuntimeError("the rollout did not realize its requested hard-case starts")
    realized_hard_fraction = (
      full_update_metrics["hard_case_start_count"] / args.num_envs
    )
    if not math.isclose(
      full_update_metrics["hard_case_transition_fraction"],
      realized_hard_fraction,
      rel_tol=0.0,
      abs_tol=1.0e-7,
    ):
      raise RuntimeError("hard-case transition mixture drifted from its fixed slots")

    full_candidate_state = _actor_state(runner.alg.actor)
    target_seed = args.seed + 1000 * round_index
    old_eval = _evaluate_state(
      runner,
      old_actor_state,
      domains=(args.train_domain,),
      num_envs=args.candidate_num_episodes,
      num_episodes=args.candidate_num_episodes,
      seed=target_seed,
      device=args.gate_device,
      runtime_filter=True,
      deployment_context=args.deployment_context,
    )[args.train_domain]
    variants: list[dict[str, Any]] = []
    for fraction in args.candidate_fractions:
      state = backtrack_actor_state(
        old_actor_state, full_candidate_state, fraction
      )
      update_metrics = _policy_step_metrics(runner, state, full_update_metrics)
      finite = runner.parameters_are_finite() and _finite_actor_state(state)
      precheck = (
        failure_focused_candidate_precheck
        if v15
        else brief_candidate_precheck
      )
      precheck_reasons = precheck(
        update_metrics=update_metrics,
        parameters_finite=finite,
        thresholds=thresholds,
      )
      score_function = failure_focused_target_score if v15 else brief_target_score
      candidate_eval: dict[str, Any] = {}
      scores: dict[str, dict[str, float]] = {
        "old": score_function(old_eval),
        "candidate": {"total": float("nan")},
      }
      accepted = False
      reasons = list(precheck_reasons)
      if not reasons:
        candidate_eval = _evaluate_state(
          runner,
          state,
          domains=(args.train_domain,),
          num_envs=args.candidate_num_episodes,
          num_episodes=args.candidate_num_episodes,
          seed=target_seed,
          device=args.gate_device,
          runtime_filter=True,
          deployment_context=args.deployment_context,
        )[args.train_domain]
        gate = failure_focused_candidate_gate if v15 else brief_candidate_gate
        accepted, reasons, scores = gate(
          update_metrics=update_metrics,
          old_eval=old_eval,
          candidate_eval=candidate_eval,
          parameters_finite=finite,
          thresholds=thresholds,
        )
      variants.append(
        {
          "fraction": fraction,
          "state": state,
          "update_metrics": update_metrics,
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
      selected_score = selected["scores"]["candidate"]["total"]

    d0_check: dict[str, Any] | None = None
    d0_rollback = False
    if round_index % 2 == 0:
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
      d0_passed, d0_reasons = brief_d0_retention_gate(
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
      "dual_cbf_reward_weight": dual_weight,
      "fall_reward_weight": args.fall_penalty_weight,
      "target_gate_accepted": target_gate_accepted,
      "selected_candidate_fraction": selected_fraction,
      "selected_target_score": selected_score,
      "old_target_eval": old_eval,
      "full_update_metrics": full_update_metrics,
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
      hard_case_bank=hard_case_bank,
      hard_case_generator=hard_case_generator,
    )
    (output_dir / "online_rounds.json").write_text(
      json.dumps(rounds, indent=2, sort_keys=True) + "\n"
    )

  final_actor_state = _actor_state(runner.alg.actor)
  final_domains = (
    ("D0", "DQH", args.train_domain, args.neighbor_domain)
    if v15
    else ("D0", args.train_domain, args.neighbor_domain)
  )
  final_eval = _evaluate_state(
    runner,
    final_actor_state,
    domains=final_domains,
    num_envs=args.final_eval_num_episodes,
    num_episodes=args.final_eval_num_episodes,
    seed=args.seed + 900000,
    device=args.gate_device,
    runtime_filter=True,
    deployment_context=args.deployment_context,
  )
  final_path = output_dir / "accepted_final.pt"
  result = {
    "method": (
      "Failure-Focused Brief PPO v15"
      if v15
      else "CBF-Guided Brief PPO Refinement v14"
    ),
    "formal_protocol": not args.smoke,
    "training_vs_paper_evidence": (
      "All per-round gates are point estimates; final independent three-seed "
      "audits and confidence intervals are produced separately."
    ),
    "seed": args.seed,
    "task": task,
    "train_domain": args.train_domain,
    "neighbor_domain": args.neighbor_domain,
    "runtime_cbf": True,
    "raw_policy_action_for_ppo": True,
    "executed_action": "runtime_cbf_safe_action",
    "reward": (
      "task + terminal fall (-2) + redistributed pre-fall penalty (-2)"
      if v15
      else "task + fall + scheduled dual CBF"
    ),
    "dual_reward_schedule": (
      {"rounds_1_5": 0.0}
      if v15
      else {"rounds_1_2": 0.0, "rounds_3_5": 0.02}
    ),
    "fall_redistribution": (
      {
        "enabled": True,
        "terminal_penalty": terminal_fall_penalty,
        "redistributed_penalty": -args.fall_redistribution_amount,
        "horizon": args.fall_redistribution_horizon,
        "decay": args.fall_redistribution_decay,
        "total_undiscounted_fall_penalty": -4.0,
      }
      if v15
      else {"enabled": False}
    ),
    "normal_start_fraction": 1.0 - args.hard_case_fraction,
    "hard_case_fraction": args.hard_case_fraction,
    "hard_case_policy_weight": args.hard_case_policy_weight,
    "candidate_fractions": list(args.candidate_fractions),
    "candidate_paired_episodes": args.candidate_num_episodes,
    "d0_check_period_rounds": 2,
    "d0_success_tolerance": thresholds.d0_success_tolerance,
    "target_fall_tolerance": thresholds.maximum_target_fall_increase,
    "training_maximum_kl": thresholds.maximum_kl,
    "catastrophic_cbf_demand_ratio_cap": (
      thresholds.maximum_cbf_demand_ratio if v15 else None
    ),
    "deployment_context": context_metadata,
    "failure_discovery": failure_discovery,
    "late_failure_bank": hard_case_bank.audit_metadata(),
    "structural_checks": structural_checks,
    "warm_start": warm_start,
    "base_checkpoint": str(args.base_checkpoint.resolve()),
    "resume_online_checkpoint": (
      str(args.resume_online_checkpoint.resolve())
      if args.resume_online_checkpoint is not None
      else None
    ),
    "critic_burn_in": burn_in,
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
    hard_case_bank=hard_case_bank,
    hard_case_generator=hard_case_generator,
  )
  (output_dir / "brief_ppo_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  env.close()


if __name__ == "__main__":
  main()

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


def _validate_brief_algorithm(runner) -> dict[str, Any]:
  alg = runner.alg
  checks = {
    "brief_ppo_refinement": alg.brief_ppo_refinement,
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
    and math.isclose(float(checks["target_kl"]), 0.005)
    and checks["target_kl_early_stopping"] is True
    and math.isclose(checks["actor_learning_rate"], 2.0e-6)
    and checks["actor_layer_multipliers"] == [1.0, 1.0, 1.0, 1.0]
    and math.isclose(checks["hard_case_policy_weight"], 0.5)
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
  if not args.smoke and args.online_rounds != 5:
    raise ValueError("the formal brief PPO protocol requires exactly five rounds")
  if tuple(args.candidate_fractions) != (0.5, 1.0):
    raise ValueError("brief PPO candidate fractions must be exactly 0.5 and 1.0")
  if not args.smoke and args.candidate_num_episodes not in (96, 128):
    raise ValueError("candidate evaluation must use 96 or 128 paired episodes")
  if not math.isclose(args.hard_case_fraction, 0.20):
    raise ValueError("brief PPO requires 20% hard-case rollout starts")
  if not math.isclose(args.hard_case_policy_weight, 0.5):
    raise ValueError("the formal brief PPO run uses hard-case actor weight 0.5")
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
  from src.tasks.stairs_cbf.hard_cases import HardCaseStateBank
  from src.tasks.stairs_cbf.online import (
    BriefPpoGateThresholds,
    backtrack_actor_state,
    brief_candidate_gate,
    brief_candidate_precheck,
    brief_d0_retention_gate,
    brief_dual_reward_weight,
    brief_target_score,
  )

  task = f"Unitree-G1-Stairs-Online-{args.train_domain}"
  env_cfg = load_env_cfg(task)
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
  alg_cfg.kl_early_stopping = True
  alg_cfg.hard_case_policy_weight = args.hard_case_policy_weight
  alg_cfg.clip_param = 0.05
  alg_cfg.desired_kl = 0.005
  alg_cfg.num_learning_epochs = 1
  alg_cfg.schedule = "fixed"
  alg_cfg.entropy_coef = 0.0
  alg_cfg.normalize_advantage_per_mini_batch = False

  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("online refinement task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)

  hard_case_bank = HardCaseStateBank(capacity=args.hard_case_capacity)
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
    if args.resume_hard_case_bank and "hard_case_bank" in payload:
      hard_case_bank.load_state_dict(payload["hard_case_bank"])
    # The historical states are shared initialization data, but each v14
    # training seed gets its own curriculum sampling stream.  Reusing v13's
    # saved generator state would make the three adaptation seeds artificially
    # share the same hard-case order.

  structural_checks = _validate_brief_algorithm(runner)
  required_hard_starts = int(round(args.num_envs * args.hard_case_fraction))
  if len(hard_case_bank) < required_hard_starts:
    raise RuntimeError(
      "formal 80/20 sampling requires a compatible historical hard-case bank; "
      f"need {required_hard_starts}, found {len(hard_case_bank)}"
    )

  obs, _ = env.reset()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  thresholds = BriefPpoGateThresholds()
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
  )
  (output_dir / "baseline_eval.json").write_text(
    json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n"
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
    )
    burn_in.append(metrics)

  runner.alg.set_critic_only(False)
  last_d0_safe_state = runner.snapshot_candidate_state()
  last_d0_safe_round = 0
  rounds: list[dict[str, Any]] = []
  for round_index in range(1, args.online_rounds + 1):
    dual_weight = brief_dual_reward_weight(round_index)
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
    )[args.train_domain]
    variants: list[dict[str, Any]] = []
    for fraction in args.candidate_fractions:
      state = backtrack_actor_state(
        old_actor_state, full_candidate_state, fraction
      )
      update_metrics = _policy_step_metrics(runner, state, full_update_metrics)
      finite = runner.parameters_are_finite() and _finite_actor_state(state)
      precheck_reasons = brief_candidate_precheck(
        update_metrics=update_metrics,
        parameters_finite=finite,
        thresholds=thresholds,
      )
      candidate_eval: dict[str, Any] = {}
      scores: dict[str, dict[str, float]] = {
        "old": brief_target_score(old_eval),
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
        )[args.train_domain]
        accepted, reasons, scores = brief_candidate_gate(
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
  final_eval = _evaluate_state(
    runner,
    final_actor_state,
    domains=("D0", args.train_domain, args.neighbor_domain),
    num_envs=args.final_eval_num_episodes,
    num_episodes=args.final_eval_num_episodes,
    seed=args.seed + 900000,
    device=args.gate_device,
    runtime_filter=True,
  )
  final_path = output_dir / "accepted_final.pt"
  result = {
    "method": "CBF-Guided Brief PPO Refinement",
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
    "reward": "task + fall + scheduled dual CBF",
    "dual_reward_schedule": {"rounds_1_2": 0.0, "rounds_3_5": 0.02},
    "normal_start_fraction": 0.80,
    "hard_case_fraction": args.hard_case_fraction,
    "hard_case_policy_weight": args.hard_case_policy_weight,
    "candidate_fractions": list(args.candidate_fractions),
    "candidate_paired_episodes": args.candidate_num_episodes,
    "d0_check_period_rounds": 2,
    "d0_success_tolerance": thresholds.d0_success_tolerance,
    "target_fall_tolerance": thresholds.maximum_target_fall_increase,
    "training_maximum_kl": thresholds.maximum_kl,
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

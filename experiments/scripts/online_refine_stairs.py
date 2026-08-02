"""Transactional CBF-protected PPO refinement on a fixed deployment stair."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch


def _actor_state(actor) -> dict[str, torch.Tensor]:
  return {key: value.detach().clone() for key, value in actor.state_dict().items()}


def _evaluate_state(
  runner,
  actor_state: dict[str, torch.Tensor],
  *,
  domains: tuple[str, ...],
  num_envs: int,
  num_episodes: int,
  seed: int,
  device: str,
  repeats: int = 1,
  runtime_filter: bool = True,
) -> dict[str, dict[str, Any]]:
  """Evaluate one actor in isolated CUDA processes.

  MuJoCo-Warp CUDA graphs are not reliably reusable after dozens of
  create/close cycles in one Python process (observed as capture error 901).
  The training environment remains in this parent process, while each paired
  replicate gets a fresh subprocess and therefore a fresh Warp context.
  """
  if num_envs != num_episodes:
    raise ValueError(
      "transactional paired evaluation requires --eval-num-envs equal to "
      "--eval-num-episodes; recycled environments are not independent pairs"
    )
  output: dict[str, dict[str, Any]] = {}
  repo = Path(__file__).resolve().parents[2]
  checkpoint_payload = runner.alg.save()
  checkpoint_payload["actor_state_dict"] = {
    key: value.detach().cpu() for key, value in actor_state.items()
  }
  # MjlabOnPolicyRunner.load() always returns checkpoint metadata even when
  # load_cfg requests the actor only.
  checkpoint_payload.setdefault("iter", 0)
  checkpoint_payload.setdefault("infos", {})
  with tempfile.TemporaryDirectory(prefix="stairs-paired-eval-") as temp_dir:
    temp_root = Path(temp_dir)
    checkpoint = temp_root / "actor.pt"
    torch.save(checkpoint_payload, checkpoint)
    for domain in domains:
      replicate_summaries = []
      for repeat in range(repeats):
        stem = f"{domain}-seed{seed + repeat}"
        output_json = temp_root / f"{stem}.json"
        output_csv = temp_root / f"{stem}.csv"
        command = [
          sys.executable,
          str(repo / "experiments/scripts/evaluate_online_stairs.py"),
          "--repo",
          str(repo),
          "--task",
          f"Unitree-G1-Stairs-Online-{domain}",
          "--checkpoint",
          str(checkpoint),
          "--num-envs",
          str(num_envs),
          "--num-episodes",
          str(num_episodes),
          "--seed",
          str(seed + repeat),
          "--device",
          device,
          "--runtime-filter",
          "on" if runtime_filter else "off",
          "--one-episode-per-env",
          "--output-json",
          str(output_json),
          "--output-csv",
          str(output_csv),
        ]
        completed = subprocess.run(
          command,
          cwd=repo,
          check=False,
          capture_output=True,
          text=True,
        )
        if completed.returncode != 0:
          diagnostic = "\n".join(
            (completed.stdout + "\n" + completed.stderr).splitlines()[-120:]
          )
          raise RuntimeError(
            f"isolated paired evaluation failed for {stem}:\n{diagnostic}"
          )
        summary = json.loads(output_json.read_text())
        replicate_summaries.append(summary)
      aggregate: dict[str, Any] = {
        "task": f"Unitree-G1-Stairs-Online-{domain}",
        "num_episodes": num_episodes * repeats,
        "repeats": repeats,
        "seeds": [seed + repeat for repeat in range(repeats)],
        "replicates": replicate_summaries,
        "runtime_filter": runtime_filter,
        "paired_one_initial_episode_per_env": True,
        "initial_state_signatures": [
          summary["initial_state_signature"] for summary in replicate_summaries
        ],
      }
      for key in (
        "success_rate",
        "fall_rate",
        "timeout_rate",
        "mean_reached_riser",
        "mean_return",
        "mean_episode_time_s",
        "intervention_per_riser",
        "correction_mean",
        "mean_correction_p95",
        "would_intervene_per_riser",
        "counterfactual_correction_mean",
        "mean_counterfactual_correction_p95",
        "geometric_active_fraction",
        "intervention_fraction",
        "would_intervene_fraction",
        "nominal_violation_fraction",
        "filtered_violation_fraction",
      ):
        values = [float(summary[key]) for summary in replicate_summaries]
        aggregate[key] = sum(values) / len(values)
        aggregate[f"{key}_std"] = (
          math.sqrt(sum((value - aggregate[key]) ** 2 for value in values) / (len(values) - 1))
          if len(values) > 1
          else 0.0
        )
      successful_times = [
        float(summary["mean_success_time_s"])
        for summary in replicate_summaries
        if summary["mean_success_time_s"] is not None
      ]
      aggregate["mean_success_time_s"] = (
        sum(successful_times) / len(successful_times)
        if successful_times
        else None
      )
      for key in (
        "minimum_cbf_h",
        "minimum_nominal_margin",
        "minimum_filtered_margin",
      ):
        finite_values = [
          float(summary[key])
          for summary in replicate_summaries
          if summary[key] is not None
        ]
        aggregate[key] = min(finite_values) if finite_values else None
      output[domain] = aggregate
  return output


def _total_actor_kl(
  runner,
  base_state: dict[str, torch.Tensor],
  actor_state: dict[str, torch.Tensor] | None = None,
) -> float:
  obs = runner.alg.storage.observations.flatten(0, 1)
  current_state = _actor_state(runner.alg.actor)
  evaluated_state = current_state if actor_state is None else actor_state
  with torch.no_grad():
    runner.alg.actor.load_state_dict(base_state, strict=True)
    base_mean = runner.alg.actor(obs).detach().clone()
    base_std = runner.alg.actor.distribution.std_param.detach().clone()
    runner.alg.actor.load_state_dict(evaluated_state, strict=True)
    candidate_mean = runner.alg.actor(obs).detach()
    # Cross-round drift constrains the deterministic deployment behavior.  A
    # separately bounded/reduced exploration std must not by itself exhaust
    # the mean-policy KL budget after a rejected candidate.
    kl = 0.5 * torch.sum(
      ((candidate_mean - base_mean) / base_std.clamp_min(1.0e-6)) ** 2,
      dim=-1,
    ).mean()
    runner.alg.actor.load_state_dict(current_state, strict=True)
  return float(kl)


def _collect_and_update(
  runner,
  obs,
  *,
  critic_only: bool,
  hard_case_bank,
  hard_case_fraction: float,
  neighbor_command_fraction: float,
  neighbor_forward_scale_range: tuple[float, float],
  neighbor_delay_step_offset_range: tuple[int, int],
  hard_case_pre_steps: int,
  hard_case_generator: torch.Generator,
):
  from rsl_rl.utils import check_nan
  from src.tasks.stairs_cbf.hard_cases import (
    capture_hard_case_state,
    reset_rollout_with_hard_cases,
  )

  del obs
  runner.alg.set_critic_only(critic_only)
  runner.alg.clear_cbf_rollout()
  runner.alg.train_mode()
  obs, start_metrics = reset_rollout_with_hard_cases(
    runner.env,
    hard_case_bank,
    hard_case_fraction=hard_case_fraction,
    neighbor_command_fraction=neighbor_command_fraction,
    neighbor_forward_scale_range=neighbor_forward_scale_range,
    neighbor_delay_step_offset_range=neighbor_delay_step_offset_range,
    generator=hard_case_generator,
  )
  state_history = deque(maxlen=hard_case_pre_steps + 1)
  valid_steps = torch.zeros(
    runner.env.num_envs, dtype=torch.long, device=runner.env.device
  )
  previous_intervention = torch.zeros(
    runner.env.num_envs, dtype=torch.bool, device=runner.env.device
  )
  bank_added = 0
  # Use no_grad rather than inference_mode because critic normalization is
  # intentionally updated and must remain rollback-compatible.
  with torch.no_grad():
    for _ in range(runner.cfg["num_steps_per_env"]):
      state_history.append(capture_hard_case_state(runner.env.unwrapped))
      actions = runner.alg.act(obs)
      obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
      check_nan(obs, rewards, dones)
      actual_intervention = extras.get("cbf_intervened")
      magnitude = extras.get("cbf_intervention_magnitude")
      riser_index = extras.get("online_stair_index")
      if (
        actual_intervention is not None
        and magnitude is not None
        and riser_index is not None
        and len(state_history) == hard_case_pre_steps + 1
      ):
        event = actual_intervention.bool() & ~previous_intervention
        eligible = event & (valid_steps >= hard_case_pre_steps)
        event_ids = eligible.nonzero(as_tuple=False).flatten()
        if len(event_ids) > 0:
          bank_added += hard_case_bank.add_batched(
            state_history[0],
            event_ids,
            magnitude[event_ids],
            riser_index[event_ids],
          )
      done_mask = dones.bool()
      previous_intervention = torch.where(
        done_mask,
        torch.zeros_like(previous_intervention),
        actual_intervention.bool()
        if actual_intervention is not None
        else torch.zeros_like(previous_intervention),
      )
      valid_steps = torch.where(
        done_mask, torch.zeros_like(valid_steps), valid_steps + 1
      )
      obs = obs.to(runner.device)
      rewards = rewards.to(runner.device)
      dones = dones.to(runner.device)
      runner.alg.process_env_step(obs, rewards, dones, extras)
    credit_metrics = runner.alg.relabel_pre_intervention_costs()
    runner.alg.compute_returns(obs)
    advantage_metrics = runner.alg.shape_intervention_advantages()
  losses = runner.alg.update()
  losses.update(credit_metrics)
  losses.update(advantage_metrics)
  losses.update(start_metrics)
  losses.update(
    {
      "hard_case_bank_added": bank_added,
      "hard_case_bank_size_after_rollout": len(hard_case_bank),
      "hard_case_bank_total_events": hard_case_bank.total_added,
      "hard_case_pre_steps": hard_case_pre_steps,
    }
  )
  return obs, losses


def _save_checkpoint(
  runner,
  path: Path,
  *,
  iteration: int,
  metadata: dict[str, Any],
  hard_case_bank=None,
  hard_case_generator: torch.Generator | None = None,
) -> None:
  payload = runner.alg.save()
  payload["iter"] = iteration
  payload["infos"] = {"online_refinement": metadata}
  if hard_case_bank is not None:
    payload["hard_case_bank"] = hard_case_bank.state_dict()
  if hard_case_generator is not None:
    payload["hard_case_generator_state"] = hard_case_generator.get_state()
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument(
    "--resume-online-checkpoint",
    type=Path,
    help="Accepted 799-D checkpoint to refine; base checkpoint remains the KL/retention reference.",
  )
  parser.add_argument(
    "--resume-hard-case-bank",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Restore hard cases only when resuming within the same target domain.",
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--rollout-steps", type=int, default=256)
  parser.add_argument("--critic-burn-in-rounds", type=int, default=2)
  parser.add_argument("--critic-burn-in-max-rounds", type=int, default=4)
  parser.add_argument(
    "--critic-min-explained-variance", type=float, default=0.50
  )
  parser.add_argument("--online-rounds", type=int, default=2)
  parser.add_argument("--eval-num-envs", type=int, default=8)
  parser.add_argument("--eval-num-episodes", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
  parser.add_argument("--pre-intervention-weight", type=float, default=0.20)
  parser.add_argument("--std-scale-from-base", type=float, default=0.35)
  parser.add_argument("--safe-bc-weight", type=float, default=0.0)
  parser.add_argument("--hard-case-fraction", type=float, default=0.20)
  parser.add_argument("--base-anchor-weight", type=float, default=0.01)
  parser.add_argument(
    "--intervention-advantage-weight", type=float, default=0.075
  )
  parser.add_argument(
    "--neighbor-command-fraction",
    type=float,
    default=0.15,
    help="Fraction of bottom starts with bounded neighboring joystick speed/delay.",
  )
  parser.add_argument(
    "--neighbor-forward-scale-range",
    nargs=2,
    type=float,
    default=(0.90, 1.10),
    metavar=("LOW", "HIGH"),
  )
  parser.add_argument(
    "--neighbor-delay-step-offset-range",
    nargs=2,
    type=int,
    default=(-2, 2),
    metavar=("LOW", "HIGH"),
  )
  parser.add_argument("--hard-case-pre-steps", type=int, default=10)
  parser.add_argument("--hard-case-capacity", type=int, default=256)
  parser.add_argument(
    "--adaptive-std",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      "Legacy exploration adaptation. This is disabled by default and cannot "
      "be combined with a non-zero frozen base-policy KL anchor."
    ),
  )
  parser.add_argument("--target-intervention-per-riser", type=float, default=0.10)
  parser.add_argument("--std-adaptation-rate", type=float, default=0.10)
  parser.add_argument(
    "--maximum-target-fall-rate",
    type=float,
    default=0.0,
    help="Hard candidate safety gate; formal shielded refinement defaults to zero falls.",
  )
  parser.add_argument(
    "--independence-audit",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Evaluate the final actor with and without the CBF in simulation.",
  )
  parser.add_argument(
    "--resume-std-scale",
    type=float,
    default=1.0,
    help="Additional bounded exploration scaling after loading an accepted checkpoint.",
  )
  parser.add_argument(
    "--fall-penalty-weight",
    type=float,
    help="Override the fall-only reward weight; MJLab multiplies it by dt.",
  )
  parser.add_argument(
    "--train-runtime-filter",
    choices=("on", "off"),
    default="on",
    help="Execute the CBF during rollout collection. Off is simulation-only finalization.",
  )
  parser.add_argument(
    "--gate-runtime-filter",
    choices=("on", "off"),
    default="on",
    help="Deployment mode used by the transactional D0/target/neighbor gate.",
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--gate-device",
    default="cuda:0",
    help="Physics device for paired candidate acceptance (GPU by default).",
  )
  parser.add_argument(
    "--gate-repeats",
    type=int,
    default=3,
    help="Independent fixed-seed GPU evaluation replicates per policy/domain.",
  )
  parser.add_argument(
    "--train-domain",
    default="DQ",
    help="Target domain used for online rollouts (DQ quick prototype or D4 formal).",
  )
  parser.add_argument("--neighbor-domain", default="DQN")
  parser.add_argument(
    "--baseline-domains",
    nargs="+",
    default=["D0", "D1", "D2", "D3", "D4", "D5", "DQ", "DQN"],
  )
  args = parser.parse_args()
  if args.adaptive_std and args.base_anchor_weight > 0.0:
    raise ValueError(
      "--adaptive-std changes the action distribution independently of the "
      "frozen base policy; use --no-adaptive-std with a non-zero KL anchor"
    )
  if args.resume_std_scale != 1.0 and args.base_anchor_weight > 0.0:
    raise ValueError(
      "--resume-std-scale must remain 1.0 with a non-zero base-policy KL anchor"
    )
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.online import (
    CandidateGateThresholds,
    adaptive_cbf_std_factor,
    candidate_gate,
    candidate_gate_intervals,
    candidate_precheck,
    cbf_independence_gate,
    safe_improvement_score,
  )
  from src.tasks.stairs_cbf.hard_cases import HardCaseStateBank

  if not 0.0 <= args.hard_case_fraction <= 1.0:
    raise ValueError("--hard-case-fraction must be in [0, 1]")
  if not 0.0 <= args.neighbor_command_fraction <= 1.0:
    raise ValueError("--neighbor-command-fraction must be in [0, 1]")
  if args.hard_case_fraction + args.neighbor_command_fraction > 1.0:
    raise ValueError("hard-case and neighboring-command fractions exceed one")
  if not 0.0 <= args.maximum_target_fall_rate <= 1.0:
    raise ValueError("--maximum-target-fall-rate must be in [0, 1]")
  if args.hard_case_pre_steps < 1:
    raise ValueError("--hard-case-pre-steps must be positive")
  if not 0 <= args.critic_burn_in_rounds <= args.critic_burn_in_max_rounds:
    raise ValueError("critic burn-in minimum/max rounds are inconsistent")

  task = f"Unitree-G1-Stairs-Online-{args.train_domain}"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env_cfg.actions["joint_pos"].enabled = args.train_runtime_filter == "on"
  if args.fall_penalty_weight is not None:
    env_cfg.rewards["fall_termination"].weight = args.fall_penalty_weight
  agent_cfg = load_rl_cfg(task)
  agent_cfg.seed = args.seed
  agent_cfg.num_steps_per_env = args.rollout_steps
  agent_cfg.algorithm.actor_learning_rate = args.actor_learning_rate
  agent_cfg.algorithm.critic_learning_rate = args.critic_learning_rate
  agent_cfg.algorithm.pre_intervention_weight = args.pre_intervention_weight
  agent_cfg.algorithm.base_anchor_weight = args.base_anchor_weight
  agent_cfg.algorithm.intervention_advantage_weight = (
    args.intervention_advantage_weight
  )
  agent_cfg.algorithm.std_scale_from_base = args.std_scale_from_base
  agent_cfg.algorithm.safe_bc_weight = args.safe_bc_weight
  agent_cfg.algorithm.use_counterfactual_cbf_credit = (
    args.train_runtime_filter == "off"
  )
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
      str(args.base_checkpoint), map_location=args.device
    )
  else:
    warm_start = runner.load_online_checkpoint(
      str(args.resume_online_checkpoint.resolve()),
      map_location=args.device,
    )
    # A backtracked candidate's saved Adam moments correspond to the full PPO
    # step, not the accepted fractional parameter point. Start each accepted
    # round with the configured conservative optimizer instead.
    runner.alg.scale_exploration_std(args.resume_std_scale)
    warm_start |= {
      "resume_online_checkpoint": str(args.resume_online_checkpoint.resolve()),
      "resume_std_scale": args.resume_std_scale,
    }
    resume_payload = torch.load(
      args.resume_online_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    if args.resume_hard_case_bank and "hard_case_bank" in resume_payload:
      hard_case_bank.load_state_dict(resume_payload["hard_case_bank"])
    if args.resume_hard_case_bank and "hard_case_generator_state" in resume_payload:
      hard_case_generator.set_state(resume_payload["hard_case_generator_state"])
  obs, _ = env.reset()
  current_actor_state = _actor_state(runner.alg.actor)
  base_payload = torch.load(
    args.base_checkpoint.resolve(), map_location=args.device, weights_only=False
  )
  from src.tasks.stairs_cbf.online import backtrack_actor_state

  # The anchor always points to the original pretrained mean policy, including
  # when this run resumes a later accepted online checkpoint.
  runner.alg.set_base_actor_reference(base_payload["actor_state_dict"])

  # Keep the accepted bounded std and frozen normalizer, but use the original
  # base MLP as the cross-round drift/retention reference.
  base_actor_state = backtrack_actor_state(
    base_payload["actor_state_dict"], current_actor_state, 0.0
  )
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  baseline_eval = _evaluate_state(
    runner,
    base_actor_state,
    domains=tuple(args.baseline_domains),
    num_envs=args.eval_num_envs,
    num_episodes=args.eval_num_episodes,
    seed=args.seed,
    device=args.device,
    repeats=args.gate_repeats,
    runtime_filter=args.gate_runtime_filter == "on",
  )
  (output_dir / "baseline_ood_matrix.json").write_text(
    json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n"
  )

  burn_in: list[dict[str, Any]] = []
  while (
    len(burn_in) < args.critic_burn_in_rounds
    or (
      bool(burn_in)
      and burn_in[-1]["explained_variance_before_update"]
      < args.critic_min_explained_variance
      and len(burn_in) < args.critic_burn_in_max_rounds
    )
  ):
    obs, metrics = _collect_and_update(
      runner,
      obs,
      critic_only=True,
      hard_case_bank=hard_case_bank,
      hard_case_fraction=args.hard_case_fraction,
      neighbor_command_fraction=args.neighbor_command_fraction,
      neighbor_forward_scale_range=tuple(args.neighbor_forward_scale_range),
      neighbor_delay_step_offset_range=tuple(
        args.neighbor_delay_step_offset_range
      ),
      hard_case_pre_steps=args.hard_case_pre_steps,
      hard_case_generator=hard_case_generator,
    )
    burn_in.append(metrics)
    (output_dir / "critic_burn_in.json").write_text(
      json.dumps(burn_in, indent=2, sort_keys=True) + "\n"
    )
  if (
    burn_in
    and burn_in[-1]["explained_variance_before_update"]
    < args.critic_min_explained_variance
  ):
    raise RuntimeError(
      "critic calibration did not reach the explained-variance threshold: "
      f"{burn_in[-1]['explained_variance_before_update']:.4f} < "
      f"{args.critic_min_explained_variance:.4f}"
    )
  runner.alg.set_critic_only(False)

  thresholds = CandidateGateThresholds(
    maximum_target_fall_rate=args.maximum_target_fall_rate
  )
  accepted_state = runner.snapshot_candidate_state()
  accepted_total_kl = _total_actor_kl(runner, base_actor_state)
  rounds: list[dict[str, Any]] = []
  for round_index in range(1, args.online_rounds + 1):
    before = runner.snapshot_candidate_state()
    old_actor_state = _actor_state(runner.alg.actor)
    obs, update_metrics = _collect_and_update(
      runner,
      obs,
      critic_only=False,
      hard_case_bank=hard_case_bank,
      hard_case_fraction=args.hard_case_fraction,
      neighbor_command_fraction=args.neighbor_command_fraction,
      neighbor_forward_scale_range=tuple(args.neighbor_forward_scale_range),
      neighbor_delay_step_offset_range=tuple(
        args.neighbor_delay_step_offset_range
      ),
      hard_case_pre_steps=args.hard_case_pre_steps,
      hard_case_generator=hard_case_generator,
    )
    candidate_actor_state = _actor_state(runner.alg.actor)
    old_total_kl = _total_actor_kl(
      runner, base_actor_state, actor_state=old_actor_state
    )
    total_kl = _total_actor_kl(
      runner, base_actor_state, actor_state=candidate_actor_state
    )
    precheck_reasons = candidate_precheck(
      update_metrics=update_metrics,
      total_kl_from_base=total_kl,
      parameters_finite=runner.parameters_are_finite(),
      thresholds=thresholds,
    )

    candidate_path = output_dir / f"candidate_round_{round_index:03d}.pt"
    _save_checkpoint(
      runner,
      candidate_path,
      iteration=round_index,
      metadata={
        "accepted": False,
        "stage": "candidate",
        "update_metrics": update_metrics,
        "total_kl_from_base": total_kl,
      },
      hard_case_bank=hard_case_bank,
      hard_case_generator=hard_case_generator,
    )

    old_eval: dict[str, dict[str, Any]] = {}
    candidate_eval: dict[str, dict[str, Any]] = {}
    gate_intervals: dict[str, tuple[float, float, float]] = {}
    gate_scores: dict[str, dict[str, float]] = {}
    accepted = False
    reasons = list(precheck_reasons)
    if not reasons:
      old_eval = _evaluate_state(
        runner,
        old_actor_state,
        domains=("D0", args.train_domain, args.neighbor_domain),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
        runtime_filter=args.gate_runtime_filter == "on",
      )
      candidate_eval = _evaluate_state(
        runner,
        candidate_actor_state,
        domains=("D0", args.train_domain, args.neighbor_domain),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
        runtime_filter=args.gate_runtime_filter == "on",
      )
      accepted, reasons = candidate_gate(
        update_metrics=update_metrics,
        old_eval=old_eval,
        candidate_eval=candidate_eval,
        base_d0_success=baseline_eval["D0"]["success_rate"],
        old_total_kl_from_base=old_total_kl,
        total_kl_from_base=total_kl,
        parameters_finite=runner.parameters_are_finite(),
        thresholds=thresholds,
        target_domain=args.train_domain,
        retention_domain="D0",
        neighbor_domain=args.neighbor_domain,
      )
      gate_intervals = candidate_gate_intervals(
        old_eval=old_eval,
        candidate_eval=candidate_eval,
        thresholds=thresholds,
        target_domain=args.train_domain,
        neighbor_domain=args.neighbor_domain,
        old_total_kl_from_base=old_total_kl,
        total_kl_from_base=total_kl,
      )
      gate_scores = {
        "old": safe_improvement_score(
          old_eval[args.train_domain],
          total_kl_from_base=old_total_kl,
        ),
        "candidate": safe_improvement_score(
          candidate_eval[args.train_domain],
          total_kl_from_base=total_kl,
        ),
      }

    adaptive_std_factor = 1.0
    if accepted:
      if args.adaptive_std:
        adaptive_std_factor = adaptive_cbf_std_factor(
          update_metrics["cbf_intervention_per_riser"],
          target_intervention_per_riser=args.target_intervention_per_riser,
          adaptation_rate=args.std_adaptation_rate,
          fall_count=update_metrics["fall_event_count"],
        )
        runner.alg.scale_exploration_std(adaptive_std_factor)
      accepted_state = runner.snapshot_candidate_state()
      accepted_total_kl = total_kl
      accepted_path = output_dir / f"accepted_round_{round_index:03d}.pt"
      _save_checkpoint(
        runner,
        accepted_path,
        iteration=round_index,
        metadata={
          "accepted": True,
          "update_metrics": update_metrics,
          "total_kl_from_base": total_kl,
          "old_eval": old_eval,
          "candidate_eval": candidate_eval,
          "gate_intervals": gate_intervals,
          "adaptive_std_factor": adaptive_std_factor,
        },
        hard_case_bank=hard_case_bank,
        hard_case_generator=hard_case_generator,
      )
    else:
      runner.restore_candidate_state(before)
      runner.reduce_after_rejection()
      accepted_state = runner.snapshot_candidate_state()

    record = {
      "round": round_index,
      "accepted": accepted,
      "rejection_reasons": reasons,
      "update_metrics": update_metrics,
      "total_kl_from_base": total_kl,
      "old_total_kl_from_base": old_total_kl,
      "old_eval": old_eval,
      "candidate_eval": candidate_eval,
      "gate_intervals": gate_intervals,
      "safe_improvement_scores": gate_scores,
      "adaptive_std_factor": adaptive_std_factor,
      "candidate_checkpoint": str(candidate_path),
    }
    rounds.append(record)
    (output_dir / "online_rounds.json").write_text(
      json.dumps(rounds, indent=2, sort_keys=True) + "\n"
    )

  runner.restore_candidate_state(accepted_state)
  final_eval = _evaluate_state(
    runner,
    _actor_state(runner.alg.actor),
    domains=("D0", args.train_domain, args.neighbor_domain),
    num_envs=args.eval_num_envs,
    num_episodes=args.eval_num_episodes,
    seed=args.seed,
    device=args.gate_device,
    repeats=args.gate_repeats,
    runtime_filter=args.gate_runtime_filter == "on",
  )
  independence_audit: dict[str, Any] | None = None
  if args.independence_audit:
    final_actor_state = _actor_state(runner.alg.actor)
    if args.gate_runtime_filter == "on":
      filter_on = final_eval[args.train_domain]
      filter_off = _evaluate_state(
        runner,
        final_actor_state,
        domains=(args.train_domain,),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
        runtime_filter=False,
      )[args.train_domain]
    else:
      filter_off = final_eval[args.train_domain]
      filter_on = _evaluate_state(
        runner,
        final_actor_state,
        domains=(args.train_domain,),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
        runtime_filter=True,
      )[args.train_domain]
    independent, independence_reasons = cbf_independence_gate(
      filter_on_eval=filter_on,
      filter_off_eval=filter_off,
    )
    if (
      filter_on["initial_state_signatures"]
      != filter_off["initial_state_signatures"]
    ):
      independent = False
      independence_reasons.append(
        "CBF-on/off paired initial-state signature differs"
      )
    independence_audit = {
      "passed": independent,
      "reasons": independence_reasons,
      "filter_on": filter_on,
      "filter_off": filter_off,
    }
  final_path = output_dir / "accepted_final.pt"
  result = {
    "task": task,
    "train_domain": args.train_domain,
    "neighbor_domain": args.neighbor_domain,
    "seed": args.seed,
    "train_runtime_filter": args.train_runtime_filter,
    "gate_runtime_filter": args.gate_runtime_filter,
    "counterfactual_cbf_credit": args.train_runtime_filter == "off",
    "hard_case_fraction": args.hard_case_fraction,
    "neighbor_command_fraction": args.neighbor_command_fraction,
    "neighbor_forward_scale_range": args.neighbor_forward_scale_range,
    "neighbor_delay_step_offset_range": args.neighbor_delay_step_offset_range,
    "hard_case_pre_steps": args.hard_case_pre_steps,
    "hard_case_bank_size": len(hard_case_bank),
    "hard_case_bank_total_events": hard_case_bank.total_added,
    "adaptive_std": args.adaptive_std,
    "target_intervention_per_riser": args.target_intervention_per_riser,
    "maximum_target_fall_rate": args.maximum_target_fall_rate,
    "paired_interval_method": "bootstrap",
    "fall_penalty_weight": env_cfg.rewards["fall_termination"].weight,
    "warm_start": warm_start,
    "base_checkpoint": str(args.base_checkpoint),
    "resume_hard_case_bank": args.resume_hard_case_bank,
    "critic_burn_in": burn_in,
    "critic_min_explained_variance": args.critic_min_explained_variance,
    "baseline_eval": baseline_eval,
    "rounds": rounds,
    "accepted_total_kl_from_base": accepted_total_kl,
    "final_eval": final_eval,
    "final_cbf_independence_audit": independence_audit,
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
  (output_dir / "online_refinement_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  env.close()


if __name__ == "__main__":
  main()

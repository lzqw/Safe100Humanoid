"""Run transactional DQ-to-D4 safe online refinement on one RTX GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _score(result: dict[str, Any], *, policy_drift: float = 0.0) -> float:
  demand = (
    float(result["would_intervene_per_riser"])
    if result.get("runtime_filter") is False
    else float(result["intervention_per_riser"])
  )
  return (
    float(result["success_rate"])
    + 0.02 * float(result["mean_return"])
    - 2.0 * float(result["fall_rate"])
    - 0.05 * demand
    - policy_drift
  )


def _run_stage(
  *,
  python: Path,
  repo: Path,
  base_checkpoint: Path,
  output_dir: Path,
  train_domain: str,
  neighbor_domain: str,
  num_envs: int,
  rollout_steps: int,
  critic_burn_in_rounds: int,
  online_rounds: int,
  eval_num_envs: int,
  gate_repeats: int,
  actor_learning_rate: float,
  seed: int,
  device: str,
  resume_checkpoint: Path | None = None,
) -> list[str]:
  command = [
    str(python),
    str(repo / "experiments/scripts/online_refine_stairs.py"),
    "--repo",
    str(repo),
    "--base-checkpoint",
    str(base_checkpoint),
    "--output-dir",
    str(output_dir),
    "--num-envs",
    str(num_envs),
    "--rollout-steps",
    str(rollout_steps),
    "--critic-burn-in-rounds",
    str(critic_burn_in_rounds),
    "--critic-burn-in-max-rounds",
    "4",
    "--critic-min-explained-variance",
    "0.50",
    "--online-rounds",
    str(online_rounds),
    "--eval-num-envs",
    str(eval_num_envs),
    "--eval-num-episodes",
    str(eval_num_envs),
    "--gate-repeats",
    str(gate_repeats),
    "--train-domain",
    train_domain,
    "--neighbor-domain",
    neighbor_domain,
    "--baseline-domains",
    "D0",
    train_domain,
    neighbor_domain,
    "--train-runtime-filter",
    "on",
    "--gate-runtime-filter",
    "on",
    "--hard-case-fraction",
    "0.20",
    "--neighbor-command-fraction",
    "0.15",
    "--hard-case-pre-steps",
    "10",
    "--actor-learning-rate",
    str(actor_learning_rate),
    "--critic-learning-rate",
    "1e-4",
    "--base-anchor-weight",
    "0.01",
    "--no-adaptive-std",
    "--intervention-advantage-weight",
    "0.075",
    "--pre-intervention-weight",
    "0.20",
    # In simulation, fall non-regression and the safe score remain hard gates.
    # Real-hardware-style validation can additionally set the absolute limit
    # to zero in the underlying runner.
    "--maximum-target-fall-rate",
    "1.0",
    "--seed",
    str(seed),
    "--device",
    device,
    "--gate-device",
    device,
  ]
  if resume_checkpoint is not None:
    command.extend(
      [
        "--resume-online-checkpoint",
        str(resume_checkpoint),
        "--no-resume-hard-case-bank",
      ]
    )
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "command.json").write_text(
    json.dumps(command, indent=2) + "\n"
  )
  subprocess.run(command, cwd=repo, check=True)
  return command


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--python", type=Path, default=Path(sys.executable))
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--rollout-steps", type=int, default=256)
  parser.add_argument("--critic-burn-in-rounds", type=int, default=1)
  parser.add_argument("--dq-rounds", type=int, default=5)
  parser.add_argument("--d4-rounds", type=int, default=3)
  parser.add_argument("--eval-num-envs", type=int, default=16)
  parser.add_argument("--gate-repeats", type=int, default=3)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--minimum-dq-success", type=float, default=0.60)
  parser.add_argument(
    "--closed-loop-centering",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use DQH/D4H human-like correction instead of the open-loop OOD benchmark.",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  repo = args.repo.resolve()
  output_root = args.output_root.resolve()
  output_root.mkdir(parents=True, exist_ok=True)
  dq_domain = "DQH" if args.closed_loop_centering else "DQ"
  dqn_domain = "DQNH" if args.closed_loop_centering else "DQN"
  d4_domain = "D4H" if args.closed_loop_centering else "D4"
  d5_domain = "D5H" if args.closed_loop_centering else "D5"
  dq_output = output_root / f"stage1_{dq_domain.lower()}"
  dq_command = _run_stage(
    python=args.python.resolve(),
    repo=repo,
    base_checkpoint=args.base_checkpoint.resolve(),
    output_dir=dq_output,
    train_domain=dq_domain,
    neighbor_domain=dqn_domain,
    num_envs=args.num_envs,
    rollout_steps=args.rollout_steps,
    critic_burn_in_rounds=args.critic_burn_in_rounds,
    online_rounds=args.dq_rounds,
    eval_num_envs=args.eval_num_envs,
    gate_repeats=args.gate_repeats,
    actor_learning_rate=args.actor_learning_rate,
    seed=args.seed,
    device=args.device,
  )
  dq = json.loads((dq_output / "online_refinement_summary.json").read_text())
  accepted_rounds = sum(bool(round_["accepted"]) for round_ in dq["rounds"])
  baseline = dq["baseline_eval"][dq_domain]
  final = dq["final_eval"][dq_domain]
  final_kl = float(dq["accepted_total_kl_from_base"])
  dq_gate = {
    "accepted_rounds": accepted_rounds,
    "minimum_success": args.minimum_dq_success,
    "baseline_success": float(baseline["success_rate"]),
    "final_success": float(final["success_rate"]),
    "baseline_safe_score": _score(baseline),
    "final_safe_score": _score(final, policy_drift=final_kl),
  }
  dq_gate["passed"] = (
    accepted_rounds > 0
    and dq_gate["final_success"] >= args.minimum_dq_success
    and dq_gate["final_safe_score"] > dq_gate["baseline_safe_score"]
  )

  result: dict[str, Any] = {
    "base_checkpoint": str(args.base_checkpoint.resolve()),
    "stage1": {
      "output": str(dq_output),
      "command": dq_command,
      "gate": dq_gate,
      "target_domain": dq_domain,
      "neighbor_domain": dqn_domain,
    },
    "stage2": None,
  }
  if dq_gate["passed"]:
    d4_output = output_root / f"stage2_{d4_domain.lower()}"
    resume = dq_output / "accepted_final.pt"
    d4_command = _run_stage(
      python=args.python.resolve(),
      repo=repo,
      base_checkpoint=args.base_checkpoint.resolve(),
      output_dir=d4_output,
      train_domain=d4_domain,
      neighbor_domain=d5_domain,
      num_envs=args.num_envs,
      rollout_steps=args.rollout_steps,
      critic_burn_in_rounds=args.critic_burn_in_rounds,
      online_rounds=args.d4_rounds,
      eval_num_envs=args.eval_num_envs,
      gate_repeats=args.gate_repeats,
      actor_learning_rate=args.actor_learning_rate,
      seed=args.seed,
      device=args.device,
      resume_checkpoint=resume,
    )
    result["stage2"] = {
      "started": True,
      "output": str(d4_output),
      "resume_checkpoint": str(resume),
      "hard_case_bank_reset_for_domain_change": True,
      "command": d4_command,
      "target_domain": d4_domain,
      "neighbor_domain": d5_domain,
    }
  else:
    result["stage2"] = {
      "started": False,
      "reason": "DQ safe-improvement gate did not pass",
    }
  (output_root / "staged_refinement_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()

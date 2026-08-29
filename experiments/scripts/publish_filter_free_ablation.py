"""Publish compact formal filter-free results and representative models."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ARMS = ("nominal_ft", "reward_only_ft", "filter_only_ft", "dual_safe_ft")
CONTEXTS = ("F1", "F2", "F3")
TRAINING_SEEDS = (201357000, 201357001, 201357002)
REPRESENTATIVE_MODEL_SEED = TRAINING_SEEDS[0]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument(
    "--result-relative-dir",
    type=Path,
    default=Path("results/online/paper_filter_free_ablation_v140"),
  )
  parser.add_argument("--branch", default="feature/online-safe-refinement")
  parser.add_argument("--push", action="store_true")
  return parser.parse_args()


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
  if not source.is_file():
    raise FileNotFoundError(source)
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, destination)


def _copy_named_summaries(source_root: Path, destination_root: Path) -> int:
  count = 0
  for source in sorted(source_root.rglob("summary.json")):
    _copy(source, destination_root / source.relative_to(source_root))
    count += 1
  return count


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  training_root = args.training_root.resolve()
  if args.result_relative_dir.is_absolute() or ".." in args.result_relative_dir.parts:
    raise ValueError("result directory must be a safe repository-relative path")
  result_dir = repo / args.result_relative_dir
  if result_dir.exists():
    raise FileExistsError(result_dir)
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("publishing requires a clean worktree")
  if _git(repo, "branch", "--show-current") != args.branch:
    raise RuntimeError("publishing branch differs from the requested branch")

  evaluation_root = training_root / "evaluation_seed201357900"
  proxy_root = training_root / "hardware_proxy_seed201357901"
  final_root = training_root / "final_summary"
  required_status = (
    training_root / "training_progress.json",
    evaluation_root / "evaluation_progress.json",
    proxy_root / "proxy_progress.json",
  )
  for path in required_status:
    if not path.is_file() or json.loads(path.read_text()).get("status") != "complete":
      raise RuntimeError(f"formal stage is not complete: {path}")
  final_results_path = final_root / "final_results.json"
  if not final_results_path.is_file():
    raise FileNotFoundError(final_results_path)

  result_dir.mkdir(parents=True)
  for name in (
    "training_manifest.json",
    "training_progress.json",
  ):
    _copy(training_root / name, result_dir / "manifests" / name)
  for name in (
    "evaluation_manifest.json",
    "evaluation_progress.json",
    "paired_checkpoint_results.json",
  ):
    _copy(evaluation_root / name, result_dir / "evaluation" / name)
  for name in (
    "final_results.json",
    "main_table.csv",
    "learning_curves.csv",
    "training_safety.csv",
    "SUMMARY.md",
  ):
    _copy(final_root / name, result_dir / name)
  for name in (
    "proxy_manifest.json",
    "proxy_progress.json",
    "hardware_proxy_results.json",
    "hardware_proxy_main_table.csv",
    "proxy_policy_results.csv",
  ):
    _copy(proxy_root / name, result_dir / "hardware_proxy" / name)

  for arm in ARMS:
    for context in CONTEXTS:
      for seed in TRAINING_SEEDS:
        source = training_root / context / arm / f"seed_{seed}"
        destination = result_dir / "training" / context / arm / f"seed_{seed}"
        for name in ("training_summary.json", "round_metrics.json", "round_metrics.csv"):
          _copy(source / name, destination / name)

  evaluation_summary_count = _copy_named_summaries(
    evaluation_root,
    result_dir / "evaluation" / "checkpoint_summaries",
  )
  proxy_summary_count = _copy_named_summaries(
    proxy_root,
    result_dir / "hardware_proxy" / "policy_summaries",
  )
  if evaluation_summary_count != 222 or proxy_summary_count != 78:
    raise RuntimeError(
      "formal summary count differs: "
      f"paired={evaluation_summary_count}, proxy={proxy_summary_count}"
    )

  checkpoint_index: list[dict[str, Any]] = []
  for arm in ARMS:
    for context in CONTEXTS:
      for seed in TRAINING_SEEDS:
        source = (
          training_root / context / arm / f"seed_{seed}" / "round_04.pt"
        )
        summary = json.loads(
          (
            training_root
            / context
            / arm
            / f"seed_{seed}"
            / "training_summary.json"
          ).read_text()
        )
        published = arm == "dual_safe_ft" and seed == REPRESENTATIVE_MODEL_SEED
        destination = None
        if published:
          destination_path = (
            result_dir
            / "checkpoints"
            / f"dual_safe_ft_{context}_seed_{seed}_round_04.pt"
          )
          _copy(source, destination_path)
          destination = str(destination_path.relative_to(repo))
        checkpoint_index.append(
          {
            "arm": arm,
            "context": context,
            "training_seed": seed,
            "round": 4,
            "checkpoint_sha256": _sha256(source),
            "actor_sha256": summary["final_actor_sha256"],
            "published_to_repository": published,
            "repository_path": destination,
            "selection_used": False,
          }
        )
  (result_dir / "checkpoint_index.json").write_text(
    json.dumps(checkpoint_index, indent=2, sort_keys=True) + "\n"
  )

  final_results = json.loads(final_results_path.read_text())
  proxy_results = json.loads(
    (proxy_root / "hardware_proxy_results.json").read_text()
  )
  table_lines = [
    "| Arm | Mean CBF-off | Improvement | Shield gap | Training falls |",
    "|---|---:|---:|---:|---:|",
  ]
  for row in final_results["main_table"]:
    falls = (
      "—"
      if row["training_falls_mean"] is None
      else f"{float(row['training_falls_mean']):.1f}"
    )
    table_lines.append(
      "| {arm} | {off:.3f} | {gain:+.3f} | {gap:+.3f} | {falls} |".format(
        arm=row["arm"],
        off=float(row["mean_cbf_off_success_rate"]),
        gain=float(row["mean_off_improvement"]),
        gap=float(row["mean_shield_gap"]),
        falls=falls,
      )
    )
  proxy_lines = [
    "| Arm | Hardware-proxy CBF-off success | Fall rate |",
    "|---|---:|---:|",
  ]
  for row in proxy_results["aggregate"]:
    proxy_lines.append(
      f"| {row['arm']} | {float(row['mean_success_rate']):.3f} | "
      f"{float(row['mean_fall_rate']):.3f} |"
    )
  readme = [
    "# Filter-free deployment ablation (v140)",
    "",
    "Formal simulation ablation from the frozen v139 actor. Adaptation uses "
    "four rounds × 128 environments × 1024 steps, three preregistered seeds, "
    "and fixed round-4 publication. Primary deployment evaluation is "
    "deterministic CBF-off.",
    "",
    *table_lines,
    "",
    f"Main claim supported: **{final_results['main_claim_supported']}**.",
    "",
    "## Hardware proxy", 
    "",
    *proxy_lines,
    "",
    "The proxy combines actor sensor noise/encoder bias, 1- and 2-step action "
    "delay, 0.95 actuator gain, +1 cm stair-height estimate bias, ±1.5 cm "
    "tread perturbation, friction variation, and command delay. All proxy "
    "evaluations execute CBF-off.",
    "",
    "## Contents",
    "",
    "- `final_results.json` and `main_table.csv`: primary result and claim checks",
    "- `learning_curves.csv`: rounds 0/1/2/4",
    "- `training_safety.csv`: falls, nominal/executed violations, recoveries, min h",
    "- `evaluation/checkpoint_summaries/`: all 222 paired-condition summaries",
    "- `hardware_proxy/`: 78 proxy summaries and aggregate table",
    "- `training/`: all 36 training summaries and round metrics",
    "- `checkpoints/`: fixed round-4 Dual Safe-FT models for F1/F2/F3, seed 201357000",
    "- `checkpoint_index.json`: hashes for all 36 fixed round-4 models",
    "",
    "Episode CSVs and redundant checkpoints remain in the archived 4080 run; "
    "the repository contains the compact evidence needed to reproduce tables.",
  ]
  (result_dir / "README.md").write_text("\n".join(readme) + "\n")

  _git(repo, "add", str(args.result_relative_dir))
  _git(repo, "commit", "-m", "Publish formal filter-free ablation results")
  commit = _git(repo, "rev-parse", "HEAD")
  if args.push:
    _git(repo, "push", "origin", args.branch)
  print(
    json.dumps(
      {
        "result_dir": str(result_dir),
        "git_commit": commit,
        "pushed": args.push,
        "evaluation_summary_count": evaluation_summary_count,
        "proxy_summary_count": proxy_summary_count,
        "representative_model_count": 3,
      },
      indent=2,
      sort_keys=True,
    )
  )


if __name__ == "__main__":
  main()

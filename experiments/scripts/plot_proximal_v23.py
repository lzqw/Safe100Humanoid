"""Create the two preregistered v23 result figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(figure, stem: Path) -> list[str]:
  stem.parent.mkdir(parents=True, exist_ok=True)
  outputs = []
  for suffix in (".png", ".pdf"):
    path = stem.with_suffix(suffix)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    outputs.append(str(path))
  plt.close(figure)
  return outputs


def _round_curve(training: dict):
  rounds = training["rounds"]
  x = np.asarray([row["round"] for row in rounds])
  success = np.asarray(
    [row["metrics"].get("rollout_success_rate", np.nan) for row in rounds]
  )
  fall = np.asarray(
    [row["metrics"].get("rollout_fall_rate", np.nan) for row in rounds]
  )
  moving_kl = np.asarray(
    [row["metrics"].get("moving_forward_kl", np.nan) for row in rounds]
  )
  clip_fraction = np.asarray(
    [row["metrics"].get("clip_fraction", np.nan) for row in rounds]
  )
  actor_loss = np.asarray(
    [row["metrics"].get("actor_loss", np.nan) for row in rounds]
  )
  value_loss = np.asarray(
    [row["metrics"].get("value", np.nan) for row in rounds]
  )
  figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.3))
  axes[0].plot(x, success, marker="o", label="rollout success")
  axes[0].plot(x, fall, marker="s", label="rollout fall")
  axes[0].set(ylabel="rate", ylim=(0.0, 1.0))
  axes[0].legend(frameon=False, fontsize=8)
  axes[1].plot(x, moving_kl, marker="o", label="forward moving KL")
  axes[1].plot(x, clip_fraction, marker="s", label="clip fraction")
  axes[1].axhline(0.003, color="tab:gray", linestyle="--", label="target KL")
  axes[1].axhline(0.01, color="tab:red", linestyle=":", label="hard ceiling")
  axes[1].set(yscale="log", ylabel="policy diagnostic")
  axes[1].legend(frameon=False, fontsize=7)
  axes[2].plot(x, actor_loss, marker="o", label="actor loss")
  axes[2].plot(x, value_loss, marker="s", label="value loss")
  axes[2].set(ylabel="loss")
  axes[2].legend(frameon=False, fontsize=8)
  for axis in axes:
    axis.set(xlabel="online round", xticks=x)
    axis.grid(alpha=0.25)
  figure.suptitle("CBF-Proximal PPO v23: fixed eight-round trajectory")
  figure.tight_layout()
  return figure


def _base_vs_final(final_test: dict):
  labels = ("Target success", "Target fall", "D0 success", "D0 fall")
  baseline = (
    final_test["target"]["success"]["baseline_mean"],
    final_test["target"]["fall"]["baseline_mean"],
    final_test["D0"]["success"]["baseline_mean"],
    final_test["D0"]["fall"]["baseline_mean"],
  )
  final = (
    final_test["target"]["success"]["final_mean"],
    final_test["target"]["fall"]["final_mean"],
    final_test["D0"]["success"]["final_mean"],
    final_test["D0"]["fall"]["final_mean"],
  )
  x = np.arange(len(labels))
  width = 0.36
  figure, axis = plt.subplots(figsize=(8.2, 3.8))
  axis.bar(x - width / 2, baseline, width, label="base")
  axis.bar(x + width / 2, final, width, label="round 8")
  axis.set(
    ylabel="paired rate",
    ylim=(0.0, 1.0),
    xticks=x,
    xticklabels=labels,
    title="Base vs fixed round-8 policy (runtime CBF enabled)",
  )
  axis.grid(axis="y", alpha=0.25)
  axis.legend(frameon=False)
  figure.tight_layout()
  return figure


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--training-summary", type=Path, required=True)
  parser.add_argument("--final-test", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  training = json.loads(args.training_summary.read_text())
  final_test = json.loads(args.final_test.read_text())
  output_dir = args.output_dir.resolve()
  outputs = []
  outputs.extend(_save(_round_curve(training), output_dir / "round_curve"))
  outputs.extend(
    _save(_base_vs_final(final_test), output_dir / "base_vs_final")
  )
  manifest = {
    "figure_count": 2,
    "files": outputs,
    "selection_or_gate_use": False,
  }
  (output_dir / "figure_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
